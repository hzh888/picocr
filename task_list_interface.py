# coding:utf-8
import os
import cv2
import time
import json
import ddddocr
from natsort import natsorted
from paddleocr import PaddleOCR
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Border, Side, Alignment
from openpyxl.utils import get_column_letter
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidgetItem, QHeaderView, QSpacerItem, QSizePolicy,
    QHBoxLayout, QLabel
)
from qfluentwidgets import (
    TableWidget, PushButton, ProgressRing, InfoBar, InfoBarPosition, TextEdit, MessageBox
)
import numpy as np
import requests
import base64
import urllib.parse  # 导入urllib.parse用于URL编码
from pymediainfo import MediaInfo
from datetime import datetime, timedelta
import pytz
from requests.exceptions import RequestException  # 导入RequestException

def load_auth_code(log_signal):
    """从config.json文件加载AuthCode"""
    try:
        if not os.path.exists('config.json'):
            log_signal.emit("config.json配置文件获取失败")
            return None

        with open('config.json', 'r', encoding='utf-8') as config_file:
            config_data = json.load(config_file)
            auth_code = config_data.get("Authorization", {}).get("AuthCode", "")
            if not auth_code:
                log_signal.emit("未配置云端OCR授权码")
                return None
            return auth_code
    except (json.JSONDecodeError, KeyError):
        log_signal.emit("未配置云端OCR授权码")
        return None

def get_baidu_credentials(log_signal):
    """从config.json文件加载百度OCR的API_KEY、SECRET_KEY和URL"""
    try:
        if not os.path.exists('config.json'):
            log_signal.emit("config.json配置文件获取失败")
            return None, None, None

        with open('config.json', 'r', encoding='utf-8') as config_file:
            config_data = json.load(config_file)
            api_key = config_data.get("BaiduOCR", {}).get("ApiKey", "")
            secret_key = config_data.get("BaiduOCR", {}).get("SecretKey", "")
            url = config_data.get("BaiduOCR", {}).get("ApiUrl", "")

            if not api_key:
                log_signal.emit("未配置百度OCR的API_KEY")
            if not secret_key:
                log_signal.emit("未配置百度OCR的SECRET_KEY")
            if not url:
                log_signal.emit("未配置百度OCR的ApiUrl")

            if not api_key or not secret_key or not url:
                return None, None, None
            return api_key, secret_key, url
    except (json.JSONDecodeError, KeyError):
        log_signal.emit("读取config.json配置时发生错误")
        return None, None, None

def get_access_token(api_key, secret_key, log_signal):
    """使用API_KEY和SECRET_KEY获取百度OCR的access_token"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    try:
        response = requests.post(url, params=params)
        response.raise_for_status()  # 如果返回错误状态码，抛出异常
        return str(response.json().get("access_token"))
    except RequestException:
        log_signal.emit("网络连接失败")
        return None

def replace_text_in_result(result, replace_option, replace_text):
    if replace_option == "替换文本":
        replacements = replace_text.split(",")
        for replacement in replacements:
            original, new_text = replacement.split("=")
            result = result.replace(original, new_text)
    elif replace_option == "去掉文本":
        remove_texts = replace_text.split(",")
        for text in remove_texts:
            result = result.replace(text, "")
    return result

class ImageProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    task_complete_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object)
    image_visualization_signal = pyqtSignal(str, str)

    def __init__(self, task, stop_flag):
        super().__init__()
        self.task = task
        self.stop_flag = stop_flag
        self.ocr_running = False
        self.ocr_model = None

    def run(self):
        try:
            task_name = self.task['task_name']
            imagestorage_dir = os.path.join('Imagestorage', task_name)
            frame_interval = int(self.task['frame_interval'])
            grayscale_option = self.task['grayscale_option']

            if os.path.exists(imagestorage_dir):
                self._delete_folder(imagestorage_dir)
                self.log_signal.emit(f"已删除 '{task_name}' 旧文件夹")

            os.makedirs(imagestorage_dir)
            self.log_signal.emit(f"开始执行 '{os.path.basename(self.task['file_path'])}' 转换图片")

            file_path = self.task['file_path']
            if not os.path.exists(file_path):
                self.log_signal.emit(f"'{os.path.basename(file_path)}' 不存在，任务停止")
                return

            recognition_image_paths = self.task['recognition_image_path'].split(",")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.log_signal.emit("无法打开视频文件")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            total_steps = int(duration // frame_interval) * len(recognition_image_paths) * 2

            step_count = 0

            for img_name in recognition_image_paths:
                if self.stop_flag['stop']:
                    self.log_signal.emit("任务已停止")
                    return
                img_name = img_name.strip()
                folder_name = img_name.split('.')[0]
                folder_path = os.path.join(imagestorage_dir, folder_name)

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                self.log_signal.emit(f"开始截取 '{img_name}' 图片")
                full_path = os.path.join("Identifyimages", task_name, img_name)
                if not os.path.exists(full_path):
                    self.log_signal.emit(f"'{img_name}' 图片不存在，任务停止")
                    return

                json_path = os.path.join("Identifyimages", task_name, "rectangles.json")
                if not os.path.exists(json_path):
                    self.log_signal.emit(f"'{img_name}' 的 rectangles.json 文件不存在，任务停止")
                    return

                with open(json_path, 'r') as json_file:
                    rectangles = json.load(json_file)

                second_count = 0
                while second_count < duration:
                    if self.stop_flag['stop']:
                        self.log_signal.emit("任务已停止")
                        return
                    cap.set(cv2.CAP_PROP_POS_MSEC, second_count * 1000)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    for rect in rectangles:
                        if rect["index"] == recognition_image_paths.index(img_name) + 1:
                            x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
                            crop_image = frame[y:y + h, x:x + w]

                            if grayscale_option in ["转换灰度图像", "转换灰度图像二值化", "转换灰度图像(固定二值化)"]:
                                crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

                            if grayscale_option == "转换灰度图像二值化":
                                crop_image = cv2.adaptiveThreshold(
                                    crop_image, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    11, 2
                                )

                            elif grayscale_option == "转换灰度图像(固定二值化)":
                                _, crop_image = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY)

                            frame_name = os.path.join(folder_path, f"{task_name}_{folder_name}_{second_count:02d}.png")
                            cv2.imencode('.png', crop_image)[1].tofile(frame_name)
                            self.log_signal.emit(f"保存图片: {frame_name}")

                            self.image_visualization_signal.emit(task_name, frame_name)

                    second_count += frame_interval
                    step_count += 1
                    self.progress_signal.emit(int((step_count / total_steps) * 100))

                self.log_signal.emit(f"'{img_name}' 截取完毕")

            cap.release()
            self.log_signal.emit("截取图片完毕，正在执行识别初始化...")

            self.ocr_running = True
            self.run_ocr(task_name, recognition_image_paths, total_steps)

            if not self.stop_flag['stop']:
                self.task_complete_signal.emit(task_name)

        except Exception as e:
            self.log_signal.emit(f"处理过程中出现错误: {e}")

        finally:
            self.release_model()

    def initialize_model(self):
        model = self.task['model']
        if model == "Ddddocr模型":
            self.ocr_model = ddddocr.DdddOcr()
        elif model == "Paddle模型":
            self.ocr_model = PaddleOCR(
                det_model_dir='./models/whl/det/ch/ch_PP-OCRv4_det_infer/',
                rec_model_dir='./models/whl/rec/ch/ch_PP-OCRv4_rec_infer/',
                cls_model_dir='./models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/',
                lang='ch', use_angle_cls=True, use_gpu=False
            )

    def release_model(self):
        if self.ocr_model:
            del self.ocr_model
            self.ocr_model = None

    def run_ocr(self, task_name, recognition_image_paths, total_steps):
        try:
            self.initialize_model()

            model = self.task['model']
            replace_option = self.task['replace_option']
            replace_text = self.task['replace_text']
            imagestorage_dir = os.path.join('Imagestorage', task_name)

            step_count = total_steps // 2
            ocr_results = {}

            if model == "百度OCR(联网)":
                api_key, secret_key, base_url = get_baidu_credentials(self.log_signal)
                if not api_key or not secret_key or not base_url:
                    self.stop_flag['stop'] = True
                    return

                access_token = get_access_token(api_key, secret_key, self.log_signal)
                if not access_token:
                    self.stop_flag['stop'] = True
                    return

                url = base_url + access_token

            else:
                auth_code = load_auth_code(self.log_signal)
                if not auth_code:
                    self.stop_flag['stop'] = True
                    return

            for img_name in recognition_image_paths:
                if self.stop_flag['stop']:
                    self.log_signal.emit("OCR任务已停止")
                    self.result_signal.emit((task_name, ocr_results))
                    return
                img_name = img_name.strip()
                self.log_signal.emit(f"识别 '{img_name}' 开始")
                full_path = os.path.join("Identifyimages", task_name, img_name)
                if not os.path.exists(full_path):
                    self.log_signal.emit(f"'{img_name}' 图片不存在，任务停止")
                    self.result_signal.emit((task_name, ocr_results))
                    return

                folder_name = img_name.split('.')[0]
                folder_path = os.path.join(imagestorage_dir, folder_name)
                if not os.path.exists(folder_path):
                    self.log_signal.emit(f"'{folder_name}' 文件夹不存在，任务停止")
                    self.result_signal.emit((task_name, ocr_results))
                    return

                ocr_results[img_name] = []

                image_files = natsorted(os.listdir(folder_path))

                for image_filename in image_files:
                    if self.stop_flag['stop']:
                        self.log_signal.emit("OCR任务已停止")
                        self.result_signal.emit((task_name, ocr_results))
                        return
                    img_path = os.path.join(folder_path, image_filename)
                    with open(img_path, "rb") as image_file:
                        image_data = image_file.read()

                    result = ""  # Initialize result
                    if model == "百度OCR(联网)":
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        image_base64_urlencoded = urllib.parse.quote(image_base64, safe='')

                        payload = f'image={image_base64_urlencoded}&detect_direction=false&detect_language=false&paragraph=false&probability=false'
                        headers = {
                            'Content-Type': 'application/x-www-form-urlencoded',
                            'Accept': 'application/json'
                        }

                        retry = True
                        while retry:
                            if self.stop_flag['stop']:
                                self.log_signal.emit("OCR任务已停止")
                                return
                            try:
                                response = requests.request("POST", url, headers=headers, data=payload)

                                if response.status_code == 200:
                                    result_json = response.json()
                                    error_code = result_json.get("error_code")

                                    if error_code == 18:
                                        self.log_signal.emit("触发限流，2秒后再次识别")
                                        time.sleep(2)
                                        retry = True
                                    elif error_code == 17:
                                        result = "识别次数不足"
                                        retry = False
                                    else:
                                        if "words_result" in result_json and result_json["words_result"]:
                                            result = "".join([item["words"] for item in result_json["words_result"]])
                                        else:
                                            result = "识别失败"
                                        retry = False
                                else:
                                    result = "识别失败"
                                    retry = False
                            except RequestException:
                                self.log_signal.emit("网络连接失败")
                                self.stop_flag['stop'] = True
                                return

                    elif model == "云端OCR(联网)":
                        image_base64 = base64.b64encode(image_data).decode('utf-8')

                        retry = True
                        while retry:
                            if self.stop_flag['stop']:
                                self.log_signal.emit("OCR任务已停止")
                                return
                            try:
                                response = requests.post(
                                    "https://api.npcbug.com/ocr/tyocr.php",
                                    data={'base64': image_base64, 'auth': auth_code}
                                )

                                if response.status_code == 200:
                                    result_json = response.json()

                                    if result_json.get("error_code") == 403:
                                        self.log_signal.emit("授权码不存在或可用次数不足")
                                        self.stop_flag['stop'] = True
                                        return

                                    if result_json.get("error_code") == 18:
                                        self.log_signal.emit("触发限流，2秒后再次识别")
                                        time.sleep(2)
                                        retry = True
                                    elif result_json.get("error_code") == 17:
                                        result = "识别次数不足"
                                        retry = False
                                    else:
                                        if "words_result" in result_json and result_json["words_result"]:
                                            result = "".join([item["words"] for item in result_json["words_result"]])
                                        else:
                                            result = "识别失败"
                                        retry = False
                                else:
                                    result = "识别失败"
                                    retry = False
                            except RequestException:
                                self.log_signal.emit("网络连接失败")
                                self.stop_flag['stop'] = True
                                return

                    else:
                        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            result = "识别失败"
                            ocr_results[img_name].append((image_filename, result))
                            continue

                        if model == "Ddddocr模型":
                            result = self.ocr_model.classification(image_data)
                        elif model == "Paddle模型":
                            paddle_result = self.ocr_model.ocr(img_path, cls=True)
                            if paddle_result and paddle_result[0]:
                                result = ''.join([line[1][0] for line in paddle_result[0]])
                            else:
                                result = "识别失败"

                    if not result:
                        result = "识别失败"

                    result = replace_text_in_result(result, replace_option, replace_text)
                    ocr_results[img_name].append((image_filename, result))
                    self.log_signal.emit(f"图片 {image_filename} 的识别结果: {result}")

                    self.image_visualization_signal.emit(task_name, img_path)

                    # 根据模型设置不同的延迟时间
                    if model == "百度OCR(联网)":
                        time.sleep(1.5)  # 1.5秒延迟
                    elif model == "云端OCR(联网)" or model == "Ddddocr模型" or model == "Paddle模型":
                        time.sleep(0.2)  # 200ms延迟

                    step_count += 1
                    self.progress_signal.emit(int((step_count / total_steps) * 100))

                self.log_signal.emit(f"识别 '{img_name}' 结束")

            self.result_signal.emit((task_name, ocr_results))
            self.progress_signal.emit(100)
            self.log_signal.emit("任务结束")

        except Exception as e:
            self.log_signal.emit(f"处理过程中出现错误: {e}")

        finally:
            self.release_model()


    def stop(self):
        self.stop_flag['stop'] = True
        self.log_signal.emit("任务手动停止")
        self.release_model()  # Ensure model is released when manually stopping

    def export_results(self, task_name, ocr_results):
        try:
            if self.task['export_option'] == "不导出表格":
                self.log_signal.emit(f"{task_name}任务完成，不导出表格")
                return

            if not os.path.exists('excel'):
                os.makedirs('excel')
            output_path = os.path.join('excel', f"{task_name}_识别结果.xlsx")

            if self.is_file_open(output_path):
                self.log_signal.emit(f"请关闭 {output_path} 文件后再次执行任务")
                return

            workbook = Workbook()

            # 获取任务相关的信息
            model = self.task['model']
            replace_option = self.task['replace_option']
            replace_text = self.task['replace_text']
            frame_interval = int(self.task['frame_interval'])
            grayscale_option = self.task['grayscale_option']
            file_path = self.task['file_path']

            # 获取视频创建时间并转换为datetime对象
            video_creation_time = self.get_video_creation_time(file_path)

            for img_name, results in ocr_results.items():
                sheet_name = img_name.split('.')[0]
                worksheet = workbook.create_sheet(title=sheet_name)

                # 新增表头，调整列顺序，将“图片时间点”放在“原图”后面
                worksheet.append([
                    "图片文件", "识别结果", "原图", "媒体时间点", "图片秒数点",
                    "识别模型", "替换选项", "替换文本", "提取间隔", "灰度图像选项"
                ])

                thin_border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )

                # 创建对齐对象，设置水平和垂直居中
                center_alignment = Alignment(horizontal='center', vertical='center')

                for r_idx, (image_file, result) in enumerate(results, start=2):
                    worksheet.cell(row=r_idx, column=1, value=image_file)
                    worksheet.cell(row=r_idx, column=2, value=result)

                    # 提取图片的秒数点，假设文件名中包含了秒数信息
                    try:
                        # 从文件名中提取秒数点，例如文件名格式为 "taskname_foldername_10.png"
                        # 其中的 "10" 是秒数点
                        second_point = int(image_file.split('_')[-1].split('.')[0])
                    except ValueError:
                        second_point = "未知秒数"

                    worksheet.cell(row=r_idx, column=5, value=second_point)

                    # 在每一行中加入这些信息
                    worksheet.cell(row=r_idx, column=6, value=model)
                    worksheet.cell(row=r_idx, column=7, value=replace_option)
                    worksheet.cell(row=r_idx, column=8, value=replace_text)
                    worksheet.cell(row=r_idx, column=9, value=frame_interval)
                    worksheet.cell(row=r_idx, column=10, value=grayscale_option)

                    # 提取图片的时间点并计算真实时间
                    image_time_point = self.calculate_image_time(video_creation_time, image_file)
                    worksheet.cell(row=r_idx, column=4, value=image_time_point)  # 调整列位置

                    img_path = os.path.join('Imagestorage', task_name, sheet_name, image_file)
                    img = Image(img_path)

                    img_width, img_height = img.width, img.height
                    col_letter = get_column_letter(3)
                    worksheet.column_dimensions[col_letter].width = img_width * 0.14
                    worksheet.row_dimensions[r_idx].height = img_height * 0.75

                    img.anchor = f'C{r_idx}'
                    worksheet.add_image(img)

                    for col in range(1, 11):  # Update range to include new column
                        cell = worksheet.cell(row=r_idx, column=col)
                        cell.border = thin_border
                        cell.alignment = center_alignment  # 设置对齐方式

                # 设置表头的边框和对齐方式
                for col in range(1, 11):  # Update range to include new column
                    header_cell = worksheet.cell(row=1, column=col)
                    header_cell.border = thin_border
                    header_cell.alignment = center_alignment  # 设置对齐方式

            del workbook['Sheet']
            workbook.save(output_path)
            self.log_signal.emit(f"识别结果已导出至: {output_path}")
        except Exception as e:
            self.log_signal.emit(f"导出结果时出现错误: {e}")

    def get_video_creation_time(self, file_path):
        """Get the creation time of the video file."""
        media_info = MediaInfo.parse(file_path)
        for track in media_info.tracks:
            if track.track_type == 'General':
                # 获取创建时间
                creation_time = track.tagged_date or track.recorded_date
                if creation_time:
                    try:
                        # 尝试解析格式为 '2024-06-13 02:03:13 UTC'
                        dt = datetime.strptime(creation_time, "%Y-%m-%d %H:%M:%S %Z")
                        # 将其转换为中国标准时间
                        return dt.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Shanghai'))
                    except ValueError:
                        self.log_signal.emit(f"无法解析视频的创建时间: {creation_time}")
        return None

    def calculate_image_time(self, video_creation_time, image_file):
        if video_creation_time is None:
            # 如果创建时间获取失败，则返回“获取失败”
            return "获取失败"
        try:
            parts = image_file.split('_')
            actual_second = int(parts[-1].split('.')[0])

            image_time_point = video_creation_time + timedelta(seconds=actual_second)

            return image_time_point.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            self.log_signal.emit(f"计算图片时间点时出现错误: {e}")
            return "时间计算错误"

    def is_file_open(self, file_path):
        if os.path.exists(file_path):
            try:
                os.rename(file_path, file_path)
                return False
            except OSError:
                return True
        return False

    def _delete_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)


class TaskListInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('TaskListInterface')

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.table_widget = TableWidget(self)

        self.table_widget.setBorderVisible(True)
        self.table_widget.setBorderRadius(8)

        self.table_widget.setWordWrap(False)
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(9)
        self.table_widget.setHorizontalHeaderLabels(
            ['任务名', '识别模型', '导出选项', '视频路径', '识别图片路径', '替换选项', '替换文本', '提取间隔',
             '灰度图像选项'])
        self.table_widget.verticalHeader().hide()

        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setColumnWidth(0, 100)
        self.table_widget.setColumnWidth(1, 120)
        self.table_widget.setColumnWidth(2, 120)
        self.table_widget.setColumnWidth(3, 120)
        self.table_widget.setColumnWidth(4, 120)
        self.table_widget.setColumnWidth(5, 120)
        self.table_widget.setColumnWidth(6, 100)
        self.table_widget.setColumnWidth(7, 100)
        self.table_widget.setColumnWidth(8, 120)

        self.main_layout.addWidget(self.table_widget)

        self.progress_ring = ProgressRing(self)
        self.progress_ring.setTextVisible(True)
        self.progress_ring.setFormat("0%")
        self.progress_ring.setStrokeWidth(10)
        self.progress_ring.setFixedSize(130, 130)

        self.start_button = PushButton('开始执行', self)
        self.start_button.clicked.connect(self.start_task)
        self.stop_button = PushButton('停止任务', self)
        self.stop_button.clicked.connect(self.stop_task)
        self.delete_button = PushButton('删除任务', self)
        self.delete_button.clicked.connect(self.delete_selected_row)
        self.clear_log_button = PushButton('清空日志', self)
        self.clear_log_button.clicked.connect(self.clear_log)

        self.log_text = TextEdit(self)
        self.log_text.setReadOnly(True)

        # 调整 QLabel 的大小，并美化边框
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(150, 130)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 1.3px solid #0078D7;
            border-radius: 5px;
            background-color: #f5f5f5;
        """)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_log_button)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        control_layout.addWidget(self.start_button)
        control_layout.addLayout(button_layout)

        progress_button_layout = QHBoxLayout()
        progress_button_layout.addSpacerItem(
            QSpacerItem(12, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        progress_button_layout.addWidget(self.progress_ring)
        progress_button_layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        progress_button_layout.addLayout(control_layout)
        progress_button_layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        progress_log_layout = QHBoxLayout()
        progress_log_layout.addLayout(progress_button_layout)
        progress_log_layout.addWidget(self.log_text)

        progress_log_layout.addSpacerItem(
            QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )

        progress_log_layout.addWidget(self.image_label)

        progress_log_layout.addSpacerItem(
            QSpacerItem(12, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        )

        self.main_layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.main_layout.addLayout(progress_log_layout)
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.main_layout.setStretch(0, 5)
        self.main_layout.setStretch(2, 1)

        self.table_widget.itemSelectionChanged.connect(self.on_task_selected)

        self.threads = {}
        self.stop_flags = {}
        self.task_logs = {}
        self.task_progress = {}
        self.task_results = {}
        self.task_images = {}

    def add_task(self, task_name, model, export_option, file_path, replace_option, image_path, replace_text,
                 frame_interval, grayscale_option):
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

        self.table_widget.setItem(row_position, 0, self.create_non_editable_item(task_name))
        self.table_widget.setItem(row_position, 1, self.create_non_editable_item(model))
        self.table_widget.setItem(row_position, 2, self.create_non_editable_item(export_option))
        self.table_widget.setItem(row_position, 3, self.create_non_editable_item(file_path))
        self.table_widget.setItem(row_position, 4, self.create_non_editable_item(image_path))
        self.table_widget.setItem(row_position, 5, self.create_non_editable_item(replace_option))
        self.table_widget.setItem(row_position, 6, self.create_non_editable_item(replace_text))
        self.table_widget.setItem(row_position, 7, self.create_non_editable_item(frame_interval))
        self.table_widget.setItem(row_position, 8, self.create_non_editable_item(grayscale_option))

        self.task_logs[task_name] = []
        self.task_progress[task_name] = 0
        self.task_results[task_name] = None
        self.task_images[task_name] = []

        # 默认选中添加的任务
        self.table_widget.selectRow(0)

    def create_non_editable_item(self, text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def show_info_bar(self, message, info_type='info'):
        if info_type == 'error':
            InfoBar.error(
                title='提示',
                content=message,
                isClosable=True,
                duration=2000,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
        elif info_type == 'success':
            InfoBar.success(
                title='提示',
                content=message,
                isClosable=True,
                duration=2000,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )

    def delete_selected_row(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.show_info_bar('请选择需要删除的任务', 'error')
            return

        selected_row = self.table_widget.row(selected_items[0])
        task_name = self.table_widget.item(selected_row, 0).text()
        self.table_widget.removeRow(selected_row)
        self.log_text.append(f"任务删除成功: {task_name}")
        self.show_info_bar('删除成功', 'success')

        # 重置进度条
        self.progress_ring.setValue(0)
        self.progress_ring.setFormat("0%")

        # 清除日志
        self.log_text.clear()
        self.image_label.clear()

        if task_name in self.task_logs:
            del self.task_logs[task_name]
        if task_name in self.task_progress:
            del self.task_progress[task_name]
        if task_name in self.threads:
            del self.threads[task_name]
        if task_name in self.task_results:
            del self.task_results[task_name]
        if task_name in self.stop_flags:
            del self.stop_flags[task_name]
        if task_name in self.task_images:
            del self.task_images[task_name]

        self.delete_task_files(task_name)

        # 自动选中下一个任务（如果有）
        if self.table_widget.rowCount() > 0:
            next_row = selected_row if selected_row < self.table_widget.rowCount() else self.table_widget.rowCount() - 1
            self.table_widget.selectRow(next_row)
            # 手动触发选中项变化的信号
            self.on_task_selected()
        else:
            self.progress_ring.setValue(0)
            self.log_text.clear()
            self.image_label.clear()

    def delete_task_files(self, task_name):
        excel_file = os.path.join('excel', f"{task_name}_识别结果.xlsx")
        if os.path.exists(excel_file):
            os.remove(excel_file)

        identifyimages_dir = os.path.join('Identifyimages', task_name)
        if os.path.exists(identifyimages_dir):
            self._delete_folder(identifyimages_dir)

        imagestorage_dir = os.path.join('Imagestorage', task_name)
        if os.path.exists(imagestorage_dir):
            self._delete_folder(imagestorage_dir)

    def _delete_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)

    def start_task(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.show_info_bar('请选择需要执行的任务', 'error')
            return

        selected_row = self.table_widget.row(selected_items[0])
        task = {
            'task_name': self.table_widget.item(selected_row, 0).text(),
            'model': self.table_widget.item(selected_row, 1).text(),
            'export_option': self.table_widget.item(selected_row, 2).text(),
            'file_path': self.table_widget.item(selected_row, 3).text(),
            'recognition_image_path': self.table_widget.item(selected_row, 4).text(),
            'replace_option': self.table_widget.item(selected_row, 5).text(),
            'replace_text': self.table_widget.item(selected_row, 6).text(),
            'frame_interval': self.table_widget.item(selected_row, 7).text(),
            'grayscale_option': self.table_widget.item(selected_row, 8).text()
        }

        self.log_text.append(f"开始执行任务: {task['task_name']}")

        if task['task_name'] in self.threads and self.threads[task['task_name']].isRunning():
            self.show_info_bar('任务已在执行中', 'error')
            return

        stop_flag = {'stop': False}
        processing_thread = ImageProcessingThread(task, stop_flag)
        processing_thread.log_signal.connect(lambda msg, t=task['task_name']: self.update_log(t, msg))
        processing_thread.progress_signal.connect(lambda value, t=task['task_name']: self.update_progress(t, value))
        processing_thread.task_complete_signal.connect(lambda t=task['task_name']: self.on_task_complete(t))
        processing_thread.result_signal.connect(self.handle_ocr_results)
        processing_thread.image_visualization_signal.connect(self.store_image_for_task)

        self.threads[task['task_name']] = processing_thread
        self.stop_flags[task['task_name']] = stop_flag
        processing_thread.start()

        self.set_task_row_color(selected_row, QColor("#f3d6ac"))

    def stop_task(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.show_info_bar('请选择任务', 'error')
            return

        selected_row = self.table_widget.row(selected_items[0])
        task_name = self.table_widget.item(selected_row, 0).text()

        if task_name not in self.threads or not self.threads[task_name].isRunning():
            self.show_info_bar('任务未在执行过程中', 'error')
            return

        self.stop_flags[task_name]['stop'] = True
        self.show_info_bar('任务停止成功', 'success')

        # 设置任务行颜色为#FF6347
        self.set_task_row_color(selected_row, QColor("#FA8072"))

        # 仅在OCR过程中弹出导出对话框
        if self.threads[task_name].ocr_running:
            self.show_export_dialog(task_name)

    def show_export_dialog(self, task_name):
        w = MessageBox(
            '说明',
            '需要导出已识别的结果吗？',
            self
        )
        w.yesButton.setText('导出')
        w.cancelButton.setText('不导出')
        if w.exec():
            if task_name in self.task_results:
                self.threads[task_name].export_results(task_name, self.task_results[task_name])
        else:
            self.show_info_bar('结果未导出', 'info')

    def handle_ocr_results(self, result):
        task_name, ocr_results = result
        self.task_results[task_name] = ocr_results
        if not self.stop_flags[task_name]['stop']:
            self.threads[task_name].export_results(task_name, ocr_results)

    def on_task_selected(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            if self.progress_ring.value() != 0:
                self.progress_ring.setValue(0)
                self.progress_ring.setFormat("0%")
            self.log_text.clear()
            self.image_label.clear()
            return

        task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        self.log_text.clear()
        if task_name in self.task_logs:
            self.log_text.append("\n".join(self.task_logs[task_name]))
        if task_name in self.task_progress:
            self.update_progress(task_name, self.task_progress[task_name])

        self.update_image_display_for_task(task_name)

    def on_task_complete(self, task_name):
        if not self.stop_flags[task_name]['stop']:
            self.show_info_bar(f'{task_name}任务完成', 'success')
            row_position = self.find_task_row(task_name)
            if row_position is not None:
                self.set_task_row_color(row_position, QColor("#c9eabe"))

    def find_task_row(self, task_name):
        for row in range(self.table_widget.rowCount()):
            if self.table_widget.item(row, 0).text() == task_name:
                return row
        return None

    def set_task_row_color(self, row, color):
        for col in range(self.table_widget.columnCount()):
            self.table_widget.item(row, col).setBackground(color)

    def update_log(self, task_name, message):
        self.task_logs[task_name].append(message)
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return
        selected_task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        if selected_task_name == task_name:
            self.log_text.append(message)

    def update_progress(self, task_name, value):
        self.task_progress[task_name] = value
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return
        selected_task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        if selected_task_name == task_name:
            self.progress_ring.setValue(value)
            self.progress_ring.setFormat(f"{value}%")

    def clear_log(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.show_info_bar('请选择要清空日志的任务', 'error')
            return

        task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        if task_name in self.task_logs:
            self.task_logs[task_name] = []

        self.log_text.clear()
        self.log_text.append("日志已清空")
        self.show_info_bar('日志已清空', 'success')

    def task_exists(self, task_name):
        for row in range(self.table_widget.rowCount()):
            if self.table_widget.item(row, 0).text() == task_name:
                return True
        return False

    def store_image_for_task(self, task_name, image_path):
        if task_name in self.task_images:
            self.task_images[task_name].append(image_path)

        if self.is_current_task_selected(task_name):
            self.set_image_to_label(image_path)

    def is_current_task_selected(self, task_name):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return False
        selected_task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        return task_name == selected_task_name

    def set_image_to_label(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_image_display_for_task(self, task_name):
        if task_name in self.task_images:
            task_images = self.task_images[task_name]
            if task_images:
                last_image_path = task_images[-1]
                self.set_image_to_label(last_image_path)
            else:
                self.image_label.clear()
