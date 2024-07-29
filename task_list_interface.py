import os
import cv2
import time
import ddddocr
from natsort import natsorted
from paddleocr import PaddleOCR
from threading import Thread
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidgetItem, QHeaderView, QSpacerItem, QSizePolicy, QHBoxLayout
from qfluentwidgets import TableWidget, PushButton, ProgressRing, InfoBar, InfoBarPosition, TextEdit
import numpy as np


def run_ocr(task, log_signal, progress_signal, result_signal, total_steps):
    """
    运行OCR识别任务，根据指定的模型识别图像中的文本，并替换结果中的特定文本。

    参数：
        task (dict): 包含任务信息的字典
        log_signal (pyqtSignal): 用于发送日志消息的信号
        progress_signal (pyqtSignal): 用于发送进度更新的信号
        result_signal (pyqtSignal): 用于发送结果的信号
        total_steps (int): 任务的总步骤数
    """
    try:
        recognition_image_paths = task['recognition_image_path'].split(",")
        model = task['model']
        replace_option = task['replace_option']
        replace_text = task['replace_text']
        imagestorage_dir = os.path.join('Imagestorage', task['task_name'])
        task_name = task['task_name']

        # 根据模型名称加载相应的OCR模型
        if model == "Ddddocr模型":
            ocr_model = ddddocr.DdddOcr()
        elif model == "Paddle模型":
            ocr_model = PaddleOCR(det_model_dir='./models/whl/det/ch/ch_PP-OCRv4_det_infer/',
                                  rec_model_dir='./models/whl/rec/ch/ch_PP-OCRv4_rec_infer/',
                                  cls_model_dir='./models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/',
                                  lang='ch', use_angle_cls=True, use_gpu=False)

        step_count = total_steps // 2  # 已完成的视频转换部分
        ocr_results = {}

        for img_name in recognition_image_paths:
            img_name = img_name.strip()
            log_signal.emit(f"识别 '{img_name}' 开始")
            full_path = os.path.join("Identifyimages", task_name, img_name)
            if not os.path.exists(full_path):
                log_signal.emit(f"'{img_name}' 图片不存在，任务停止")
                return

            folder_name = img_name.split('.')[0]
            folder_path = os.path.join(imagestorage_dir, folder_name)
            if not os.path.exists(folder_path):
                log_signal.emit(f"'{folder_name}' 文件夹不存在，任务停止")
                return

            ocr_results[img_name] = []

            # 对图片文件名进行自然排序
            image_files = natsorted(os.listdir(folder_path))

            for image_file in image_files:
                img_path = os.path.join(folder_path, image_file)

                # 读取图片
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    log_signal.emit(f"无法读取 '{image_file}' 图片，任务停止")
                    return

                # 找到轮廓
                contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # 根据外接矩形分割字符
                    char_image = image[y:y + h, x:x + w]

                    # 保存字符图像到临时路径
                    char_path = f"{folder_path}/temp_char.png"
                    cv2.imwrite(char_path, char_image)

                    result = ""
                    if model == "Ddddocr模型":
                        with open(char_path, "rb") as img_file:
                            char_data = img_file.read()
                            result = ocr_model.classification(char_data)
                    elif model == "Paddle模型":
                        paddle_result = ocr_model.ocr(char_path, cls=True)
                        if paddle_result and paddle_result[0]:
                            result = ''.join([line[1][0] for line in paddle_result[0]])

                    os.remove(char_path)  # 删除临时文件

                    # 调用替换文本方法
                    result = replace_text_in_result(result, replace_option, replace_text)
                    ocr_results[img_name].append((image_file, result))
                    log_signal.emit(f"图片 {image_file} 的识别结果: {result}")
                    time.sleep(0.1)  # 延迟100ms

                    step_count += 1
                    progress_signal.emit(int((step_count / total_steps) * 100))

            log_signal.emit(f"识别 '{img_name}' 结束")

        result_signal.emit((task_name, ocr_results))
        progress_signal.emit(100)
        log_signal.emit("任务结束")

    except Exception as e:
        log_signal.emit(f"处理过程中出现错误: {e}")

def replace_text_in_result(result, replace_option, replace_text):
    """
    根据替换选项处理识别结果文本。

    参数：
        result (str): 识别结果文本
        replace_option (str): 替换选项，可以是 "替换文本" 或 "去掉文本"
        replace_text (str): 替换文本或去掉的文本

    返回：
        str: 处理后的结果文本
    """
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

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        """
        运行图像处理任务，包括从视频中截取图像并运行OCR识别。
        """
        try:
            task_name = self.task['task_name']
            imagestorage_dir = os.path.join('Imagestorage', task_name)
            frame_interval = int(self.task['frame_interval'])
            grayscale_option = self.task['grayscale_option'] == "转换灰度图像"

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

                recognition_image = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if recognition_image is None:
                    self.log_signal.emit(f"无法读取 '{img_name}' 图片，任务停止")
                    return

                h, w = recognition_image.shape[:2]

                second_count = 0
                while second_count < duration:
                    cap.set(cv2.CAP_PROP_POS_MSEC, second_count * 1000)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    res = cv2.matchTemplate(frame, recognition_image, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)

                    crop_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                    if grayscale_option:
                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

                    frame_name = os.path.join(folder_path, f"{task_name}_{folder_name}_{second_count:02d}.png")
                    cv2.imencode('.png', crop_image)[1].tofile(frame_name)
                    self.log_signal.emit(f"保存图片: {frame_name}")

                    second_count += frame_interval
                    step_count += 1
                    self.progress_signal.emit(int((step_count / total_steps) * 100))

                self.log_signal.emit(f"'{img_name}' 截取完毕")

            cap.release()
            self.log_signal.emit("截取图片完毕，正在执行识别初始化...")

            ocr_thread = Thread(target=run_ocr, args=(
                self.task, self.log_signal, self.progress_signal, self.result_signal, total_steps))
            ocr_thread.start()

            while ocr_thread.is_alive():
                time.sleep(0.1)

            ocr_thread.join()

            self.task_complete_signal.emit(task_name)

        except Exception as e:
            self.log_signal.emit(f"处理过程中出现错误: {e}")

    def export_results(self, task_name, ocr_results):
        """
        导出OCR识别结果到Excel文件。

        参数：
            task_name (str): 任务名称
            ocr_results (dict): OCR识别结果
        """
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

            for img_name, results in ocr_results.items():
                sheet_name = img_name.split('.')[0]
                worksheet = workbook.create_sheet(title=sheet_name)

                # 添加表头
                worksheet.append(["图片文件", "识别结果", "原图"])

                thin_border = Border(left=Side(style='thin'),
                                     right=Side(style='thin'),
                                     top=Side(style='thin'),
                                     bottom=Side(style='thin'))

                for r_idx, (image_file, result) in enumerate(results, start=2):
                    worksheet.cell(row=r_idx, column=1, value=image_file)
                    worksheet.cell(row=r_idx, column=2, value=result)

                    img_path = os.path.join('Imagestorage', task_name, sheet_name, image_file)
                    img = Image(img_path)

                    # 获取图片尺寸并调整单元格大小
                    img_width, img_height = img.width, img.height
                    col_letter = get_column_letter(3)
                    worksheet.column_dimensions[col_letter].width = img_width * 0.14  # 调整列宽
                    worksheet.row_dimensions[r_idx].height = img_height * 0.75  # 调整行高

                    img.anchor = f'C{r_idx}'
                    worksheet.add_image(img)

                    # 为单元格添加边框
                    for col in range(1, 4):
                        worksheet.cell(row=r_idx, column=col).border = thin_border

                # 为表头添加边框
                for col in range(1, 4):
                    worksheet.cell(row=1, column=col).border = thin_border

            del workbook['Sheet']  # 删除默认生成的工作表
            workbook.save(output_path)
            self.log_signal.emit(f"识别结果已导出至: {output_path}")
        except Exception as e:
            self.log_signal.emit(f"导出结果时出现错误: {e}")

    def is_file_open(self, file_path):
        """
        检查文件是否被占用。

        参数：
            file_path (str): 文件路径

        返回：
            bool: 文件是否被占用
        """
        if os.path.exists(file_path):
            try:
                os.rename(file_path, file_path)
                return False
            except OSError:
                return True
        return False

    def _delete_folder(self, folder_path):
        """
        删除文件夹及其内容。

        参数：
            folder_path (str): 文件夹路径
        """
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

        # 启用边框并设置圆角
        self.table_widget.setBorderVisible(True)
        self.table_widget.setBorderRadius(8)

        self.table_widget.setWordWrap(False)
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(9)  # 增加两列
        self.table_widget.setHorizontalHeaderLabels(
            ['任务名', '识别模型', '导出选项', '视频路径', '识别图片路径', '替换选项', '替换文本', '帧提取间隔', '灰度图像选项'])  # 新增列头
        self.table_widget.verticalHeader().hide()

        # 设置列宽
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setColumnWidth(0, 100)
        self.table_widget.setColumnWidth(1, 120)
        self.table_widget.setColumnWidth(2, 120)
        self.table_widget.setColumnWidth(3, 120)
        self.table_widget.setColumnWidth(4, 120)
        self.table_widget.setColumnWidth(5, 120)
        self.table_widget.setColumnWidth(6, 100)
        self.table_widget.setColumnWidth(7, 100)  # 新增列宽
        self.table_widget.setColumnWidth(8, 120)  # 新增列宽

        self.main_layout.addWidget(self.table_widget)

        # 添加环形进度条和按钮
        self.progress_ring = ProgressRing(self)
        self.progress_ring.setTextVisible(True)
        self.progress_ring.setFormat("0%")
        self.progress_ring.setStrokeWidth(10)
        self.progress_ring.setFixedSize(130, 130)

        self.start_button = PushButton('开始执行', self)
        self.start_button.clicked.connect(self.start_task)
        self.delete_button = PushButton('删除任务', self)
        self.delete_button.clicked.connect(self.delete_selected_row)
        self.clear_log_button = PushButton('清空日志', self)  # 新增清空日志按钮
        self.clear_log_button.clicked.connect(self.clear_log)  # 连接清空日志方法

        # 日志显示区
        self.log_text = TextEdit(self)
        self.log_text.setReadOnly(True)

        # 布局管理
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.clear_log_button)  # 添加清空日志按钮到布局

        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)  # 减少控件之间的间隙
        control_layout.addWidget(self.start_button)
        control_layout.addLayout(button_layout)

        # 创建一个带有边距的水平布局
        progress_button_layout = QHBoxLayout()
        progress_button_layout.addSpacerItem(
            QSpacerItem(12, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 左边距
        progress_button_layout.addWidget(self.progress_ring)
        progress_button_layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 环形进度条右边距
        progress_button_layout.addLayout(control_layout)
        progress_button_layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 右边距

        # 新增一个水平布局，用于放置日志显示区
        progress_log_layout = QHBoxLayout()
        progress_log_layout.addLayout(progress_button_layout)
        progress_log_layout.addWidget(self.log_text)
        progress_log_layout.addSpacerItem(
            QSpacerItem(12, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 右边距

        # 添加环形进度条上下面固定边距
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 上边距
        self.main_layout.addLayout(progress_log_layout)
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))  # 下边距

        # 移除空白区域延长表格高度
        self.main_layout.setStretch(0, 5)  # 表格占用较大部分
        self.main_layout.setStretch(2, 1)  # 控件部分占用较小部分

        # 选择任务时显示对应的日志和进度条
        self.table_widget.itemSelectionChanged.connect(self.on_task_selected)

        self.threads = {}
        self.task_logs = {}
        self.task_progress = {}
        self.task_results = {}

    def add_task(self, task_name, model, export_option, file_path, replace_option, image_path, replace_text, frame_interval, grayscale_option):
        """
        添加新的任务到任务列表。

        参数：
            task_name (str): 任务名称
            model (str): 识别模型
            export_option (str): 导出选项
            file_path (str): 视频文件路径
            replace_option (str): 替换选项
            image_path (str): 识别图片路径
            replace_text (str): 替换文本
            frame_interval (str): 帧提取间隔
            grayscale_option (str): 灰度图像选项
        """
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

        self.table_widget.setItem(row_position, 0, self.create_non_editable_item(task_name))
        self.table_widget.setItem(row_position, 1, self.create_non_editable_item(model))
        self.table_widget.setItem(row_position, 2, self.create_non_editable_item(export_option))
        self.table_widget.setItem(row_position, 3, self.create_non_editable_item(file_path))
        self.table_widget.setItem(row_position, 4, self.create_non_editable_item(image_path))
        self.table_widget.setItem(row_position, 5, self.create_non_editable_item(replace_option))
        self.table_widget.setItem(row_position, 6, self.create_non_editable_item(replace_text))
        self.table_widget.setItem(row_position, 7, self.create_non_editable_item(frame_interval))  # 新增列
        self.table_widget.setItem(row_position, 8, self.create_non_editable_item(grayscale_option))  # 新增列

        # 初始化日志和进度
        self.task_logs[task_name] = []
        self.task_progress[task_name] = 0
        self.task_results[task_name] = None

    def create_non_editable_item(self, text):
        """
        创建不可编辑的表格项。

        参数：
            text (str): 表格项文本

        返回：
            QTableWidgetItem: 不可编辑的表格项
        """
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def show_info_bar(self, message, info_type='info'):
        """
        显示信息条。

        参数：
            message (str): 信息内容
            info_type (str): 信息类型，可以是 'info' 或 'error'
        """
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
        """
        删除选中的任务行。
        """
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.show_info_bar('请选择需要删除的任务', 'error')
            return

        selected_row = self.table_widget.row(selected_items[0])
        task_name = self.table_widget.item(selected_row, 0).text()  # 获取任务名
        self.table_widget.removeRow(selected_row)
        self.log_text.append(f"任务删除成功: {task_name}")
        self.show_info_bar('删除成功', 'success')

        # 删除任务相关的日志和进度
        if task_name in self.task_logs:
            del self.task_logs[task_name]
        if task_name in self.task_progress:
            del self.task_progress[task_name]
        if task_name in self.threads:
            del self.threads[task_name]
        if task_name in self.task_results:
            del self.task_results[task_name]

        # 删除对应的文件或文件夹
        self.delete_task_files(task_name)

    def delete_task_files(self, task_name):
        """
        删除任务相关的文件或文件夹。

        参数：
            task_name (str): 任务名称
        """
        # 删除 excel 文件夹中的文件
        excel_file = os.path.join('excel', f"{task_name}_识别结果.xlsx")
        if os.path.exists(excel_file):
            os.remove(excel_file)

        # 删除 Identifyimages 文件夹中的文件夹
        identifyimages_dir = os.path.join('Identifyimages', task_name)
        if os.path.exists(identifyimages_dir):
            self._delete_folder(identifyimages_dir)

        # 删除 Imagestorage 文件夹中的文件夹
        imagestorage_dir = os.path.join('Imagestorage', task_name)
        if os.path.exists(imagestorage_dir):
            self._delete_folder(imagestorage_dir)

    def _delete_folder(self, folder_path):
        """
        删除文件夹及其内容。

        参数：
            folder_path (str): 文件夹路径
        """
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)

    def start_task(self):
        """
        开始执行选中的任务。
        """
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

        processing_thread = ImageProcessingThread(task)
        processing_thread.log_signal.connect(lambda msg, t=task['task_name']: self.update_log(t, msg))
        processing_thread.progress_signal.connect(lambda value, t=task['task_name']: self.update_progress(t, value))
        processing_thread.task_complete_signal.connect(lambda t=task['task_name']: self.on_task_complete(t))
        processing_thread.result_signal.connect(self.handle_ocr_results)  # 连接处理结果的信号

        self.threads[task['task_name']] = processing_thread
        processing_thread.start()

        self.set_task_row_color(selected_row, QColor("#f3d6ac"))

    def handle_ocr_results(self, result):
        """
        处理OCR识别结果。

        参数：
            result (tuple): 包含任务名称和识别结果的元组
        """
        task_name, ocr_results = result
        self.task_results[task_name] = ocr_results
        self.threads[task_name].export_results(task_name, ocr_results)

    def on_task_selected(self):
        """
        任务选择变化时的处理。
        """
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return

        task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        self.log_text.clear()
        if task_name in self.task_logs:
            self.log_text.append("\n".join(self.task_logs[task_name]))
        if task_name in self.task_progress:
            self.update_progress(task_name, self.task_progress[task_name])

    def on_task_complete(self, task_name):
        """
        任务完成时的处理。

        参数：
            task_name (str): 任务名称
        """
        self.show_info_bar(f'{task_name}任务完成', 'success')
        row_position = self.find_task_row(task_name)
        if row_position is not None:
            self.set_task_row_color(row_position, QColor("#c9eabe"))

    def find_task_row(self, task_name):
        """
        根据任务名称查找任务所在的行。

        参数：
            task_name (str): 任务名称

        返回：
            int: 任务所在的行号
        """
        for row in range(self.table_widget.rowCount()):
            if self.table_widget.item(row, 0).text() == task_name:
                return row
        return None

    def set_task_row_color(self, row, color):
        """
        设置任务行的背景颜色。

        参数：
            row (int): 任务行号
            color (QColor): 颜色
        """
        for col in range(self.table_widget.columnCount()):
            self.table_widget.item(row, col).setBackground(color)

    def update_log(self, task_name, message):
        """
        更新任务日志。

        参数：
            task_name (str): 任务名称
            message (str): 日志信息
        """
        self.task_logs[task_name].append(message)
        if self.table_widget.item(self.table_widget.currentRow(), 0).text() == task_name:
            self.log_text.append(message)

    def update_progress(self, task_name, value):
        """
        更新任务进度。

        参数：
            task_name (str): 任务名称
            value (int): 进度值
        """
        self.task_progress[task_name] = value
        if self.table_widget.item(self.table_widget.currentRow(), 0).text() == task_name:
            self.progress_ring.setValue(value)
            self.progress_ring.setFormat(f"{value}%")

    def clear_log(self):
        """
        清空选中任务的日志。
        """
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
        """
        检查任务是否存在。

        参数：
            task_name (str): 任务名称

        返回：
            bool: 任务是否存在
        """
        for row in range(self.table_widget.rowCount()):
            if self.table_widget.item(row, 0).text() == task_name:
                return True
        return False
