# coding:utf-8
import os
import cv2
import time
import json
import ddddocr
from natsort import natsorted
from paddleocr import PaddleOCR
from threading import Thread
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidgetItem, QHeaderView, QSpacerItem, QSizePolicy, QHBoxLayout, QDialog, QLabel
from qfluentwidgets import TableWidget, PushButton, ProgressRing, InfoBar, InfoBarPosition, TextEdit, MessageBox
import numpy as np

def run_ocr(task, log_signal, progress_signal, result_signal, total_steps, thread_stop_flag):
    try:
        recognition_image_paths = task['recognition_image_path'].split(",")
        model = task['model']
        replace_option = task['replace_option']
        replace_text = task['replace_text']
        imagestorage_dir = os.path.join('Imagestorage', task['task_name'])
        task_name = task['task_name']

        if model == "Ddddocr模型":
            ocr_model = ddddocr.DdddOcr()
        elif model == "Paddle模型":
            ocr_model = PaddleOCR(det_model_dir='./models/whl/det/ch/ch_PP-OCRv4_det_infer/',
                                  rec_model_dir='./models/whl/rec/ch/ch_PP-OCRv4_rec_infer/',
                                  cls_model_dir='./models/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/',
                                  lang='ch', use_angle_cls=True, use_gpu=False)

        step_count = total_steps // 2
        ocr_results = {}

        for img_name in recognition_image_paths:
            if thread_stop_flag['stop']:
                log_signal.emit("OCR任务已停止")
                result_signal.emit((task_name, ocr_results))
                return
            img_name = img_name.strip()
            log_signal.emit(f"识别 '{img_name}' 开始")
            full_path = os.path.join("Identifyimages", task_name, img_name)
            if not os.path.exists(full_path):
                log_signal.emit(f"'{img_name}' 图片不存在，任务停止")
                result_signal.emit((task_name, ocr_results))
                return

            folder_name = img_name.split('.')[0]
            folder_path = os.path.join(imagestorage_dir, folder_name)
            if not os.path.exists(folder_path):
                log_signal.emit(f"'{folder_name}' 文件夹不存在，任务停止")
                result_signal.emit((task_name, ocr_results))
                return

            ocr_results[img_name] = []

            image_files = natsorted(os.listdir(folder_path))

            for image_file in image_files:
                if thread_stop_flag['stop']:
                    log_signal.emit("OCR任务已停止")
                    result_signal.emit((task_name, ocr_results))
                    return
                img_path = os.path.join(folder_path, image_file)
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    log_signal.emit(f"无法读取 '{image_file}' 图片，任务停止")
                    result_signal.emit((task_name, ocr_results))
                    return

                contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if thread_stop_flag['stop']:
                        log_signal.emit("OCR任务已停止")
                        result_signal.emit((task_name, ocr_results))
                        return
                    x, y, w, h = cv2.boundingRect(contour)
                    char_image = image[y:y + h, x:x + w]

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

                    os.remove(char_path)

                    result = replace_text_in_result(result, replace_option, replace_text)
                    ocr_results[img_name].append((image_file, result))
                    log_signal.emit(f"图片 {image_file} 的识别结果: {result}")
                    time.sleep(0.1)

                    step_count += 1
                    progress_signal.emit(int((step_count / total_steps) * 100))

            log_signal.emit(f"识别 '{img_name}' 结束")

        result_signal.emit((task_name, ocr_results))
        progress_signal.emit(100)
        log_signal.emit("任务结束")

    except Exception as e:
        log_signal.emit(f"处理过程中出现错误: {e}")

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

    def __init__(self, task, stop_flag):
        super().__init__()
        self.task = task
        self.stop_flag = stop_flag
        self.ocr_running = False

    def run(self):
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

                # 检查 index 是否存在于 recognition_image_paths 中
                indices = [i + 1 for i in range(len(recognition_image_paths))]
                for rect in rectangles:
                    if rect["index"] not in indices:
                        self.log_signal.emit("Json文件配置异常")
                        return

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

            self.ocr_running = True
            ocr_thread = Thread(target=run_ocr, args=(
                self.task, self.log_signal, self.progress_signal, self.result_signal, total_steps, self.stop_flag))
            ocr_thread.start()

            while ocr_thread.is_alive():
                if self.stop_flag['stop']:
                    self.log_signal.emit("任务已停止")
                    self.ocr_running = False
                    return
                time.sleep(0.1)

            ocr_thread.join()
            self.ocr_running = False

            if not self.stop_flag['stop']:
                self.task_complete_signal.emit(task_name)

        except Exception as e:
            self.log_signal.emit(f"处理过程中出现错误: {e}")

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

            for img_name, results in ocr_results.items():
                sheet_name = img_name.split('.')[0]
                worksheet = workbook.create_sheet(title=sheet_name)

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

                    img_width, img_height = img.width, img.height
                    col_letter = get_column_letter(3)
                    worksheet.column_dimensions[col_letter].width = img_width * 0.14
                    worksheet.row_dimensions[r_idx].height = img_height * 0.75

                    img.anchor = f'C{r_idx}'
                    worksheet.add_image(img)

                    for col in range(1, 4):
                        worksheet.cell(row=r_idx, column=col).border = thin_border

                for col in range(1, 4):
                    worksheet.cell(row=1, column=col).border = thin_border

            del workbook['Sheet']
            workbook.save(output_path)
            self.log_signal.emit(f"识别结果已导出至: {output_path}")
        except Exception as e:
            self.log_signal.emit(f"导出结果时出现错误: {e}")

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

    def stop(self):
        self.stop_flag['stop'] = True

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
            ['任务名', '识别模型', '导出选项', '视频路径', '识别图片路径', '替换选项', '替换文本', '提取间隔', '灰度图像选项'])
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
            QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        progress_button_layout.addLayout(control_layout)
        progress_button_layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        progress_log_layout = QHBoxLayout()
        progress_log_layout.addLayout(progress_button_layout)
        progress_log_layout.addWidget(self.log_text)
        progress_log_layout.addSpacerItem(
            QSpacerItem(12, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.main_layout.addSpacerItem(
            QSpacerItem(0, 12, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.main_layout.addLayout(progress_log_layout)
        self.main_layout.addSpacerItem(
            QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.main_layout.setStretch(0, 5)
        self.main_layout.setStretch(2, 1)

        self.table_widget.itemSelectionChanged.connect(self.on_task_selected)

        self.threads = {}
        self.stop_flags = {}  # Initialize stop_flags attribute
        self.task_logs = {}
        self.task_progress = {}
        self.task_results = {}

    def add_task(self, task_name, model, export_option, file_path, replace_option, image_path, replace_text, frame_interval, grayscale_option):
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

        self.delete_task_files(task_name)

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
        if not self.stop_flags[task_name]['stop']:  # Add this condition
            self.threads[task_name].export_results(task_name, ocr_results)

    def on_task_selected(self):
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            if self.progress_ring.value() != 0:
                self.progress_ring.setValue(0)
                self.progress_ring.setFormat("0%")
            self.log_text.clear()
            return

        task_name = self.table_widget.item(self.table_widget.currentRow(), 0).text()
        self.log_text.clear()
        if task_name in self.task_logs:
            self.log_text.append("\n".join(self.task_logs[task_name]))
        if task_name in self.task_progress:
            self.update_progress(task_name, self.task_progress[task_name])

    def on_task_complete(self, task_name):
        if not self.stop_flags[task_name]['stop']:  # Add this condition
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
