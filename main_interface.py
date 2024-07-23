# coding:utf-8
import re
import os
from PyQt6.QtCore import Qt, QTimer, QRegularExpression, QByteArray
from PyQt6.QtGui import QPixmap, QImage, QRegularExpressionValidator, QIcon, QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QFrame, QLabel, QSizePolicy
from qfluentwidgets import PushButton, LineEdit, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, MessageBox, \
    Slider, ToolButton, ElevatedCardWidget, SingleDirectionScrollArea
import cv2
from picture_popup import ImageDialog
from Identifyimages import IdentifyImagesDialog


def svg_to_icon(svg_data):
    """
    将SVG数据转换为QIcon对象。

    参数：
        svg_data (str): SVG图标数据字符串

    返回：
        QIcon: 转换后的图标对象
    """
    svg_renderer = QSvgRenderer(QByteArray(svg_data.encode()))
    pixmap = QPixmap(200, 200)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    svg_renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


# SVG图标数据
play_svg = """
<svg t="1719457742458" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1214" width="200" height="200"><path d="M783.74 401.86L372.23 155.28c-85.88-51.46-195.08 10.41-195.08 110.53v493.16c0 100.12 109.2 161.99 195.08 110.53l411.51-246.58c83.5-50.04 83.5-171.03 0-221.06z" p-id="1215"></path></svg>
"""

pause_svg = """
<svg t="1719457759478" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1365" width="200" height="200"><path d="M332.26 853.89c-49.71 0-90-40.29-90-90v-503c0-49.71 40.29-90 90-90s90 40.29 90 90v503c0 49.7-40.3 90-90 90zM691.26 853.89c-49.71 0-90-40.29-90-90v-503c0-49.71 40.29-90 90-90s90 40.29 90 90v503c0 49.7-40.3 90-90 90z" p-id="1366"></path></svg>
"""

rewind_svg = """
<svg t="1719458651503" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="912" width="200" height="200"><path d="M356.77 605.22L701.8 849.53c69.8 49.42 166.29-0.49 166.29-86.01V261.25c0-85.52-96.49-135.43-166.29-86.01L356.77 419.55c-64.02 45.34-64.02 140.34 0 185.67zM210.42 154.36c30.38 0 55 24.62 55 55v606.07c0 30.38-24.62 55-55 55s-55-24.62-55-55V209.36c0-30.38 24.62-55 55-55z" p-id="913"></path></svg>
"""

forward_svg = """
<svg t="1719458670424" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1063" width="200" height="200"><path d="M665.47 417.65l-345.03-244.3c-69.8-49.42-166.29 0.49-166.29 86.01v502.27c0 85.52 96.49 135.43 166.29 86.01l345.03-244.31c64.02-45.34 64.02-140.34 0-185.68zM811.82 868.52c-30.38 0-55-24.62-55-55V207.46c0-30.38 24.62-55 55-55s55 24.62 55 55v606.07c0 30.37-24.62 54.99-55 54.99z" p-id="1064"></path></svg>
"""


class Addtaskinterface(QWidget):
    def __init__(self, task_list_interface, parent=None):
        """
        初始化添加任务界面。

        参数：
            task_list_interface (TaskListInterface): 任务列表界面实例
            parent (QWidget, 可选): 父级窗口
        """
        super().__init__(parent)
        self.setObjectName('Addtask')

        self.task_list_interface = task_list_interface
        self.additional_input_layouts = []

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 滚动区域
        scroll_area = SingleDirectionScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")  # 设置透明背景和无边框
        scroll_area_widget = QWidget()
        scroll_area_widget.setStyleSheet("QWidget{background: transparent}")  # 设置内部视图透明背景
        scroll_area_layout = QVBoxLayout(scroll_area_widget)

        # 文件选择布局
        file_layout = QHBoxLayout()
        self.file_entry = LineEdit(self)
        self.file_entry.setPlaceholderText("请选择文件路径")
        self.file_entry.setClearButtonEnabled(True)
        self.file_entry.setReadOnly(True)
        self.file_button = PushButton('选择文件', self)
        self.file_button.setIcon(FIF.FOLDER)
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_entry)
        file_layout.addWidget(self.file_button)

        # 图片截取布局
        capture_layout = QHBoxLayout()
        self.capture_entry = LineEdit(self)
        self.capture_entry.setPlaceholderText("请使用下面的视频工具截取图片")
        self.capture_entry.setClearButtonEnabled(True)
        self.capture_entry.setReadOnly(True)
        self.clear_images_button = PushButton('清空图片', self)
        self.clear_images_button.setIcon(FIF.DELETE)
        self.clear_images_button.clicked.connect(self.clear_images)
        self.view_image_button = PushButton('查看识别图片', self)
        self.view_image_button.setIcon(FIF.SEARCH)
        self.view_image_button.clicked.connect(self.view_image)
        capture_layout.addWidget(self.capture_entry)
        capture_layout.addWidget(self.clear_images_button)
        capture_layout.addWidget(self.view_image_button)

        # 输入布局
        input_layout = QHBoxLayout()
        self.input_entry = LineEdit(self)
        self.input_entry.setPlaceholderText("请输入任务名")
        self.input_entry.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))
        self.input_entry.setClearButtonEnabled(True)
        self.dropdown1 = ComboBox(self)
        self.dropdown1.addItems(["Ddddocr模型", "Paddle模型"])
        self.dropdown3 = ComboBox(self)
        self.dropdown3.addItems(["导出表格", "不导出表格"])
        self.dropdown4 = ComboBox(self)
        self.dropdown4.addItems(["不替换文本", "替换文本", "去掉文本"])
        self.dropdown4.currentIndexChanged.connect(self.update_input_visibility)

        input_layout.addWidget(self.input_entry, 2)
        input_layout.addWidget(self.dropdown1, 1)
        input_layout.addWidget(self.dropdown3, 1)
        input_layout.addWidget(self.dropdown4, 1)

        input_layout.setStretch(0, 2)
        input_layout.setStretch(1, 1)
        input_layout.setStretch(2, 1)
        input_layout.setStretch(3, 1)

        self.additional_input_container = QVBoxLayout()
        self.top_input_layout = QHBoxLayout()
        self.top_input1 = LineEdit(self)
        self.top_input1.setPlaceholderText("请输入替换的文本")
        self.top_input1.setClearButtonEnabled(True)
        self.top_input1.setMaxLength(50)
        self.top_input1.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))

        self.top_input2 = LineEdit(self)
        self.top_input2.setPlaceholderText("请输入替换后的文本")
        self.top_input2.setClearButtonEnabled(True)
        self.top_input2.setMaxLength(50)
        self.top_input2.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))

        self.add_button = PushButton('新增', self)
        self.add_button.clicked.connect(self.add_additional_input)

        self.top_input_layout.addWidget(self.top_input1)
        self.top_input_layout.addWidget(self.top_input2)
        self.top_input_layout.addWidget(self.add_button)

        self.full_width_input = LineEdit(self)
        self.full_width_input.setPlaceholderText("请输入需要去掉的文本(多个使用,隔开)")
        self.full_width_input.setClearButtonEnabled(True)
        self.full_width_input.setMaxLength(50)
        self.full_width_input.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))

        self.additional_input_container.addLayout(self.top_input_layout)
        self.additional_input_container.addWidget(self.full_width_input)

        self.action_button = PushButton('添加任务', self)
        self.action_button.setIcon(FIF.ADD)
        self.action_button.clicked.connect(self.button_action)

        self.info_group = QFrame(self)
        self.info_group.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_group.setObjectName("infoGroup")
        self.info_group.setStyleSheet("""
            QFrame#infoGroup {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 8px;
                margin-top: 15px;
            }
        """)
        self.info_group.setFixedHeight(450)

        self.info_layout = QVBoxLayout(self.info_group)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.info_layout.addWidget(self.label)

        control_card = ElevatedCardWidget(self)
        control_card.setFixedHeight(100)
        control_layout = QVBoxLayout()
        control_card.setLayout(control_layout)

        slider_layout = QHBoxLayout()

        self.elapsed_time_label = QLabel("00:00", self)
        slider_layout.addWidget(self.elapsed_time_label)

        self.slider = Slider(Qt.Orientation.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.mousePressEvent = self.on_slider_click
        slider_layout.addWidget(self.slider)

        self.total_time_label = QLabel("00:00", self)
        slider_layout.addWidget(self.total_time_label)

        control_layout.addLayout(slider_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()

        self.rewind_button = ToolButton(self)
        self.rewind_icon = svg_to_icon(rewind_svg)
        self.rewind_button.setIcon(self.rewind_icon)
        self.rewind_button.clicked.connect(self.rewind_video)
        self.rewind_button.setFixedSize(40, 40)
        button_layout.addWidget(self.rewind_button)

        self.play_pause_button = ToolButton(self)
        self.play_icon = svg_to_icon(play_svg)
        self.pause_icon = svg_to_icon(pause_svg)
        self.play_pause_button.setIcon(self.play_icon)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setFixedSize(40, 40)
        button_layout.addWidget(self.play_pause_button)

        self.forward_button = ToolButton(self)
        self.forward_icon = svg_to_icon(forward_svg)
        self.forward_button.setIcon(self.forward_icon)
        self.forward_button.clicked.connect(self.forward_video)
        self.forward_button.setFixedSize(40, 40)
        button_layout.addWidget(self.forward_button)
        button_layout.addStretch()

        control_layout.addLayout(button_layout)

        self.info_layout.addWidget(control_card)
        self.info_layout.setContentsMargins(0, 10, 0, 0)

        frame_button_layout = QHBoxLayout()
        self.get_frame_button = PushButton("获取当前帧图片", self)
        self.get_frame_button.clicked.connect(self.get_current_frame)
        frame_button_layout.addWidget(self.get_frame_button)
        frame_button_layout.setContentsMargins(0, 9, 0, 0)
        self.info_layout.addLayout(frame_button_layout)

        scroll_area_layout.addLayout(file_layout)
        scroll_area_layout.addLayout(capture_layout)
        scroll_area_layout.addLayout(input_layout)
        scroll_area_layout.addLayout(self.additional_input_container)
        scroll_area_layout.addWidget(self.action_button)
        scroll_area_layout.addWidget(self.info_group)
        scroll_area.setWidget(scroll_area_widget)

        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

        self.update_input_visibility()

        self.video_path = ""
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = None
        self.frame_position = 0
        self.is_slider_pressed = False
        self.was_playing_before_drag = False

    def select_file(self):
        """
        选择视频文件，并加载视频。
        """
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '视频文件 (*.mkv *.mov *.wmv *.mp4 *.avi)')
        if file_path:
            self.file_entry.setText(file_path)
            self.load_video(file_path)

    def update_input_visibility(self):
        """
        根据选择的替换选项更新输入框的可见性。
        """
        text_option = self.dropdown4.currentText()
        if text_option == "不替换文本":
            self.top_input1.hide()
            self.top_input2.hide()
            self.add_button.hide()
            self.full_width_input.hide()
            for layout in self.additional_input_layouts:
                for i in range(layout.count()):
                    layout.itemAt(i).widget().hide()
        elif text_option == "替换文本":
            self.top_input1.show()
            self.top_input2.show()
            self.add_button.show()
            self.full_width_input.hide()
            for layout in self.additional_input_layouts:
                for i in range(layout.count()):
                    layout.itemAt(i).widget().show()
        elif text_option == "去掉文本":
            self.top_input1.hide()
            self.top_input2.hide()
            self.add_button.hide()
            self.full_width_input.show()
            for layout in self.additional_input_layouts:
                for i in range(layout.count()):
                    layout.itemAt(i).widget().hide()

    def add_additional_input(self):
        """
        添加额外的替换文本输入框。
        """
        layout = QHBoxLayout()

        additional_input1 = LineEdit(self)
        additional_input1.setPlaceholderText("请输入替换的文本")
        additional_input1.setClearButtonEnabled(True)
        additional_input1.setMaxLength(50)
        additional_input1.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))

        additional_input2 = LineEdit(self)
        additional_input2.setPlaceholderText("请输入替换后的文本")
        additional_input2.setClearButtonEnabled(True)
        additional_input2.setMaxLength(50)
        additional_input2.setValidator(QRegularExpressionValidator(QRegularExpression("[^\s]*")))

        remove_button = PushButton('删除', self)
        remove_button.clicked.connect(lambda: self.remove_additional_input(layout))

        layout.addWidget(additional_input1)
        layout.addWidget(additional_input2)
        layout.addWidget(remove_button)

        self.additional_input_layouts.append(layout)
        self.additional_input_container.addLayout(layout)

    def remove_additional_input(self, layout):
        """
        删除额外的替换文本输入框。

        参数：
            layout (QHBoxLayout): 要删除的布局
        """
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.additional_input_layouts.remove(layout)
        self.additional_input_container.removeItem(layout)

    def clear_inputs(self):
        """
        清空所有输入框。
        """
        self.top_input1.clear()
        self.top_input2.clear()
        self.full_width_input.clear()
        self.file_entry.clear()
        self.input_entry.clear()
        self.capture_entry.clear()
        self.dropdown1.setCurrentIndex(0)
        self.dropdown3.setCurrentIndex(0)
        self.dropdown4.setCurrentIndex(0)
        while self.additional_input_layouts:
            self.remove_additional_input(self.additional_input_layouts[-1])
        self.update_input_visibility()

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

    def button_action(self):
        """
        处理“添加任务”按钮的点击事件。
        """
        file_path = self.file_entry.text()
        task_name = self.input_entry.text()
        model = self.dropdown1.currentText()
        export_option = self.dropdown3.currentText()
        replace_option = self.dropdown4.currentText()
        image_path = self.capture_entry.text()

        if not file_path:
            self.show_info_bar('文件路径不能为空', 'error')
            return
        if not task_name or not re.match(r'^[a-zA-Z0-9\u4e00-\u9fa5]+$', task_name):
            self.show_info_bar('任务名不能为空或包含特殊字符', 'error')
            return
        if not image_path:
            self.show_info_bar('请添加识别图片', 'error')
            return

        replace_texts = []
        if replace_option == "替换文本":
            replace_text1 = self.top_input1.text()
            replace_text2 = self.top_input2.text()
            if not replace_text1 or not replace_text2:
                self.show_info_bar('替换文本不能为空', 'error')
                return
            replace_texts.append(f"{replace_text1}={replace_text2}")
            for layout in self.additional_input_layouts:
                text1 = layout.itemAt(0).widget().text()
                text2 = layout.itemAt(1).widget().text()
                if not text1 or not text2:
                    self.show_info_bar('替换文本不能为空', 'error')
                    return
                replace_texts.append(f"{text1}={text2}")
        elif replace_option == "去掉文本":
            remove_text = self.full_width_input.text()
            if not remove_text:
                self.show_info_bar('去掉文本不能为空', 'error')
                return
            replace_texts.append(f"{remove_text}")
        else:
            replace_texts.append("/")

        replace_text = ",".join(replace_texts)

        if self.task_list_interface.task_exists(task_name):
            self.show_info_bar('当前任务名已存在', 'error')
            return

        image_names = image_path.split(",")
        missing_images = []
        for image_name in image_names:
            if not os.path.exists(os.path.join("temp_images", image_name)):
                missing_images.append(image_name)
        if missing_images:
            self.show_info_bar('识别图片不全', 'error')
            return

        if os.path.exists(os.path.join("Identifyimages", task_name)):
            msg_box = MessageBox(
                title='警告',
                content='Identifyimages文件夹存在与任务名相同名称的文件夹，请修改任务名！',
                parent=self
            )
            msg_box.cancelButton.hide()
            msg_box.buttonLayout.insertStretch(1)
            msg_box.exec()
            return

        task_dir = os.path.join("Identifyimages", task_name)
        os.makedirs(task_dir)

        for image_name in image_names:
            src_path = os.path.join("temp_images", image_name)
            dst_path = os.path.join(task_dir, image_name)
            os.rename(src_path, dst_path)

        self.task_list_interface.add_task(task_name, model, export_option, file_path, replace_option, image_path,
                                          replace_text)
        self.show_info_bar('任务添加成功', 'success')

        self.clear_inputs()

        self.clear_video_display()

    def clear_video_display(self):
        """
        清空视频显示。
        """
        self.video_path = ""
        self.cap = None
        self.timer.stop()
        self.label.clear()
        self.slider.setEnabled(False)
        self.slider.setValue(0)
        self.elapsed_time_label.setText("00:00")
        self.total_time_label.setText("00:00")
        self.play_pause_button.setIcon(self.play_icon)
        self.frame_position = 0

    def load_video(self, file_path):
        """
        加载视频文件，并初始化视频播放。

        参数：
            file_path (str): 视频文件路径
        """
        self.video_path = file_path
        if self.video_path:
            self.pause_video()
            self.cap = cv2.VideoCapture(self.video_path)
            self.slider.setEnabled(True)
            self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_seconds = total_frames / fps
            self.total_time_label.setText(self.format_time(total_seconds))
            self.elapsed_time_label.setText("00:00")
            self.slider.setValue(0)
            self.frame_position = 0
            self.initialize_video()

    def initialize_video(self):
        """
        初始化视频，从头开始播放。
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.update_frame()
            self.pause_video()

    def toggle_play_pause(self):
        """
        切换播放/暂停状态。
        """
        if not self.cap:
            self.show_info_bar("请添加视频文件", "error")
            return

        if self.timer.isActive():
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        """
        播放视频。
        """
        self.timer.start(33)
        self.play_pause_button.setIcon(self.pause_icon)

    def pause_video(self):
        """
        暂停视频。
        """
        self.timer.stop()
        self.play_pause_button.setIcon(self.play_icon)

    def rewind_video(self):
        """
        快退视频。
        """
        if not self.cap:
            self.show_info_bar("请添加视频文件", "error")
            return

        self.pause_video()
        current_position = self.frame_position
        new_position = max(0, current_position - int(self.cap.get(cv2.CAP_PROP_FPS)))
        self.set_frame_position(new_position)

    def forward_video(self):
        """
        快进视频。
        """
        if not self.cap:
            self.show_info_bar("请添加视频文件", "error")
            return

        self.pause_video()
        current_position = self.frame_position
        new_position = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
                           current_position + int(self.cap.get(cv2.CAP_PROP_FPS)))
        self.set_frame_position(new_position)

    def on_slider_pressed(self):
        """
        滑块按下事件处理。
        """
        if self.cap:
            self.is_slider_pressed = True
            self.was_playing_before_drag = self.timer.isActive()
            self.pause_video()

    def on_slider_moved(self, position):
        """
        滑块移动事件处理。

        参数：
            position (int): 滑块位置
        """
        if self.cap:
            self.frame_position = position
            elapsed_seconds = self.frame_position / self.cap.get(cv2.CAP_PROP_FPS)
            self.elapsed_time_label.setText(self.format_time(elapsed_seconds))

    def on_slider_released(self):
        """
        滑块释放事件处理。
        """
        if self.cap:
            self.is_slider_pressed = False
            self.set_frame_position(self.frame_position)
            if self.was_playing_before_drag:
                self.play_video()

    def on_slider_click(self, event):
        """
        滑块点击事件处理。

        参数：
            event (QMouseEvent): 鼠标事件
        """
        if self.cap:
            position = event.pos().x() / self.slider.width() * self.slider.maximum()
            self.slider.setValue(int(position))
            self.on_slider_moved(int(position))
            self.on_slider_released()

    def next_frame(self):
        """
        显示下一帧。
        """
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.update_frame()
        else:
            self.timer.stop()
            self.play_pause_button.setIcon(self.play_icon)

    def set_frame_position(self, position):
        """
        设置帧位置。

        参数：
            position (int): 帧位置
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_position = position
            self.update_frame()

    def update_frame(self):
        """
        更新当前帧显示。
        """
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)
            qImg = qImg.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(qImg))

            if not self.is_slider_pressed:
                self.slider.setValue(self.frame_position)

            elapsed_seconds = self.frame_position / self.cap.get(cv2.CAP_PROP_FPS)
            self.elapsed_time_label.setText(self.format_time(elapsed_seconds))

    def format_time(self, seconds):
        """
        格式化时间显示。

        参数：
            seconds (float): 秒数

        返回：
            str: 格式化后的时间字符串
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02}:{seconds:02}"

    def get_current_frame(self):
        """
        获取当前帧并显示在对话框中。
        """
        if not self.cap:
            self.show_info_bar("请添加视频文件", "error")
            return

        if self.timer.isActive():
            self.pause_video()

        if hasattr(self, 'current_frame') and self.current_frame is not None:
            frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)

            dialog = ImageDialog(qImg, self)
            dialog.exec()

    def view_image(self):
        """
        查看识别图片。
        """
        save_dir = "temp_images"
        png_files_exist = any(file_name.lower().endswith('.png') for file_name in os.listdir(save_dir))

        if not png_files_exist:
            self.show_info_bar('temp_images文件夹缺少识别图片', 'error')
            return

        dialog = IdentifyImagesDialog(self)
        dialog.exec()

    def update_capture_entry(self, filenames):
        """
        更新图片截取路径输入框。

        参数：
            filenames (list[str]): 文件名列表
        """
        self.capture_entry.setText(",".join(filenames))

    def clear_images(self):
        """
        清空识别图片。
        """
        save_dir = "temp_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        png_files_exist = any(file_name.lower().endswith('.png') for file_name in os.listdir(save_dir))

        if not png_files_exist:
            self.show_info_bar('temp_images文件夹缺少识别图片', 'error')
            self.capture_entry.clear()
            return

        for file_name in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        self.capture_entry.clear()
        self.show_info_bar('图片已清空', 'success')