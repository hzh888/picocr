import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QPen, QPainter
from PyQt6.QtWidgets import QVBoxLayout, QDialog, QLabel, QWidget
from qfluentwidgets import FlowLayout, PushButton, SingleDirectionScrollArea, ElevatedCardWidget


class IdentifyImagesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("查看识别图像")
        self.setFixedSize(800, 600)

        main_layout = QVBoxLayout(self)

        # 创建流式布局
        self.flow_layout = FlowLayout()

        # 创建流式布局的容器
        flow_widget = QWidget()
        flow_widget.setLayout(self.flow_layout)

        # 创建滚动区域
        scroll_area = SingleDirectionScrollArea(orient=Qt.Orientation.Vertical)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(flow_widget)
        scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")
        flow_widget.setStyleSheet("QWidget{background: transparent}")

        # 加载Identifyimages文件夹中的所有png图片
        self.load_images()

        # 将滚动区域添加到主布局中
        main_layout.addWidget(scroll_area)

        # 底部确定按钮布局
        button_widget = ElevatedCardWidget()
        button_layout = QVBoxLayout(button_widget)
        button_layout.setContentsMargins(20, 10, 20, 10)  # 设置按钮布局的边距
        button_layout.setSpacing(10)  # 设置按钮之间的间距

        self.confirm_button = PushButton("确定", self)
        self.confirm_button.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_button, alignment=Qt.AlignmentFlag.AlignCenter)
        button_widget.setFixedHeight(60)  # 设置按钮小部件的固定高度

        main_layout.addWidget(button_widget, alignment=Qt.AlignmentFlag.AlignBottom)
        self.setLayout(main_layout)

    def load_images(self):
        image_folder = "temp_images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for file_name in os.listdir(image_folder):
            if file_name.endswith(".png"):
                file_path = os.path.join(image_folder, file_name)
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    label = QLabel(self)
                    label.setPixmap(self.add_dashed_border_to_pixmap(pixmap, 150, 150))
                    self.flow_layout.addWidget(label)

    def add_dashed_border_to_pixmap(self, pixmap, width, height):
        # 调整 pixmap 尺寸
        scaled_pixmap = pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        bordered_pixmap = QPixmap(scaled_pixmap.size() + QSize(12, 12))  # 增加12像素的边距
        bordered_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(bordered_pixmap)
        painter.drawPixmap(6, 6, scaled_pixmap)  # 在边距内绘制 pixmap

        pen = QPen(Qt.GlobalColor.gray)
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(3, 3, bordered_pixmap.width() - 6, bordered_pixmap.height() - 6)  # 调整绘制位置和尺寸

        painter.end()
        return bordered_pixmap
