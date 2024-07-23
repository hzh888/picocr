import os
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QDialog, QWidget
from qfluentwidgets import PushButton, InfoBar, InfoBarPosition, ElevatedCardWidget, SingleDirectionScrollArea, FluentIcon as FIF


class ImageLabel(QLabel):
    images_saved = pyqtSignal(list)

    def __init__(self, pixmap, parent=None):
        """
        初始化ImageLabel类，用于显示和处理图像的标签。

        参数：
            pixmap (QPixmap): 要显示的图像
            parent (QWidget, 可选): 父级窗口
        """
        super().__init__(parent)
        self.original_pixmap = pixmap
        self.current_pixmap = pixmap
        self.scale_factor = 1.0
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rectangles = []
        self.undo_stack = []
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.adjust_size()

    def adjust_size(self):
        """
        根据缩放因子调整标签大小。
        """
        new_size = self.original_pixmap.size() * self.scale_factor
        self.setFixedSize(new_size)

    def mousePressEvent(self, event):
        """
        处理鼠标按下事件，开始绘制矩形。

        参数：
            event (QMouseEvent): 鼠标事件
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start_point = self.mapToPixmap(event.position().toPoint())
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        """
        处理鼠标移动事件，更新矩形的结束点并重绘。

        参数：
            event (QMouseEvent): 鼠标事件
        """
        if self.drawing:
            self.end_point = self.mapToPixmap(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event):
        """
        处理鼠标释放事件，完成矩形绘制并保存矩形区域。

        参数：
            event (QMouseEvent): 鼠标事件
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.end_point = self.mapToPixmap(event.position().toPoint())
            rect = QRect(self.start_point, self.end_point).normalized()
            self.rectangles.append(self.scale_rect(rect, 1 / self.scale_factor))
            self.update()

    def paintEvent(self, event):
        """
        绘制事件，用于绘制图像和矩形。

        参数：
            event (QPaintEvent): 绘制事件
        """
        super().paintEvent(event)
        painter = QPainter(self)
        self.current_pixmap = self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(0, 0, self.current_pixmap)

        pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)

        for rect in self.rectangles:
            scaled_rect = self.scale_rect(rect, self.scale_factor)
            painter.drawRect(scaled_rect)

        if self.drawing:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

    def resizeEvent(self, event):
        """
        调整大小事件，更新图像显示。

        参数：
            event (QResizeEvent): 调整大小事件
        """
        self.update()

    def mapToPixmap(self, point):
        """
        将窗口坐标映射到图像坐标。

        参数：
            point (QPoint): 窗口坐标点

        返回：
            QPoint: 图像坐标点
        """
        scaled_pixmap = self.current_pixmap

        px = max(0, min(scaled_pixmap.width() - 1, round(point.x())))
        py = max(0, min(scaled_pixmap.height() - 1, round(point.y())))

        return QPoint(px, py)

    def scale_point(self, point, factor):
        """
        按因子缩放点坐标。

        参数：
            point (QPoint): 原始点坐标
            factor (float): 缩放因子

        返回：
            QPoint: 缩放后的点坐标
        """
        return QPoint(round(point.x() * factor), round(point.y() * factor))

    def scale_rect(self, rect, factor):
        """
        按因子缩放矩形区域。

        参数：
            rect (QRect): 原始矩形
            factor (float): 缩放因子

        返回：
            QRect: 缩放后的矩形
        """
        top_left = self.scale_point(rect.topLeft(), factor)
        bottom_right = self.scale_point(rect.bottomRight(), factor)
        return QRect(top_left, bottom_right)

    def undo_rectangle(self):
        """
        撤销最后一个绘制的矩形。
        """
        if self.rectangles:
            last_rect = self.rectangles.pop()
            self.undo_stack.append(last_rect)
            self.update()

    def redo_rectangle(self):
        """
        重做最后一个撤销的矩形。
        """
        if self.undo_stack:
            rect = self.undo_stack.pop()
            self.rectangles.append(rect)
            self.update()

    def zoom_in(self):
        """
        放大图像。
        """
        self.scale_factor = min(3.0, self.scale_factor + 0.1)
        self.adjust_size()
        self.update()

    def zoom_out(self):
        """
        缩小图像。
        """
        self.scale_factor = max(0.3, self.scale_factor - 0.1)
        self.adjust_size()
        self.update()

    def save_cropped_images(self):
        """
        保存裁剪后的图像。
        """
        if not self.rectangles:
            InfoBar.error(
                title="提示",
                content="请框选识别区域",
                isClosable=True,
                duration=2000,
                parent=self.window(),
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        save_dir = "temp_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        saved_files = []
        for i, rect in enumerate(self.rectangles, start=1):
            scale_x = self.original_pixmap.width() / (self.current_pixmap.width() / self.scale_factor)
            scale_y = self.original_pixmap.height() / (self.current_pixmap.height() / self.scale_factor)

            orig_x = rect.left() * scale_x
            orig_y = rect.top() * scale_y
            orig_w = rect.width() * scale_x
            orig_h = rect.height() * scale_y

            orig_rect = QRect(int(orig_x), int(orig_y), int(orig_w), int(orig_h))

            cropped_image = self.original_pixmap.copy(orig_rect)
            save_path = os.path.join(save_dir, f"ocr_{i}.png")
            cropped_image.save(save_path, "PNG")
            saved_files.append(save_path)

        self.images_saved.emit(saved_files)
        InfoBar.success(
            title="提示",
            content="图片已保存到temp_images文件夹",
            isClosable=True,
            duration=2000,
            parent=self.window(),
            position=InfoBarPosition.TOP_RIGHT
        )

        self.rectangles.clear()
        self.update()


class ImageDialog(QDialog):
    def __init__(self, image, parent=None):
        """
        初始化ImageDialog类，用于显示和处理图像的对话框。

        参数：
            image (QImage): 要显示的图像
            parent (QWidget, 可选): 父级窗口
        """
        super().__init__(parent)
        self.setWindowTitle("截取识别区域")
        self.setFixedSize(800, 600)

        self.image_label = ImageLabel(QPixmap.fromImage(image), self)

        self.scroll_area = SingleDirectionScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")

        scroll_area_widget = QWidget()
        scroll_area_widget.setStyleSheet("QWidget{background: transparent}")

        scroll_area_layout = QVBoxLayout(scroll_area_widget)
        scroll_area_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area_layout.addWidget(self.scroll_area)

        card = ElevatedCardWidget(self)
        card.setFixedHeight(65)
        button_layout = QHBoxLayout()

        self.undo_button = PushButton("撤回", self)
        self.undo_button.setIcon(FIF.LEFT_ARROW)
        self.undo_button.clicked.connect(self.image_label.undo_rectangle)

        self.redo_button = PushButton("回退", self)
        self.redo_button.setIcon(FIF.RIGHT_ARROW)
        self.redo_button.clicked.connect(self.image_label.redo_rectangle)

        self.zoom_in_button = PushButton("放大", self)
        self.zoom_in_button.setIcon(FIF.ZOOM_IN)
        self.zoom_in_button.clicked.connect(self.image_label.zoom_in)

        self.zoom_out_button = PushButton("缩小", self)
        self.zoom_out_button.setIcon(FIF.ZOOM_OUT)
        self.zoom_out_button.clicked.connect(self.image_label.zoom_out)

        self.confirm_button = PushButton("确定", self)
        self.confirm_button.setIcon(FIF.ADD)
        self.confirm_button.clicked.connect(self.image_label.save_cropped_images)

        button_layout.addStretch()
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.redo_button)
        button_layout.addWidget(self.zoom_in_button)
        button_layout.addWidget(self.zoom_out_button)
        button_layout.addWidget(self.confirm_button)
        button_layout.addStretch()

        card.setLayout(button_layout)

        layout = QVBoxLayout()
        layout.addWidget(scroll_area_widget)
        layout.addWidget(card)
        self.setLayout(layout)

        self.image_label.images_saved.connect(self.handle_images_saved)

    def handle_images_saved(self, saved_files):
        """
        处理图片保存完成事件，更新父窗口中的捕获条目。

        参数：
            saved_files (list[str]): 保存的文件路径列表
        """
        filenames = [os.path.basename(file) for file in saved_files]
        self.parent().update_capture_entry(filenames)
