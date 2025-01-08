# coding:utf-8
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QIcon, QDesktopServices
from PyQt6.QtWidgets import QFrame, QHBoxLayout
from qfluentwidgets import (MessageBox, NavigationAvatarWidget)
from qfluentwidgets import NavigationItemPosition, FluentWindow, SubtitleLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from main_interface import Addtaskinterface
from task_list_interface import TaskListInterface
from settingspage import SettingsPage  # 导入 SettingsPage

class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)
        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignmentFlag.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))


class Window(FluentWindow):
    def __init__(self):
        super().__init__()
        self.task_list_interface = TaskListInterface(self)
        self.searchInterface = Addtaskinterface(self.task_list_interface, self)
        self.settingInterface = SettingsPage(self)  # 初始化 SettingsPage
        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        self.navigationInterface.addSeparator()
        self.addSubInterface(self.searchInterface, FIF.ADD, '新增任务')
        self.addSubInterface(self.task_list_interface, FIF.IOT, '任务列表')
        self.navigationInterface.addWidget(
            routeKey='avatar',
            widget=NavigationAvatarWidget('赤瞳', 'resource/shoko.png'),
            onClick=self.showMessageBox,
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(self.settingInterface, FIF.SETTING, '设置', NavigationItemPosition.BOTTOM)  # 添加设置页面

    def initWindow(self):
        self.resize(980, 695)
        self.setWindowIcon(QIcon('resource/logo.png'))
        self.setWindowTitle('PicOCR V1.4')

    def showMessageBox(self):
        w = MessageBox(
            '说明',
            '本软件采用Python编写，UI界面使用PyQT6+QFluentWidgets组件库，软件已经开源至Github，软件的诞生是因为公司原来的识别工具不准确并且开发人员已经离职许久，身为测试人员的我又需要识别工具，所以只能利用空余时间开发了这款软件，当然，我对代码不太熟，写的很垃圾，属于现学现写，人和代码，有一个能跑就行。',
            self
        )
        w.yesButton.setText('访问Github仓库')
        w.cancelButton.setText('下次一定')
        if w.exec():
            QDesktopServices.openUrl(QUrl("https://github.com/hzh888/picocr"))
