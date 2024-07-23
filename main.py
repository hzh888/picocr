# coding:utf-8
import sys
from PyQt6.QtWidgets import QApplication
from future.moves import multiprocessing
from qfluentwidgets import (setTheme, Theme)
from sidebar import Window

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    #window = Window()
    #window.setFixedSize(800, 600)
    w = Window()
    w.show()
    #window.show()
    sys.exit(app.exec())
