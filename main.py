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
    w = Window()
    w.show()
    sys.exit(app.exec())
