from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qfluentwidgets import LineEdit, PrimaryPushButton, InfoBar, InfoBarPosition, CardWidget, ComboBox, BodyLabel
from qfluentwidgets import QConfig, OptionsConfigItem, OptionsValidator, qconfig

# 定义常量，避免重复定义
GENERAL_BASIC_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token="
ACCURATE_BASIC_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token="

# 定义配置类，用于管理百度 OCR 和授权码配置
class Config(QConfig):
    ocrApiUrl = OptionsConfigItem(
        "BaiduOCR", "ApiUrl", "", OptionsValidator([
            GENERAL_BASIC_URL,
            ACCURATE_BASIC_URL
        ])
    )
    apiKey = OptionsConfigItem("BaiduOCR", "ApiKey", "")
    secretKey = OptionsConfigItem("BaiduOCR", "SecretKey", "")
    authCode = OptionsConfigItem("Authorization", "AuthCode", "")  # 独立的授权码配置项

cfg = Config()
qconfig.load("config.json", cfg)

class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPage")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 使组件靠上排列

        # OCR 配置卡片
        self.ocrConfigCard = CardWidget(self)
        ocrCardLayout = QVBoxLayout()

        # 添加较小的标题
        ocrTitleLabel = BodyLabel("百度OCR配置", self)
        ocrCardLayout.addWidget(ocrTitleLabel)

        # 创建一个水平布局，将下拉框、输入框和按钮放在同一行
        ocrHBoxLayout = QHBoxLayout()

        # 创建下拉框
        self.comboBox = ComboBox(self)
        self.comboBox.addItems(["标准版", "高精度版"])
        ocrHBoxLayout.addWidget(self.comboBox)

        # 创建输入框
        self.apiKeyInput = LineEdit(self)
        self.apiKeyInput.setPlaceholderText("请输入API_KEY")
        ocrHBoxLayout.addWidget(self.apiKeyInput)

        self.secretKeyInput = LineEdit(self)
        self.secretKeyInput.setPlaceholderText("请输入SECRET_KEY")
        ocrHBoxLayout.addWidget(self.secretKeyInput)

        # 创建保存按钮
        self.saveButton = PrimaryPushButton("保存", self)
        ocrHBoxLayout.addWidget(self.saveButton)

        # 将水平布局添加到卡片布局中
        ocrCardLayout.addLayout(ocrHBoxLayout)
        self.ocrConfigCard.setLayout(ocrCardLayout)
        layout.addWidget(self.ocrConfigCard)

        # 授权码卡片
        self.authCard = CardWidget(self)
        authCardLayout = QVBoxLayout()

        # 添加授权码标题
        authTitleLabel = BodyLabel("云端OCR授权码配置", self)
        authCardLayout.addWidget(authTitleLabel)

        # 创建一个水平布局，将输入框和按钮放在同一行
        authHBoxLayout = QHBoxLayout()

        # 创建授权码输入框
        self.authCodeInput = LineEdit(self)
        self.authCodeInput.setPlaceholderText("请输入云端OCR授权码")
        authHBoxLayout.addWidget(self.authCodeInput)

        # 创建保存授权码按钮
        self.saveAuthButton = PrimaryPushButton("保存", self)
        authHBoxLayout.addWidget(self.saveAuthButton)

        # 将水平布局添加到卡片布局中
        authCardLayout.addLayout(authHBoxLayout)
        self.authCard.setLayout(authCardLayout)
        layout.addWidget(self.authCard)

        # 根据配置文件中的值设置控件的默认选项和内容
        if cfg.ocrApiUrl.value == GENERAL_BASIC_URL:
            self.comboBox.setCurrentText("标准版")
        elif cfg.ocrApiUrl.value == ACCURATE_BASIC_URL:
            self.comboBox.setCurrentText("高精度版")

        self.apiKeyInput.setText(cfg.apiKey.value)
        self.secretKeyInput.setText(cfg.secretKey.value)
        self.authCodeInput.setText(cfg.authCode.value)  # 自动从配置中加载授权码

        # 连接保存按钮的点击事件
        self.saveButton.clicked.connect(self.saveConfig)
        self.saveAuthButton.clicked.connect(self.saveAuthCode)

        self.setLayout(layout)

    def saveConfig(self):
        apiKey = self.apiKeyInput.text()
        secretKey = self.secretKeyInput.text()

        # 检查输入框是否为空
        if not apiKey:
            InfoBar.error(
                title='提示',
                content='请输入API_KEY',
                isClosable=True,
                duration=2000,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        if not secretKey:
            InfoBar.error(
                title='提示',
                content='请输入SECRET_KEY',
                isClosable=True,
                duration=2000,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        # 根据下拉框选项设置对应的 URL
        selectedText = self.comboBox.currentText()
        if selectedText == "标准版":
            cfg.ocrApiUrl.value = GENERAL_BASIC_URL
        elif selectedText == "高精度版":
            cfg.ocrApiUrl.value = ACCURATE_BASIC_URL

        # 保存 API_KEY 和 SECRET_KEY
        cfg.apiKey.value = apiKey
        cfg.secretKey.value = secretKey

        # 将配置保存到 config.json
        cfg.save()

        InfoBar.success(
            title='提示',
            content='配置已保存',
            isClosable=True,
            duration=2000,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT
        )

    def saveAuthCode(self):
        authCode = self.authCodeInput.text()

        # 检查授权码输入框是否为空
        if not authCode:
            InfoBar.error(
                title='提示',
                content='请输入云端OCR授权码',
                isClosable=True,
                duration=2000,
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        # 保存授权码
        cfg.authCode.value = authCode

        # 将配置保存到 config.json
        cfg.save()

        InfoBar.success(
            title='提示',
            content='云端OCR授权码已保存',
            isClosable=True,
            duration=2000,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT
        )
