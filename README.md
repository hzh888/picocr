<p align="center">
  <a href="https://github.com/hzh888/picocr"><img src="https://raw.githubusercontent.com/hzh888/picocr/main/resource/logo.png" alt="picocr" width="115" /></a>
</p>

<div align="center">

# PicOCR

***✨ 视频指定区域文本提取工具 ✨***</div>
<p align="center">
  <a href="https://github.com/hzh888/picocr"><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10-8A2BE2?style=flat"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub License" src="https://img.shields.io/github/license/hzh888/picocr"></a>
  <a href="https://github.com/hzh888/picocr/releases"><img alt="GitHub release" src="https://img.shields.io/github/v/release/hzh/picocr?style=flat"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/hzh888/picocr?style=flat"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub forks" src="https://img.shields.io/github/forks/hzh888/picocr?style=flat"></a>
</p>

<a href="https://github.com/hzh888/picocr"><img src="https://raw.githubusercontent.com/hzh888/picocr/main/resource/tool.png" alt="picocr"></a>

## 什么是PicOCR?
基于Qt框架+QFluentWidgets组件库开发的视频图像OCR识别开源工具，集成了多线程技术，能够高效处理视频中的图像并通过内置的OCR模型（Ddddocr和PaddleOCR）识别文本，轻松导出识别结果至Excel表格，并支持文本替换和去除文本功能，灵活满足不同需求，友好的GUI用户界面和实时任务进度显示，操作简便且直观。

## 功能
- [x] 多种OCR模型
- [x] 任务执行进度显示
- [x] 多任务执行
- [x] 替换文本和去除文本
- [x] Excel表格数据导出
- [x] 自定义视频识别区域
- [x] 视频多区域识别
- [ ] USB投屏实时识别
- [ ] 更多功能欢迎提交意见

## 支持的OCR模型
| 模型名字 |
| :-----: |
| Ddddocr |
| PaddleOCR | 
| 更多模型正在更新 | 

## 下载
[PicOCR.zip](releases)

## 关于项目以及联系方式
软件的诞生是因为公司原来的识别工具不准确并且开发人员已经离职许久，身为测试人员的我又需要识别工具，所以这款软件诞生了，当然，我对代码有亿点不太熟，写的很垃圾，属于现学现写，人和代码，有一个能跑就行。

| 联系方式 | 账号 |
| :-----: | :-----: |
| 邮箱 | 2695197253@qq.com |
| QQ | 2695197253 |

## 遇到问题？
如您的问题未在此处列出或遇到不明BUG等情况，您可以[搜索或提交issue](issues)。

## 常见问题
- 问：软件开源吗？  
  答：软件是开源的。
- 问：软件需要网络嘛？  
  答：不需要，识别部分均使用本地化模型。
- 问：遇到BUG或者有想增加的功能怎么办？  
  答：[搜索或提交issue](issues)
- 问：可以同时执行多少个任务？   
  答：只要你电脑性能足够强，理论无限个，推荐同时执行3个任务，怕你电脑扛不住。
- 问：目前支持多少个识别OCR模型？   
  答：现在只支持Ddddocr模型和PaddleOCR模型，未来计划引入更多开源OCR模型
