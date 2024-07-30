<p align="center">
  <a href="https://github.com/hzh888/picocr"><img src="https://raw.githubusercontent.com/hzh888/picocr/main/resource/logo.png" alt="picocr" width="115" /></a>
</p>

<div align="center">

# PicOCR

***✨ 视频自定义区域文本提取工具 ✨***</div>
<p align="center">
  <a href="https://github.com/hzh888/picocr"><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10-8A2BE2?style=flat"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub License" src="https://img.shields.io/github/license/hzh888/picocr"></a>
  <a href="https://github.com/hzh888/picocr/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/hzh888/picocr?style=flat&color=32CD32"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/hzh888/picocr?style=flat"></a>
  <a href="https://github.com/hzh888/picocr"><img alt="GitHub forks" src="https://img.shields.io/github/forks/hzh888/picocr?style=flat"></a>
</p>

<a href="https://github.com/hzh888/picocr"><img src="https://raw.githubusercontent.com/hzh888/picocr/main/resource/tool.png" alt="picocr"></a>

## 什么是PicOCR?
基于Qt框架+QFluentWidgets组件库开发的视频自定义图像区域OCR识别开源工具，集成了多线程技术，通过内置的OCR开源模型进行识别自定义视频区域内的文本，识别区域支持多区域识别，识别完成后结果会导出Excel表格，同时支持自定义文本替换和自定义去除文本功能。

## 当前功能
- [x] 内置多个OCR模型
- [x] 任务执行进度显示
- [x] 多任务执行
- [x] 自定义替换文本功能
- [x] 去除指定文本功能
- [x] Excel表格数据导出
- [x] 自定义视频识别区域
- [x] 视频多区域识别
- [x] 转换灰度图像自由选择
- [x] 自定义秒数提取帧
- [x] 任务停止功能

## 计划开发
- [ ] USB投屏实时识别

## 支持的OCR模型
| 模型名字 |
| :-----: |
| Ddddocr |
| PaddleOCR | 

## 下载
[PicOCR.zip](https://github.com/hzh888/picocr/releases/)

## 关于项目以及联系方式
软件的诞生是因为公司原来的识别工具不准确并且开发人员已经离职许久，身为测试人员的我又需要识别工具，所以这款软件诞生了，当然，我对代码有亿点不太熟，写的很垃圾，属于现学现写，人和代码，有一个能跑就行。

| 联系方式 | 账号 |
| :-----: | :-----: |
| 邮箱 | 2695197253@qq.com |
| QQ | 2695197253 |

## 遇到问题？
如您的问题未在此处列出或遇到不明BUG等情况，您可以[搜索或提交issue](https://github.com/hzh888/picocr/issues)。

## 常见问题
- 问：软件开源吗？  
  答：开源。
- 问：软件需要网络嘛？  
  答：不需要。
- 问：遇到BUG或者有想增加的功能怎么办？  
  答：[搜索或提交issue](https://github.com/hzh888/picocr/issues)
- 问：可以同时执行多少个任务？   
  答：只要你电脑性能足够强，理论无限个，推荐最高同时执行3个任务，多了怕你电脑扛不住。
- 问：目前支持多少个识别OCR模型？   
  答：现在只支持Ddddocr模型和PaddleOCR模型，未来计划引入更多开源OCR模型。
