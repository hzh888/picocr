# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['G:\\chitoya\\picocr\\.venv\\Lib\\site-packages\\paddleocr','G:\\chitoya\\picocr\\.venv\\Lib\\site-packages\\paddle\\libs','G:\\chitoya\\picocr\\.venv\\Lib\\site-packages'],
    binaries=[('G:\\chitoya\\picocr\\.venv\\Lib\\site-packages\\paddleocr','.'),('G:\\chitoya\\picocr\\.venv\\Lib\\site-packages\\paddle\\libs', '.'),('G:\chitoya\picocr\.venv\Lib\site-packages\shapely','.')],
    datas=[('./models/onnxruntime_providers_shared.dll','onnxruntime\\capi'),('./models/common.onnx','ddddocr'),('./models/common_old.onnx','ddddocr')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PicOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resource\\logo.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PicOCR',
)
