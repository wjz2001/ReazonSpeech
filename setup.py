from setuptools import setup

setup(
    name="ReazonSpeech", # 项目名称
    version="2.3.0",
    py_modules=["asr"], # python 文件名 (不带 .py 后缀)
    entry_points={
        "console_scripts": [
            # 格式： "命令名 = 文件名:函数名"
            "reazonspeech = asr:main",
        ],
    },
)