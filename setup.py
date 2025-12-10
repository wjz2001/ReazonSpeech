import sys
from setuptools import setup

# 拦截直接运行 (如: python setup.py)
if __name__ == "__main__":
    print("此文件不可运行")
    # 直接退出，不显示后续的 setup 帮助信息或其他报错
    sys.exit(1)

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