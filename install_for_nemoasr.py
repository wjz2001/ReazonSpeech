import subprocess
import sys
import os
import platform
import re
import shutil

def check_runtime_requirements():
    # Python 版本锁死：只支持 3.10.x
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print(f"当前版本：{sys.version.split()[0]} 不符合要求，请使用 Python 3.10.x"
        )
        sys.exit(1)

    # Intel Mac 直接中止
    if sys.platform == "darwin" and platform.machine().lower() in ("x86_64", "amd64"):
            print("由于 NeMo 在 Intel macOS 上已弃用，所以本项目不支持您的电脑，有条件的话请安装 arm64 Python")
            sys.exit(1)


def check_ffmpeg():
    # ffmpeg 自检，找不到就提示 brew 安装
    if shutil.which("ffmpeg"):
        return

    print("未找到 ffmpeg，本项目运行需要 ffmpeg。请先安装 ffmpeg 并加入 PATH 或把它放在本项目根目录中")

    if sys.platform == "darwin":
        print("请先 brew install ffmpeg libsndfile")

def detect_cuda_version():
    """通过 nvidia-smi 检测 CUDA 版本"""
    if not shutil.which("nvidia-smi"):
        print("未找到命令 nvidia-smi，将跳过安装 PyTorch")
        return None
        
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        match = re.search(r"CUDA Version: (\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = match.groups()
            print(f"检测到 CUDA 版本：{major}.{minor}")
            return int(major), int(minor)
        else:
            print("在 nvidia-smi 的输出中未找到 CUDA 版本信息，将跳过安装 PyTorch")
            return None
    except Exception as e:
        print(f"执行 nvidia-smi 时发生错误：{e}，无法确定 CUDA 版本，将跳过安装 PyTorch")
        return None

def patch_exp_manager():
    """在已安装的 nemo_toolkit 中查找并修补 exp_manager.py 文件 (仅限 Windows)"""
    print("\n> 尝试修补 nemo_toolkit ……")
    try:
        import nemo
        # 找到 nemo 包的 __init__.py 文件
        nemo_init_path = nemo.__file__
        # 从 __init__.py 向上找到 nemo 包的根目录
        nemo_root_path = os.path.dirname(nemo_init_path)
        # 构造目标文件的绝对路径
        file_path = os.path.join(nemo_root_path, 'utils', 'exp_manager.py')

        if not os.path.exists(file_path):
            print(f"未能找到文件 {file_path}，修补失败，请确认 nemo_toolkit 是否已正确安装")
            sys.exit(1)

        print(f"正在修补文件：{file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_line = "rank_termination_signal: signal.Signals = signal.SIGKILL"
        patched_line = "rank_termination_signal: signal.Signals = signal.SIGKILL if os.name != 'nt' else signal.SIGTERM"

        if patched_line in content:
            print("文件已被修补")
            return

        if original_line not in content:
            print("未找到需要修补的目标代码行")
            return
        
        new_content = content.replace(original_line, patched_line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("修补成功")

    except Exception as e:
        print(f"修补过程中发生意外错误：{e}")
        sys.exit(1)

def run_command(command_list):
    cmd_str = ' '.join(command_list)
    print(f"\n> 正在执行：{cmd_str}")
    
    try:
        # 确保调用的是当前虚拟环境的可执行文件
        if command_list[0] == 'python':
            command_list[0] = sys.executable
        elif command_list[0] == 'pip':
            command_list = [sys.executable, '-m', 'pip'] + command_list[1:]
        
        subprocess.run(
            command_list,
            check=True,
            capture_output=True,
            text=True,
        )
        print("命令执行成功")

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，失败代码：{e.returncode}")
        # 仅失败时打印 stdout/stderr
        if e.stdout:
            print("\n=== stdout ===")
            print(e.stdout)
        if e.stderr:
            print("\n=== stderr ===")
            print(e.stderr)

        # 仅 mac 下、且这次失败是 pip 安装 nemo_toolkit 相关时，给出 youtokentome 预修复提示
        if sys.platform == "darwin":
            print("如果你的报错信息中包含 youtokentome 或 No module named 'Cython'，可尝试先执行：")
            print("python -m pip install git+https://github.com/LahiLuk/YouTokenToMe")
            print("然后重新运行本安装脚本")

        sys.exit(1)
    except FileNotFoundError:
        print(f"命令 {command_list[0]} 未找到，请确保它已安装并在系统 PATH 中")
        sys.exit(1)

def main():
    print("开始执行安装脚本")

    check_runtime_requirements()
    check_ffmpeg()

    # 升级 pip
    run_command(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # 智能预安装 PyTorch
    cuda_version = detect_cuda_version()
    torch_variant_to_install = None

    if cuda_version:
        major, _ = cuda_version
        if major >= 12:
            print(f"匹配到 CUDA {major}.x 系列，准备安装 PyTorch (cu121)")
            torch_variant_to_install = "cu121"
        elif major == 11:
            print(f"匹配到 CUDA {major}.x 系列，准备安装 PyTorch (cu118)")
            torch_variant_to_install = "cu118"
        else:
            print(f"不支持 CUDA 版本 {major}.x，将跳过安装 PyTorch")

        if torch_variant_to_install:
            pytorch_command = [
                'pip', 'install', 'torch', 'torchaudio',
                '--index-url', f'https://download.pytorch.org/whl/{torch_variant_to_install}'
            ]
            run_command(pytorch_command)
    else:
        print("未检测到 CUDA 或 GPU，将跳过安装 PyTorch")

    # 一次性安装所有核心包，并锁定版本
    print("正在安装依赖……")
    core_packages_command = [
        'pip', 'install',
        'nemo_toolkit[asr]==2.1.0',
        'pyannote.pipeline==3.0.1',
        'pyannote.core==6.0.1',
        'pyannote.database==6.1.0',
        'pyannote.metrics==4.0.0',
        'onnxruntime==1.22.1',
        'annotated-doc==0.0.4',
        'fastapi==0.124.2',
        'python-multipart==0.0.20',
        'starlette==0.50.0',
        'uvicorn==0.38.0',
    ]
    run_command(core_packages_command)
    
    # 如果是 Windows，执行 Nemo 的特殊修补
    if sys.platform == "win32":
        patch_exp_manager()
    
    # 以可编辑模式安装本地的 ReazonSpeech ASR 包
    reazonspeech_install_command = ['pip', 'install', '-e', './pkg/nemo-asr']
    run_command(reazonspeech_install_command)

    # 以可编辑模式安装当前根目录项目
    run_command(['pip', 'install', '-e', '.'])
    
    print("安装流程执行完毕")

if __name__ == "__main__":
    main()