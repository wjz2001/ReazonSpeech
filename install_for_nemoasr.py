import subprocess
import sys
import os
import re
import shutil

# --- Helper 函数 ---
def run_command(command_list):
    cmd_str = ' '.join(command_list)
    print(f"\n> 正在执行: {cmd_str}")
    
    try:
        executable = command_list[0]
        # 确保调用的是当前虚拟环境的可执行文件
        if executable in ('pip', 'python'):
            # 优先使用 Scripts (Windows)，其次是 bin (Linux/macOS)
            for subdir in ('Scripts', 'bin'):
                path = os.path.join(sys.prefix, subdir, executable)
                if os.path.exists(path):
                    command_list[0] = path
                    break
        
        subprocess.run(command_list, check=True) # shell=False 是默认值
        print("... 命令执行成功!")

    except subprocess.CalledProcessError as e:
        print(f"!!! 命令执行失败，退出代码: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"!!! 错误: 命令 '{command_list[0]}' 未找到。请确保它已安装并在系统的 PATH 中。")
        sys.exit(1)

def detect_cuda_version():
    """通过 nvidia-smi 检测 CUDA 版本"""
    if not shutil.which("nvidia-smi"):
        print("--- 未找到 'nvidia-smi' 命令，将跳过 PyTorch 预安装。")
        return None
        
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        match = re.search(r"CUDA Version: (\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = match.groups()
            print(f"--- 检测到 CUDA 驱动版本: {major}.{minor}")
            return int(major), int(minor)
        else:
            print("--- 在 'nvidia-smi' 的输出中未能找到 CUDA 版本信息，将跳过 PyTorch 预安装。")
            return None
    except Exception as e:
        print(f"--- 执行 'nvidia-smi' 时发生错误: {e}")
        print("--- 无法确定 CUDA 版本，将跳过 PyTorch 预安装。")
        return None

def patch_exp_manager():
    """在已安装的 nemo_toolkit 中查找并修补 exp_manager.py 文件 (仅限 Windows)"""
    print("\n> 尝试修补 nemo_toolkit...")
    try:
        # sys.prefix 在虚拟环境中指向 venv 目录
        site_packages_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
        file_path = os.path.join(site_packages_path, 'nemo', 'utils', 'exp_manager.py')

        if not os.path.exists(file_path):
            print(f"!!! 错误: 未能找到文件 {file_path}")
            print("!!! 修补失败。请确认 'nemo_toolkit' 是否已正确安装。")
            sys.exit(1)

        print(f"--- 正在修补文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_line = 'rank_termination_signal: signal.Signals = signal.SIGKILL'
        patched_line = "rank_termination_signal: signal.Signals = getattr(signal, 'SIGKILL', signal.SIGTERM)"

        if patched_line in content:
            print("--- 文件已被修补过，跳过此步骤。")
            return

        if original_line not in content:
            print("!!! 警告: 未找到需要修补的目标代码行。nemo_toolkit 的版本可能已发生变化。")
            return
        
        new_content = content.replace(original_line, patched_line)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("... 修补成功!")

    except Exception as e:
        print(f"!!! 修补过程中发生意外错误: {e}")
        sys.exit(1)

def main():
    print("--- 开始执行安装脚本 ---")

    # 步骤 1: 升级 pip
    run_command(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # 步骤 2: 智能预安装 PyTorch
    cuda_version = detect_cuda_version()
    torch_variant_to_install = None

    if cuda_version:
        major, _ = cuda_version
        if major >= 12:
            print(f"--- 匹配到 CUDA {major}.x 系列，准备预安装 PyTorch (cu121)。")
            torch_variant_to_install = "cu121"
        elif major == 11:
            print(f"--- 匹配到 CUDA {major}.x 系列，准备预安装 PyTorch (cu118)。")
            torch_variant_to_install = "cu118"
        else:
            print(f"--- 警告: 不支持 CUDA 版本 {major}.x。将跳过 PyTorch 预安装。")

        if torch_variant_to_install:
            pytorch_command = [
                'pip', 'install', 'torch', 'torchaudio',
                '--index-url', f'https://download.pytorch.org/whl/{torch_variant_to_install}'
            ]
            run_command(pytorch_command)
    else:
        print("--- 未检测到 CUDA 或 GPU，将跳过 PyTorch 预安装。")

    # 步骤 3: 一次性安装所有核心包，并精确锁定版本
    print("\n--- 正在安装核心依赖包 ---")
    core_packages_command = [
        'pip', 'install',
        'nemo_toolkit[asr]==2.1.0',
        'pyannote.pipeline==3.0.1',
        'pyannote.core==6.0.1',
        'pyannote.database==6.1.0',
        'pyannote.metrics==4.0.0',
        'onnxruntime==1.22.1',
    ]
    run_command(core_packages_command)
    
    # 步骤 4: 如果是 Windows，执行 Nemo 的特殊修补
    if sys.platform == "win32":
        patch_exp_manager()
    
    # 步骤 5: 以可编辑模式安装本地的 ReazonSpeech ASR 包
    reazonspeech_install_command = ['pip', 'install', '-e', './pkg/nemo-asr']
    run_command(reazonspeech_install_command)
    
    print("\n--- 安装流程执行完毕！ ---")

if __name__ == "__main__":
    main()