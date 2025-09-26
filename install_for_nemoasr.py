import subprocess
import sys
import os

# --- 配置信息 ---
# 您的主项目路径
MAIN_PROJECT_PATH = "ReazonSpeech/pkg/nemo-asr"
# 有问题的依赖包 (仅在 Windows 上需要)
NEMO_TOOLKIT_PACKAGE = "nemo_toolkit[asr]<2.2"

def run_command(command):
    """执行一个 shell 命令，并提供中文提示，在失败时退出"""
    # 如果 command 是列表，为了打印方便，将其转换成字符串
    cmd_str = ' '.join(command) if isinstance(command, list) else command
    print(f"\n> 正在执行: {cmd_str}")
    try:
        # 在 Windows 上，有时需要 shell=True
        use_shell = sys.platform == "win32"
        subprocess.run(command, check=True, shell=use_shell)
        print("... 命令执行成功!")
    except subprocess.CalledProcessError as e:
        print(f"!!! 命令执行失败，退出代码: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"!!! 错误: 命令 '{command[0]}' 未找到。请确保它已安装并在系统的 PATH 中。")
        sys.exit(1)

def detect_cuda_version():
    """通过 nvidia-smi 检测 CUDA 版本"""
    if not shutil.which("nvidia-smi"):
        print("--- 未找到 'nvidia-smi' 命令，将安装 CPU 版本的 PyTorch。")
        return None

    try:
        # 执行 nvidia-smi 命令并获取输出
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 使用正则表达式匹配 "CUDA Version: XX.X"
        match = re.search(r"CUDA Version: (\d+)\.(\d+)", output)
        if match:
            major, minor = match.groups()
            print(f"--- 检测到 CUDA 驱动版本: {major}.{minor}")
            return int(major), int(minor)
        else:
            print("--- 在 'nvidia-smi' 的输出中未能找到 CUDA 版本信息，将安装 CPU 版本的 PyTorch。")
            return None
    except Exception as e:
        print(f"--- 执行 'nvidia-smi' 时发生错误: {e}")
        print("--- 无法确定 CUDA 版本，将安装 CPU 版本的 PyTorch。")
        return None

def get_pytorch_install_command():
    """根据检测到的 CUDA 版本，生成对应的 PyTorch 安装命令"""
    cuda_version = detect_cuda_version()
    
    # 基础包
    packages = ['torch', 'torchaudio']
    
    if cuda_version:
        major, _ = cuda_version
        # 根据 PyTorch 官网提供的版本支持进行映射
        if major >= 12:
            print("--- 匹配到 CUDA 12.x 系列，准备安装 PyTorch (cu121)。")
            return packages + ['--index-url', 'https://download.pytorch.org/whl/cu121']
        elif major == 11:
            print("--- 匹配到 CUDA 11.x 系列，准备安装 PyTorch (cu118)。")
            return packages + ['--index-url', 'https://download.pytorch.org/whl/cu118']
        else:
            print(f"--- 警告: 不支持 CUDA 版本 {major}.x。将尝试安装 CPU 版本的 PyTorch。")
    
    print("--- 准备安装 CPU 版本的 PyTorch。")
    # 明确指定 CPU 版本以避免意外
    return packages + ['--index-url', 'https://download.pytorch.org/whl/cpu']

def patch_exp_manager():
    """在已安装的 nemo_toolkit 中查找并修补 exp_manager.py 文件 (仅限 Windows)"""
    print("\n> 尝试修补 nemo_toolkit...")
    try:
        # 自动查找 site-packages 路径，这比硬编码更可靠
        site_packages_path = next(p for p in sys.path if 'site-packages' in p)
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

if __name__ == "__main__":
    print("--- 开始执行项目安装脚本 ---")

    # 优先安装与硬件匹配的 PyTorch
    pytorch_command = ['pip', 'install'] + get_pytorch_install_command()
    run_command(pytorch_command)

    # 检查操作系统
    if sys.platform == "win32":
        # --- Windows 平台的安装流程 ---
        print("--- 检测到 Windows 系统，将执行特殊的安装与修补流程 ---")

        # 步骤 1: 安装有问题的依赖
        run_command(['pip', 'install', NEMO_TOOLKIT_PACKAGE])
        
        # 步骤 2: 执行修补
        patch_exp_manager()
        
        # 步骤 3: 安装主项目
        run_command(['pip', 'install', MAIN_PROJECT_PATH])
    
    else:
        # --- Linux/macOS 等其他平台的安装流程 ---
        print("--- 检测到非 Windows 系统 (如 Linux/macOS)，将执行标准安装流程 ---")
        run_command(['pip', 'install', MAIN_PROJECT_PATH])
    
    print("\n--- 安装流程执行完毕！ ---")
    print("--- 虚拟环境已准备就绪，可以开始使用。 ---")