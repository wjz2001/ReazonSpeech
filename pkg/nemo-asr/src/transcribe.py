import os
import dataclasses
import torch
from .interface import TranscribeConfig
from .decode import decode_hypothesis, PAD_SECONDS
from .audio import audio_to_file, pad_audio, norm_audio
from .fs import create_tempfile

def find_project_root(marker_file="asr.py"):
    """
    从当前脚本位置开始向上查找，直到找到包含指定标记文件的项目根目录。

    Args:
        marker_file (str): 用于标识项目根目录的文件名。

    Returns:
        str: 项目根目录的绝对路径。
        
    Raises:
        FileNotFoundError: 如果向上查找到文件系统顶层仍未找到标记文件。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir  # 找到了根目录
        
        parent_dir = os.path.dirname(current_dir)
        
        # 如果已经到达文件系统的顶层 (e.g., 'C:\\')，则停止
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"无法找到项目根目录，请确保在项目根目录下有一个 '{marker_file}' 文件"
            )
            
        current_dir = parent_dir

def load_model(device=None):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """

    local_model_path = None  # 默认本地路径为无效
    try:
        # 使用根目录下的 'asr.py' 文件作为锚点来定位项目根目录
        project_root = find_project_root(marker_file="asr.py")
    except FileNotFoundError as e:
        # 如果找不到根目录，不退出，只打印一个提示信息
        print(f"【提示】未能定位到项目根目录（{e}）")
        print("【提示】将直接尝试从 Hugging Face 下载模型")

    # 基于找到的根目录，安全地构建 models 文件夹的路径
    model_dir = os.path.join(project_root, 'models')
    model_name = 'reazonspeech-nemo-v2.nemo'
    local_model_path = os.path.join(model_dir, model_name)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print("【信息】检测到可用的 GPU，将自动选择 CUDA 加载模型")
        else:
            device = "cpu"
            print("【信息】未检测到可用的 GPU，将自动选择 CPU 加载模型")
    else:
        print(f"【信息】用户已指定使用 {device} 加载模型")        

    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    # 4. 核心逻辑：检查本地模型是否存在
    if os.path.exists(local_model_path):
        # 如果文件存在，就从本地加载
        print(f"【提示】在 '{model_dir}' 找到本地模型 '{model_name}'")
        print(f"【提示】正在从本地加载模型……")
        return EncDecRNNTBPEModel.restore_from(restore_path=local_model_path,
                                              map_location=device)
    else:
        # 如果文件不存在，执行原始的下载逻辑
        print(f"【提示】在 '{model_dir}' 未找到本地模型 '{model_name}'")
        print(f"【提示】准备从 Hugging Face 下载模型……")
        # 确保模型目录存在，以便下载的文件可以被 NeMo 缓存到默认位置
        # (注意: from_pretrained 有自己的缓存机制，通常在用户主目录下的 .cache)
        # 这里的打印信息主要是为了告知用户将要发生网络下载
        return EncDecRNNTBPEModel.from_pretrained('reazon-research/reazonspeech-nemo-v2',
                                              map_location=device)

def transcribe(model, audio, config=None):
    """Inference audio data using NeMo model

    Args:
        model (nemo.collections.asr.models.EncDecRNNTBPEModel): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = pad_audio(norm_audio(audio), PAD_SECONDS)

    # TODO Study NeMo's transcribe() function and make it
    # possible to pass waveforms on memory.
    with create_tempfile() as tmpf:
        audio_to_file(tmpf, audio)

        if os.name == 'nt':
            tmpf.close()

        hyp, _ = model.transcribe(
            [tmpf.name],
            batch_size=1,
            return_hypotheses=True,
            verbose=config.verbose
        )
        hyp = hyp[0]

    ret = decode_hypothesis(model, hyp)

    if config.raw_hypothesis:
        ret.hypothesis = hyp

    return ret
