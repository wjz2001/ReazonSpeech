import os
import dataclasses
import torch
from .interface import TranscribeConfig
from .decode import decode_hypothesis, PAD_SECONDS
from .audio import audio_to_file, pad_audio, norm_audio
from .fs import create_tempfile

def load_model(device=None):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """

    #  动态获取当前脚本文件所在的目录
    #    __file__ 是一个内置变量，代表当前脚本的文件名
    #    os.path.dirname(os.path.abspath(__file__)) 可以获得脚本所在的绝对目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义模型应该存放的本地路径和文件名
    #    使用 '..' 来代表上一级目录，从当前目录 (asr) 出发，
    #    上溯六层到达 D:\ReazonSpeech，然后再进入 models 目录
    model_dir = os.path.join(script_dir, '..', '..', '..', '..',     '..', '..', 'models')
    
    #    使用 os.path.normpath 来规范化路径，使其看起来更整洁 (例如     D:\ReazonSpeech\models)
    model_dir = os.path.normpath(model_dir)
    model_name = 'reazonspeech-nemo-v2.nemo'
    local_model_path = os.path.join(model_dir, model_name)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    # 4. 核心逻辑：检查本地模型是否存在
    if os.path.exists(local_model_path):
        # 如果文件存在，就从本地加载
        print(f"[提示] 在 '{local_model_path}' 找到本地模型。")
        print(f"[提示] 正在从本地加载模型...")
        return EncDecRNNTBPEModel.restore_from(restore_path=local_model_path,
                                              map_location=device)
    else:
        # 如果文件不存在，执行原始的下载逻辑
        print(f"[提示] 在 '{local_model_path}' 未找到本地模型。")
        print(f"[提示] 准备从 Hugging Face 下载模型...")
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
