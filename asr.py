import argparse
import copy
import ctypes
import math
import os
import threading
import tempfile
import sys
import subprocess
import torch
import io
import json
import time
import numpy as np
import onnxruntime
import wave

# 仅 Windows 使用 WinAPI
if sys.platform == "win32":
    from ctypes import wintypes, windll
else:
    import resource  

from collections import Counter, deque
from fractions import Fraction
from pathlib import Path
from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.audio import SAMPLERATE
from reazonspeech.nemo.asr.decode import find_end_of_segment, decode_hypothesis, PAD_SECONDS, SECONDS_PER_STEP, SUBWORDS_PER_SEGMENTS, TOKEN_PUNC
from reazonspeech.nemo.asr.interface import PreciseSubword, PreciseSegment
from reazonspeech.nemo.asr.writer import (
    SRTWriter,
    ASSWriter,
    TextWriter,
    TSVWriter,
    VTTWriter,
)

# --- 日志控制 ---
class Logger:
    def __init__(self):
        self.debug_mode = False

    def set_debug(self, enabled):
        self.debug_mode = enabled

    def info(self, *args, **kwargs):
        """普通信息：始终显示"""
        print(*args, **kwargs)

    def warn(self, *args, **kwargs):
        """警告信息：始终显示"""
        print("【警告】", *args, **kwargs)

    def debug(self, *args, **kwargs):
        """调试信息：仅在 --debug 开启时显示"""
        if self.debug_mode:
            print("【调试】", *args, **kwargs)

# 初始化全局日志实例
logger = Logger()

# === 内存 / 显存监控 ===
# 设定最大保留样本数。假设 interval=0.5s，保留 14400 个样本相当于保留最近 2 小时的数据
# 超过这个时间的数据会被丢弃，避免内存无限增长
MEM_SAMPLES = {
    "ram_gb": deque(maxlen=14400),
    "gpu_gb": deque(maxlen=14400),
}

# 额外增加一个字典，用于单独记录“全过程历史峰值”
# 因为 deque 会丢弃旧数据，如果峰值发生在很久以前，deque 里就找不到了
MEM_PEAKS = {
    "ram_gb": 0.0,
    "gpu_gb": 0.0,
}

if sys.platform == "win32":
    # HANDLE GetCurrentProcess(void);
    GetCurrentProcess = windll.kernel32.GetCurrentProcess
    GetCurrentProcess.restype = wintypes.HANDLE

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    GetProcessMemoryInfo = windll.psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.restype = wintypes.BOOL
    GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]

    def _get_ram_gb_sample():
        """
        Windows：通过 WinAPI 获取当前进程 Working Set，单位 GB
        """
        counters = PROCESS_MEMORY_COUNTERS_EX()
        cb = ctypes.sizeof(counters)
        counters.cb = cb

        handle = GetCurrentProcess()
        if not GetProcessMemoryInfo(handle, ctypes.byref(counters), cb):
            return None

        # WorkingSetSize 是字节
        return counters.WorkingSetSize / (1024 ** 3)

else:
    # ---- Linux / macOS / 其它 Unix ----
    def _get_ram_gb_sample():
        """
        返回当前进程大致的 RAM 使用量（GB）。
        - 在 Linux 上优先读 /proc/self/status 的 VmRSS（当前常驻内存）
        - 其他系统退化为 resource.getrusage 的 ru_maxrss（进程历史峰值）
        """
        # Linux: 读 /proc/self/status 里的 VmRSS
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # 例如: 'VmRSS:   123456 kB'
                            parts = line.split()
                            if len(parts) >= 2:
                                kb = int(parts[1])
                                return kb / (1024 ** 2)  # GB
            except FileNotFoundError:
                pass  # 某些精简系统可能没有 /proc/self/status

        # 退化方案：用 resource 的 ru_maxrss（历史最大常驻集）
        try:
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS 上 ru_maxrss 单位是 bytes；
            if sys.platform == "darwin":
                return maxrss / (1024 ** 3)
            else:
                # Linux 一般是 KB
                return maxrss / (1024 ** 2)
        except Exception:
            pass

        # 实在拿不到就返回 None
        return None

def _memory_monitor(stop_event, device, interval=0.5):
    """
    后台线程：周期性采样当前进程的 RAM 和 GPU 显存使用情况
    interval: 采样间隔（秒），表示每几秒采样一次，可按需要调大/调小
    """
    while not stop_event.is_set():
        # --- 获取 RAM ---
        # 缓存采样结果，避免多次调用系统 API
        ram_sample = _get_ram_gb_sample()

        if ram_sample is not None:
            MEM_SAMPLES["ram_gb"].append(ram_sample)
            # 更新历史最大值
            if ram_sample > MEM_PEAKS["ram_gb"]:
                MEM_PEAKS["ram_gb"] = ram_sample

        # --- 获取 GPU ---
        # 如果有 CUDA，就记录一次显存
        if torch.cuda.is_available():
            # torch.cuda.memory_allocated 返回的是字节，缓存结果
            gpu_sample = torch.cuda.memory_allocated(device) / (1024 ** 3)
            MEM_SAMPLES["gpu_gb"].append(gpu_sample)
            # 更新历史最大值
            if gpu_sample > MEM_PEAKS["gpu_gb"]:
                MEM_PEAKS["gpu_gb"] = gpu_sample

        time.sleep(interval)

def _calc_mode(samples):
    """
    对连续型数据求“众数”的简单方法：
    bin_size表示按 几GB 一档来求众数
    先按 bin_size (GB) 做分箱，再统计出现次数最多的箱。
    返回 (众数中心值, 该箱出现次数)
    """
    bin_size = 0.05
    if not samples:
        return None, 0
    most_bin, freq = Counter([int(x // bin_size) * bin_size for x in samples]).most_common(1)[0]
    # 返回区间中点更直观一点
    return most_bin + bin_size / 2, freq
# === 内存 / 显存监控结束 ===

OVERLAP_MS = 1000  # 此处定义重叠时长
PAD_MS = int(Fraction(str(PAD_SECONDS)) * 1000)

def arg_parser():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型识别语音，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题"
    )

    # 音频/视频文件路径
    parser.add_argument(
        "input_file",
        nargs="?",                 # 0 或 1 个参数
        help="需要识别语音的音频/视频文件路径",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，处理结束后不删除临时文件，并自动打开临时分块目录",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        choices=range(1, 1281),
        default=None,
        metavar="[1-1280]",
        help="设置语音识别批量推理的大小，数字越大批量推理的速度越快，超过16可能会增加延迟，不填则自动根据显存估算（只使用 CPU 则默认为 1）",
        )

    parser.add_argument(
        "--beam",
        type=int,
        choices=range(4, 257), # range(4, 257) 包含 4 到 256，不包含 257
        default=4,
        metavar="[4-256]", # 设置这个参数，帮助信息里就会显示为 [4-256]，而不是列出几十个数字
        help="设置集束搜索（Beam Search）宽度，范围为 4 到 256 之间的整数，默认值是 4 ，更大的值可能更准确但更慢",
        )

    parser.add_argument(
        "--no-remove-punc",
        action="store_true",
        help="禁止自动剔除句末标点，保留原始识别结果",
    )

    # --- 音频滤镜参数 ---
    parser.add_argument(
        "--audio-filter",
        nargs="?",                 # 0 或 1 个参数
        default=None,
        const="highpass=f=60,lowpass=f=8000,afftdn=nf=-25",  # 只写 --audio-filter 时取这个值
        help=(
            "给 ffmpeg 解码阶段增加音频滤镜链（传给 -af）\n"
            "  不写              -> 不开启滤镜；\n"
            "  --audio-filter    -> 使用内置默认滤镜；\n"
            "  --audio-filter \"highpass=f=60,lowpass=f=8000\" -> 使用自定义滤镜链"
        ),
    )

    parser.add_argument(
        "--limiter-filter",
        type=str,
        help=(
            "必须显式传入滤镜链，且程序会自动在滤镜链末尾附加 alimiter 限制器，\n"
            "确保音频电平不超过 -0.2dB (0.98)，防止硬削波失真"
        ),
    )

    # --- VAD参数 ---
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="禁用智能分块功能，一次性处理整个音频文件",
    )

    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.4,
        metavar="(0.05-1]",
        help="【VAD】判断为语音的置信度阈值",
    )
    # VAD 结束阈值（双阈值滞回）
    parser.add_argument(
        "--vad_end_threshold",
        type=float,
        default=None,
        metavar="[0.05-1]",
        help="【VAD】判断为语音结束后静音的置信度阈值，默认值是vad_threshold的值减去0.15",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=int,
        default=100,
        help="【VAD】移除短于此时长（毫秒）的语音块",
    )
     # 静音最小时长，用于智能合并/分段
    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        default=200,
        help="【VAD】短于此时长（毫秒）的语音块不被视为间隔",
    )
    parser.add_argument(
        "--keep_silence",
        type=int,
        default=300,
        help="【VAD】在语音块前后扩展时长（毫秒）",
    )

    # --- ZCR 参数 ---
    parser.add_argument(
        "--zcr",
        action="store_true",
        help="开启过零率检测，防止切断清辅音",
    )
    parser.add_argument(
        "--zcr_threshold",
        type=float,
        default=0.15,
        help="【ZCR】手动设置 ZCR 阈值",
    )
    parser.add_argument(
        "--auto_zcr",
        action="store_true",
        help="【ZCR】开启自适应 ZCR 阈值计算，zcr_threshold作为兜底",
    )

    # --- 精修参数（必须先使用VAD） ---
    parser.add_argument(
        "--refine-tail",
        action="store_true",
        help="使用段尾精修",
    )

    parser.add_argument(
        "--tail_percentile",
        type=float,
        default=20,
        metavar="[0-100]",
        help="【精修】自适应阈值，值越大，越容易将高概率语音区域判为静音",
    )
    parser.add_argument(
        "--tail_offset",
        type=float,
        default=0.05,
        help="【精修】在自适应阈值的基础上增加的固定偏移量，值越大越容易将高概率区域语音区域判为静音",
    )
    parser.add_argument(
        "--tail_energy_percentile",
        type=float,
        default=30,
        metavar="[0-100]",
        help="【精修】自适应能量阈值，通常取 20~40，低于此值则判定为静音",
    )
    parser.add_argument(
        "--tail_energy_offset",
        type=float,
        default=0.0,
        help="【精修】在自适应能量阈值基础上增加的固定偏移量，一般保持为 0 即可，值越大判定标准越宽松",
    )
    parser.add_argument(
        "--tail_lookahead_ms",
        type=int,
        default=80,
        help="【精修】滞回检查向前看的时长（毫秒），用于确认静音的稳定性，不会马上又回到语音",
    )
    parser.add_argument(
        "--tail_safety_margin_ms",
        type=int,
        default=30,
        help="【精修】在找到的切点后增加的安全边距（毫秒），避免切得太生硬",
    )
    parser.add_argument(
        "--tail_min_keep_ms",
        type=int,
        default=30,
        help="【精修】强制保留在段尾的最小时长（毫秒），保证听感自然",
    )
    parser.add_argument(
        "--tail_zcr_high_ratio",
        type=float,
        default=0.3, 
        metavar="[0.1-0.5]",
        help="【精修】ZCR保护触发比例：在疑似静音窗口内，高于 ZCR 阈值的帧超过此比例时，才会判定为清音",
    )

    # 输出格式参数
    parser.add_argument(
        "-text",
        action="store_true",
        help="仅输出完整的识别文本并保存为 .txt 文件",
    )
    parser.add_argument(
        "-segment",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并保存为 .segments.txt 文件",
    )
    parser.add_argument(
        "-segment2srt",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为 .srt 字幕文件",
    )
    parser.add_argument(
        "-segment2vtt",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为 .vtt 字幕文件",
    )
    parser.add_argument(
        "-segment2tsv",
        action="store_true",
        help="输出带时间戳的文本片段（Segment）并转换为由制表符分隔的 .tsv 文件",
    )
    parser.add_argument(
        "-subword",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并保存为 .subwords.txt 文件",
    )
    parser.add_argument(
        "-subword2srt",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.srt 字幕文件",
    )
    parser.add_argument(
        "-subword2json",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.json 文件",
    )
    parser.add_argument(
        "-kass",
        action="store_true",
        help="生成逐字计时的卡拉OK式 .ass 字幕文件",
    )
    return parser

def assign_vad_ownership_keep_windows(vad_segments, keep_silence_ms, total_duration_ms):
    """
    输入：vad_segments = list[dict]，每个 dict 包含:
      {"start_ms": ..., "end_ms": ..., "keep_start_ms": ..., "keep_end_ms": ...}

    输出：直接在每个 dict 上写入：
      own_keep_start_ms / own_keep_end_ms
    """
    n = len(vad_segments)
    own_keep_start_ms = [None] * n
    own_keep_end_ms = [None] * n

    # 第一段：左侧最多扩 keep_silence_ms
    own_keep_start_ms[0] = max(0, vad_segments[0]["start_ms"] - keep_silence_ms)

    for i in range(n - 1):
        e_i = vad_segments[i]["end_ms"]
        s_j = vad_segments[i + 1]["start_ms"]
        gap = s_j - e_i

        if gap < keep_silence_ms:
            # gap 全分配给后一段：边界取 e_i
            own_keep_end_ms[i] = e_i
            own_keep_start_ms[i + 1] = e_i
        else:
            # 首选中点
            mid = (e_i + s_j) // 2
            left_max = e_i + keep_silence_ms
            right_min = s_j - keep_silence_ms

            if right_min <= mid <= left_max:
                # gap <= 2*keep_silence_ms：中点可行
                own_keep_end_ms[i] = mid
                own_keep_start_ms[i + 1] = mid
            else:
                # gap > 2*keep_silence_ms：各扩 keep_silence_ms，中间留空
                own_keep_end_ms[i] = left_max
                own_keep_start_ms[i + 1] = right_min

    # 最后一段：右侧最多扩 keep_silence_ms
    own_keep_end_ms[-1] = min(total_duration_ms, vad_segments[-1]["end_ms"] + keep_silence_ms)

    # 补齐字段
    for i in range(n):
        vad_segments[i]["own_keep_start_ms"] = own_keep_start_ms[i]
        vad_segments[i]["own_keep_end_ms"] = own_keep_end_ms[i]

    return vad_segments

_BATCH_SIZE_CACHE = None
_BATCH_SIZE_LOCK = threading.Lock()
def auto_tune_batch_size(model, max_duration_s):
    """
    根据显存自动估算 Batch Size
    """
    global _BATCH_SIZE_CACHE

     # 检查缓存，无锁读取，如果已存在直接返回
    if _BATCH_SIZE_CACHE is not None:
        return _BATCH_SIZE_CACHE

    # 加锁计算，防止并发请求导致重复计算
    with _BATCH_SIZE_LOCK:
        # 双重检查，防止在等待锁的过程中已经被其他线程计算完了
        if _BATCH_SIZE_CACHE is not None:
            return _BATCH_SIZE_CACHE

        if not torch.cuda.is_available():
            _BATCH_SIZE_CACHE = 1
            return 1
    
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    
        try:
            with torch.inference_mode():
                # 构造 dummy 数据 (30s 空白音频)
                _ = transcribe_audio(model, [torch.zeros(int(max_duration_s * SAMPLERATE), dtype=torch.float32)])
    
        except RuntimeError:
            _BATCH_SIZE_CACHE = 1
            return 1 # OOM 或其他错误，回退到 1
    
        per_sample = max(torch.cuda.max_memory_allocated(device) - baseline, 1)
        result = max(1, int(torch.cuda.mem_get_info(device)[0] * 0.7 // per_sample // 2))

        # 计算理论最大值，至少为 1，0.7和2是安全限制
        _BATCH_SIZE_CACHE = result
        return result

def calibrate_zcr_threshold(speech_probs, zcr_array):
    """
    自适应 ZCR 阈值校准函数 (Method A: Percentile).
    """
    logger.info("【ZCR】正在校准自适应 ZCR 阈值……")

    # 分群
    # 资料建议：VAD > 0.8 为浊音(voiced)，VAD < 0.2 为非语音/清音背景(unvoiced)
    # 注意：这里的 unvoiced 其实包含静音和清音，但这不影响找下界
    voiced_mask = speech_probs > 0.8
    unvoiced_mask = speech_probs < 0.2
    if voiced_mask.sum() < 10 or unvoiced_mask.sum() < 10:
        logger.debug("【ZCR】浊音/静音样本区分度不足，回退默认值")
        return None

    # 统计分位数
    # 资料建议：τ_v = P90 (浊音上限), τ_u = P10 (非语音下限)
    tau_v = np.percentile(zcr_array[voiced_mask], 90)
    tau_u = np.percentile(zcr_array[unvoiced_mask], 10) # 非语音区域的 ZCR 下限（P10）
    
    # 计算自适应阈值
    # 安全检查：如果计算出的阈值太极端，说明数据分布有问题
    if not (0.05 <= (adaptive_th := 0.5 * (tau_v + tau_u)) <= 0.5):
        logger.debug("【ZCR】计算阈值超出安全范围（0.05~0.5），回退默认值")
        return None
    
    logger.debug(f"【ZCR】浊音 P90 = {tau_v:.6f}，非语音 P10 = {tau_u:.6f}，计算阈值 = {adaptive_th:.6f}")
    return adaptive_th

def create_precise_segments_from_subwords(raw_subwords, vad_chunk_end_times_ms, total_duration_ms, no_remove_punc):

    all_subwords = []
    durations = [] 

    vad_chunk_end_times_s = [t / 1000 for t in vad_chunk_end_times_ms]
    
    vad_cursor = 0
    for i, sub in enumerate(raw_subwords):
        # 计算自然结束时间 (Next Start)
        next_start = raw_subwords[i + 1].seconds if i < len(raw_subwords) - 1 else sub.seconds + SECONDS_PER_STEP * 2 # 最后一个子词兜底
        
        # 计算 VAD 限制 (VAD Limit)
        # 移动游标找到当前子词所属的 VAD 块
        # 利用短路逻辑：如果 vad_chunk_end_times_ms 为空，循环不会执行
        while vad_cursor < len(vad_chunk_end_times_s) and sub.seconds >= vad_chunk_end_times_s[vad_cursor]:
            vad_cursor += 1
        # 最后一段没有vad边界，或者 no_chunk 模式下 vad_chunk_end_times_ms 列表为空，用整个音频的总时长作为边界
        current_vad_limit = vad_chunk_end_times_s[vad_cursor] if vad_cursor < len(vad_chunk_end_times_s) else total_duration_ms / 1000

        # 收集 duration 用于后续停顿阈值计算
        durations.append(next_start - sub.seconds)

        all_subwords.append(PreciseSubword(
            seconds=sub.seconds,
            # 保证不超过 VAD 边界和下一个词的开始
            end_seconds = min(next_start, current_vad_limit),
            token_id=sub.token_id,
            token=sub.token,
            vad_limit=current_vad_limit
        ))

    all_segments = []
    start = 0
    while start < len(all_subwords):
        current_limit = all_subwords[start].vad_limit
        # 现在我们可以直接在 all_subwords 中向后扫描，找到 vad_limit 变化的索引。
        next_group_start_idx = start
        while next_group_start_idx < len(all_subwords) and all_subwords[next_group_start_idx].vad_limit == current_limit:
            next_group_start_idx += 1
            
        # 确定当前 VAD 块内的搜索边界
        end_idx = min(next_group_start_idx - 1, find_end_of_segment(all_subwords, start))
        # --- 动态计算停顿阈值 ---
        pause_split_indices = []
        # 基于语速/停顿的边界补充，只有当有足够多的数据点（例如超过20个子词间隔）时，才计算并启用阈值
        if (end_idx - start + 1) > SUBWORDS_PER_SEGMENTS * 1.5:
            pause_threshold = np.percentile(durations[start:end_idx], 90)
            # 使用列表推导式直接筛选
            pause_split_indices = [
                start + k for k, d in enumerate(durations[start:end_idx]) 
                if d > pause_threshold
            ]

        # 调试日志逻辑
        if logger.debug_mode and pause_split_indices:
            preview_parts = []
            pause_split_set = set(pause_split_indices)
            for i in range(start, end_idx + 1):
                preview_parts.append(all_subwords[i].token)
                if i in pause_split_set:
                    preview_parts.append(" || ")
            logger.debug(f"在 {SRTWriter._format_time(all_subwords[start].seconds)} 开始的长片段：“{''.join(preview_parts)}”内找到显著停顿点")

        # 将片段的起始点和最终的结束点加入，形成完整的切分区间
        # start - 1 表示前一个片段的结束索引，作为新片段的基准，为了让循环 for i in range(len(all_split_points) - 1): 能够从 start 索引开始处理第一个子片段
        all_split_points = [start - 1] + pause_split_indices + [end_idx]

        for i in range(len(all_split_points) - 1):
            # 提取当前片段的子词
            current_subwords = all_subwords[all_split_points[i] + 1:all_split_points[i + 1] + 1]
    
            # 预先获取时间边界和文本，用于判断和日志
            curr_start = current_subwords[0].seconds
            curr_end = current_subwords[-1].end_seconds
            curr_limit = current_subwords[0].vad_limit
    
            # 判断是否为纯标点/空格片段
            if all((s.token in TOKEN_PUNC or not s.token.strip()) for s in current_subwords):
                # === 分支 A：纯标点片段 ===
                curr_text_raw = "".join(s.token for s in current_subwords) # 原始文本
                if all_segments:
                    prev_seg = all_segments[-1]
                    # 注意：此时 prev_seg 还没更新，curr_text_raw 是即将被合并（或丢弃）的标点
                    logger.debug((
                        f"{SRTWriter._format_time(prev_seg.start_seconds)} --> {SRTWriter._format_time(prev_seg.end_seconds)}：{prev_seg.text}"
                        f" 和 "
                        f"{SRTWriter._format_time(curr_start)} --> {SRTWriter._format_time(curr_end)}：{curr_text_raw}"
                        f" 已合并"
                    ))
    
                    # 无论是否保留标点，时间与vad边界都要吸收
                    prev_seg.end_seconds = curr_end
                    prev_seg.vad_limit = curr_limit 
                    
                    # 将上一段最后一个子词的结束时间也拉长
                    if no_remove_punc:
                        # 如果保留标点，合并文本和实体
                        prev_seg.subwords[-1].end_seconds = curr_start
                        prev_seg.text += curr_text_raw
                        prev_seg.subwords.extend(current_subwords)
                    else:
                        prev_seg.subwords[-1].end_seconds = curr_end
    
            else:
                # === 分支 B：含正文的片段 ===
                # 如果需要剔除句末标点，且最后一个子词是标点
                if not no_remove_punc and current_subwords[-1].token in TOKEN_PUNC:
                    removed_punc = current_subwords.pop()

                    logger.debug(f"已剔除 {SRTWriter._format_time(removed_punc.seconds)} --> {SRTWriter._format_time(removed_punc.end_seconds)}：{removed_punc.token}")

                    # 因为不是全标点句，pop 后列表一定不为空
                    # 填补时间空缺
                    current_subwords[-1].end_seconds = removed_punc.end_seconds
                
                text = "".join(s.token for s in current_subwords)

                # 创建新片段
                all_segments.append(PreciseSegment(
                    start_seconds=curr_start,
                    end_seconds=curr_end,
                    text=text,
                    vad_limit=curr_limit,
                    subwords=current_subwords
                ))

                if len(text) > SUBWORDS_PER_SEGMENTS * 2:
                    logger.warn(
                        f"{SRTWriter._format_time(curr_start)} --> {SRTWriter._format_time(curr_end)} 段字数超过 {SUBWORDS_PER_SEGMENTS * 2}，制作字幕时有可能溢出屏幕"
                    )

        # 更新外层循环游标
        start = end_idx + 1

    return all_segments, all_subwords

def format_duration(seconds):
    """将秒数格式化为 'X时Y分Z.ZZ秒' 的形式"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)} 小时 ")
    if minutes > 0:
        parts.append(f"{int(minutes)} 分 ")

    # 总是显示秒，并保留两位小数
    parts.append(f"{secs:.2f} 秒")

    return "".join(parts)

# 全局变量：用于缓存加载好的模型
_ASR_MODEL = None
_ASR_MODEL_LOAD_COST = 0.0
_ASR_MODEL_LOCK = threading.Lock()
def get_asr_model():
    """
    获取 Reazonspeech 模型的单例
    如果模型未加载，则加载并缓存；如果已加载，则直接返回
    """
    global _ASR_MODEL, _ASR_MODEL_LOAD_COST
    with _ASR_MODEL_LOCK: # 加锁
        if _ASR_MODEL is None: # 检查
            logger.info("正在加载 Reazonspeech ……")
            asr_model_load_start = time.perf_counter()  # <--- 计时开始
            # 调用原有的 load_model 函数
            _ASR_MODEL = load_model()
            asr_model_load_end = time.perf_counter()  # <--- 计时结束
            logger.info(f"Reazonspeech 加载完成")
            _ASR_MODEL_LOAD_COST = asr_model_load_end - asr_model_load_start
    return _ASR_MODEL

def get_speech_timestamps_onnx(
    waveform,
    onnx_session,
    threshold,
    neg_threshold, #语音结束阈值（低阈值）
    min_speech_duration_ms, #最小语音段时长
    min_silence_duration_ms #多长静音视为真正间隔
):
    """
    返回：
      speeches_ms: List[[start_ms, end_ms], ...]  # int ms
      speech_probs: np.ndarray                    # per-frame prob
      frame_duration_ms: float                    # ms per frame
      frame_times_ms: np.ndarray                  # 每帧对应的绝对时间轴（ms），与 speech_probs 对齐
    使用 segmentation-3.0 ONNX + ≤10 秒滑窗 + 双阈值后处理，
    将一段 waveform 转换为若干语音时间段
    """
    # 通过 ≤10 秒滑窗，拼出整段 speech_probs 和 frame_times_ms
    all_probs = []
    all_times = []
    start_sample = 0
    total = waveform.shape[1]
    # 预先计算重叠样本数
    overlap_samples = int(round(OVERLAP_MS * SAMPLERATE / 1000))
    while start_sample < total:
        end_sample = min(start_sample + int(10 * SAMPLERATE), total)
        # 如果尾巴太短于约1s，直接把本窗扩到结尾（吞尾巴）
        if (total - end_sample) < SAMPLERATE:
            end_sample = total
        # [1, T_chunk]
        chunk = waveform[:, start_sample:end_sample]
        #  ONNX 输出 (batch, T, classes)，取 [0] 变成 (T, classes)，再取[0]
        logits = torch.from_numpy(
            onnx_session.run(
                None,
                {"input_values": chunk.contiguous().unsqueeze(0).numpy()}
            )[0]
        )[0]

        # 聚合语音组能量 (Index 1-6)
        # torch.logsumexp 默认 keepdim=False，输入 (T,6) -> 输出 (T)
        # 获取静音能量 (Index 0) -> (T)
        # 竞争: 语音 - 静音，再Sigmoid
        # 转回 Numpy 以便进入 Python 循环
        # logits[:, 0] > 静音维度，logits[:,1:] > 语音维度
        probs_full = torch.sigmoid(torch.logsumexp(logits[:,1:], dim=1) - logits[:, 0]).numpy()
        # 当前窗口的实际样本数
        window_samples = end_sample - start_sample

        is_first = start_sample == 0
        is_last = end_sample == total
        # 计算“帧/样本”的缩放比例，需要裁剪的帧数
        crop_frames = int(round(overlap_samples * len(probs_full) / window_samples))
        
        # 确定起止索引，如果是首窗，起点为0；否则裁剪掉头部重叠区
        start_idx = 0 if is_first else crop_frames
        
        # 如果是尾窗，终点为总长度；否则裁剪掉尾部重叠区
        end_idx = len(probs_full) if is_last else (len(probs_full) - crop_frames)

        # 拼 frame_times_ms：用该窗“真实覆盖时长 / 该窗输出帧数”做线性映射，再按同样裁剪
        hop_ms = (window_samples * 1000.0 / SAMPLERATE) / len(probs_full)
        times_full = start_sample * 1000.0 / SAMPLERATE + np.arange(len(probs_full), dtype=np.float32) * hop_ms

        all_probs.append(probs_full[start_idx:end_idx])
        all_times.append(times_full[start_idx:end_idx])

        if end_sample >= total:
            break
        start_sample += int(round((10000 - OVERLAP_MS * 2) * SAMPLERATE / 1000))

    speech_probs = all_probs[0] if len(all_probs) == 1 else np.concatenate(all_probs, axis=0)
    frame_times_ms = all_times[0] if len(all_times) == 1 else np.concatenate(all_times, axis=0)

    num_frames = speech_probs.shape[0]

    # 统一按「整段时长 / 帧数」来近似每帧对应的时间长度
    # segmentation-3.0 实际上是 10ms 一帧，这个计算方式在任意长度下都一致
    frame_duration_ms = total * 1000 / SAMPLERATE / num_frames

    # 双阈值 + 填平短静音 + 丢弃短语音 + 直接生成秒级区间
    # 最小语音段帧数（小于这个长度的语音段会被丢弃）
    min_speech_frames = math.ceil(min_speech_duration_ms / frame_duration_ms)
    # 最小静音段帧数（短于这个长度的静音视为“短静音”，会被填平）
    min_silence_frames = math.ceil(min_silence_duration_ms / frame_duration_ms)

    speeches = []           # 直接存毫秒级结果 [[start_ms, end_ms], ...]
    in_speech = False       # 当前是否处于语音段中
    seg_start_f = 0         # 当前语音段起点（帧）
    silence_start_f = None  # 语音段内部，疑似静音开始帧

    for i, p in enumerate(speech_probs):
        if not in_speech:
            # 还在「非语音」状态，只有超过高阈值 threshold 才进入语音段
            if p >= threshold:
                in_speech = True
                seg_start_f = i
                silence_start_f = None
        else:
            # 已在语音段中
            if p < neg_threshold:
                # 掉到结束阈值以下，可能开始进入静音
                if silence_start_f is None:
                    silence_start_f = i
                # 连续低于 neg_threshold 的帧数达到 min_silence_frames 才真正结束一段
                if (i - silence_start_f) >= min_silence_frames:
                    seg_end_f = silence_start_f
                    # 大于等于 min_speech_frames 才是真正的语音块
                    if (seg_end_f - seg_start_f) >= min_speech_frames:
                        start_ms = int(seg_start_f * frame_duration_ms)
                        end_ms = math.ceil(seg_end_f * frame_duration_ms)
                        speeches.append([start_ms, end_ms])
                    # 重置状态，等待下一段
                    in_speech = False
                    seg_start_f = 0
                    silence_start_f = None
            else:
                # 概率又回到了 neg_threshold 以上，静音计数作废
                silence_start_f = None

    # 处理结尾残留：文件结束时仍在语音段中
    if in_speech:
        seg_end_f = num_frames
        if (seg_end_f - seg_start_f) >= min_speech_frames:
            start_ms = int(seg_start_f * frame_duration_ms)
            end_ms = math.ceil(seg_end_f * frame_duration_ms)
            speeches.append([start_ms, end_ms])

    # 完整的语音概率数组 speech_probs 、每一帧的持续时间 frame_duration_ms、每帧对应的绝对时间轴 frame_times_ms
    return speeches, speech_probs, frame_duration_ms, frame_times_ms

def global_smart_segmenter(
    start_ms, end_ms, speech_probs,
    energy_array, frame_times_ms, max_duration_ms,
    vad_threshold, zcr_threshold, zcr_array):
    """
    入口函数：分析一个长 VAD 段，返回切割好的子段列表。

    - 如果大于 max_duration_s：
        * 在低 VAD 概率的局部极小值处生成候选点，并计算 cost
        * 从左到右贪心切分（必要时回退到硬切）
        * 以切点 cut 为准：
            - 上一段推理范围 end = cut + OVERLAP_MS
            - 下一段推理范围 start = cut - OVERLAP_MS
          （相邻段推理范围重叠 2*OVERLAP_MS，但提交窗口在 cut 处分割，保证不重复提交）

    返回：List[dict]
      每个元素：
        {
          "start_ms": 推理段起点（毫秒，包含重叠区），
          "end_ms": 推理段终点（毫秒，包含重叠区），
          "keep_start_ms": 提交窗口起点（毫秒，中段起点），
          "keep_end_ms": 提交窗口终点（毫秒，中段终点），
        }
    """
    def _frame_cost(frame_idx):
        """计算单帧作为切分点的代价（代价越低越适合切分）"""
        # 基础 VAD 概率成本
        if speech_probs[frame_idx] > vad_threshold:
            return 100.0  # 惩罚高概率点，强硬约束：确信是语音的地方绝不切

        # --- 动态抗噪过零率 (Robust ZCR) ---
        # 逻辑：(过零) AND (穿越点的幅度足够大，不仅是微小抖动)
        # 标准做法：check max(abs(x[n]), abs(x[n+1])) > th
        zcr_cost = 0.0
        if zcr_array is not None and zcr_array[frame_idx] > zcr_threshold:
            zcr_cost = zcr_array[frame_idx] * 50

        # 局部平滑度 (斜率)
        slope_cost = abs(speech_probs[frame_idx + 1] - speech_probs[frame_idx - 1]) * 2

        return speech_probs[frame_idx] + min(energy_array[frame_idx], 0.5) * 10 + slope_cost + zcr_cost

    # 时长不超长直接返回
    if end_ms - start_ms <= max_duration_ms:
        return [{
            "start_ms": start_ms,
            "end_ms": end_ms,
            "keep_start_ms": start_ms,
            "keep_end_ms": end_ms,
        }]

    # 生成候选点（筛选 VAD 概率 < max(0.15, vad_threshold - 0.15) 的局部低点）
    candidates = []
    # 边界保护：避免切在段首/段尾靠得太近
    # 用 frame_times_ms 反查 frame 索引
    search_start = int(np.searchsorted(frame_times_ms, start_ms + OVERLAP_MS, side="left"))
    search_end = int(np.searchsorted(frame_times_ms, end_ms - OVERLAP_MS, side="right"))

    # 提取搜索区间的概率
    # search_end > search_start + 2 是为了确保区间内至少有 3 帧数据
    # 寻找局部极小值需要比较：当前帧(curr) <= 前一帧(prev) 且 当前帧(curr) <= 后一帧(next)
    # 这对应了下方的切片逻辑：[1:-1] (curr), [:-2] (prev), [2:] (next)
    # 如果总帧数少于 3 (即 end - start <= 2)，切片后的维度将无法对齐或为空，因此跳过
    if search_end > search_start + 2:
        # 利用 numpy 寻找局部极小值且概率 < max(0.15, vad_threshold - 0.15) 的点
        # 逻辑：当前点 <= 前一点 AND 当前点 <= 后一点 AND 当前点 < max(0.15, vad_threshold - 0.15)
        p_curr = speech_probs[search_start + 1:search_end - 1]  # 对应原数组的索引 i
        p_prev = speech_probs[search_start:search_end - 2]      # 对应 i-1
        p_next = speech_probs[search_start + 2:search_end]      # 对应 i+1
        # 生成布尔掩码
        mask = (p_curr < max(0.15, vad_threshold - 0.15)) & (p_curr <= p_prev) & (p_curr <= p_next)
        # 获取满足条件的相对索引
        # 还原绝对索引：+1 是因为 p_curr 从切片的第1个元素开始，+search_start 是切片偏移
        # 遍历筛选出的索引计算成本
        for idx in (np.where(mask)[0] + 1 + search_start):
            candidates.append({
                "frame": idx,
                "time": int(round(frame_times_ms[idx])),
                "cost": _frame_cost(idx)
            })

    segments = []
    curr_start = start_ms
    prev_cut = None
    # 初始化索引指针，指向 candidates 列表的开头
    cand_idx = 0
    while True:
        # 剩余长度已经不超过上限，直接收尾
        if end_ms - curr_start <= max_duration_ms:
            segments.append({
                "start_ms": curr_start,
                "end_ms": end_ms,
                "keep_start_ms": start_ms if prev_cut is None else prev_cut,
                "keep_end_ms": end_ms,
            })
            break

        # 向前移动指针：跳过窗口左侧的旧候选点（切点必须 > curr_start + OVERLAP_MS，才能保证 next_start > curr_start）
        while cand_idx < len(candidates) and candidates[cand_idx]["time"] <= curr_start + OVERLAP_MS:
            cand_idx += 1

        # 收集当前窗口内的候选点
        cands = []
        # 使用一个临时游标 temp_idx 向后扫描，直到超出窗口右侧
        # 不修改 cand_idx，因为下一个窗口可能还会用到当前的起始点，如果有重叠或回退逻辑
        temp_idx = cand_idx
        while temp_idx < len(candidates):
            c = candidates[temp_idx]
            if c["time"] > curr_start + max_duration_ms - OVERLAP_MS:
                # 因为列表是有序的，一旦超过右边界，后面的都不用看了
                break
            cands.append(c)
            temp_idx += 1

        if cands:
            # 直接选 cost 最小的
            cut_t = min(cands, key=lambda c: c["cost"])["time"]
        else:
            # 当前窗口里没有任何合适候选，只能硬切
            cut_t = curr_start + max_duration_ms - OVERLAP_MS

        segments.append({
            "start_ms": curr_start,
            "end_ms": cut_t + OVERLAP_MS,
            "keep_start_ms": start_ms if prev_cut is None else prev_cut,
            "keep_end_ms": cut_t,
        })

        prev_cut = cut_t
        curr_start = cut_t - OVERLAP_MS

    return segments

def load_audio_ffmpeg(file, audio_filter, limiter_filter):
    """
    使用 ffmpeg 将任何输入格式转为:
      - 单声道
      - 采样率 SAMPLERATE
      - 输出端量化为 16-bit PCM (s16le)，再在 Python 中转回 float32 [-1, 1)
      根据 audio_filter 和 limiter_filter 决定是否附加 -af 滤镜链，limiter_filter 会追加 alimiter
    然后转为 PyTorch Tensor 和总时长（毫秒）
    """
    def _append_tail(graph, tail):
        """
        把 tail 接到 -af simple filtergraph 的最终输出后面：
          - 若 graph 最终输出是隐式的：  graph + "," + tail
          - 若 graph 最终输出是显式标签 [out]： graph + ";[out]" + tail
        目标：覆盖 -af 支持的“复杂但仍 1-in-1-out”的语法（含 ; 和 [label]）。
        """
        if not (s := graph.strip()):
            return tail
    
        produced = set()  # 作为输出标签出现过
        consumed = set()  # 作为输入标签出现过
        i = 0
        n = len(s)
    
        def skip_ws():
            # 跳过空格
            nonlocal i
            while i < n and s[i].isspace():
                i += 1
    
        while True:
            skip_ws()
            if i >= n:
                break
    
            # 解析输入标签 [..][..]，提取标签名，加入 consumed 集合，意味着这个标签代表的流已经进入了某个滤镜，不再是最终输出
            while True:
                skip_ws()
                if i >= n or s[i] != "[":
                    break
                i += 1
                label = []
                esc = False
                while i < n:
                    ch = s[i]
                    if esc:
                        label.append(ch)
                        esc = False
                        i += 1
                        continue
                    if ch == "\\":
                        esc = True
                        i += 1
                        continue
                    if ch == "]":
                        break
                    label.append(ch)
                    i += 1
                if i >= n or s[i] != "]":
                    raise RuntimeError(f"audio_filter 解析失败：方括号未闭合：{s!r}")
                consumed.add("".join(label))
                i += 1
    
            skip_ws()
            if i >= n:
                break
    
            # 读滤镜名，遇到 = (开始参数) , (下一级) ; (图分割) [ (下一个标签) 时停止
            start = i
            while i < n and s[i] not in "=,;[":
                i += 1
            if not s[start:i].strip():
                raise RuntimeError(f"audio_filter 解析失败：缺少滤镜名（位置 {start}）：{s!r}")
    
            # 跳过参数（若有 '='，一直到未转义/未在引号中的 , ; [）
            if i < n and s[i] == "=":
                i += 1
                in_quote = None
                esc = False
                while i < n:
                    ch = s[i]
                    if esc:
                        esc = False
                        i += 1
                        continue
                    if ch == "\\":
                        esc = True
                        i += 1
                        continue
                    if in_quote:
                        if ch == in_quote:
                            in_quote = None
                        i += 1
                        continue
                    else:
                        if ch in ("'", '"'):
                            in_quote = ch
                            i += 1
                            continue
                        if ch in ",;[":
                            break
                        i += 1
    
            # 读输出标签 [..][..]，将识别到的标签加入 produced 集合
            while True:
                skip_ws()
                if i >= n or s[i] != "[":
                    break
                i += 1
                label = []
                esc = False
                while i < n:
                    ch = s[i]
                    if esc:
                        label.append(ch)
                        esc = False
                        i += 1
                        continue
                    if ch == "\\":
                        esc = True
                        i += 1
                        continue
                    if ch == "]":
                        break
                    label.append(ch)
                    i += 1
                if i >= n or s[i] != "]":
                    raise RuntimeError(f"audio_filter 解析失败：方括号未闭合：{s!r}")
                produced.add("".join(label))
                i += 1
    
            # 分隔符 , 或 ;（或结束），如果遇到 ,，说明是线性连接，遇到 ;，说明是图连接，继续解析下一个
            skip_ws()
            if i < n and s[i] in ",;":
                i += 1
                continue
            break
    
        # 如果在 produced 里但不在 consumed 里，说明这个流是当前滤镜链的最终输出
        dangling = produced - consumed
        s_clean = s.rstrip(" ,;")
    
        # 没有悬空标签，直接加一个逗号 , 然后接上 tail
        if len(dangling) == 0:
            return s_clean + "," + tail
    
        # 有一个悬空标签，显式地引用这个标签作为 tail 的输入，使用分号 ; 开始新的一段，写上标签，再接 tail
        if len(dangling) == 1:
            return s_clean + f";[{next(iter(dangling))}]" + tail
    
        # 多个悬空标签，报错
        raise RuntimeError(
            "audio_filter 产生了多个最终输出标签，这通常不属于 -af 的值\n"
            f"dangling outputs: {[repr(x) for x in sorted(dangling)]}\n"
            f"filtergraph: {s!r}"
        )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",   # 不要 banner
        "-i", str(file),
        "-vn",                   # 不要视频
        "-sn",                   # 不要字幕
        "-dn",                   # 不要 data stream
        "-map", "0:a:0",         # 明确选第一条音频流
        "-ac", "1",
        "-ar", str(SAMPLERATE),
    ]
    tail_s16 = "aresample=osf=s16"
    
    if limiter_filter:
        cmd += ["-af", _append_tail(limiter_filter, "alimiter=limit=0.98:level=disabled:attack=5:release=50:latency=1" + "," + tail_s16)]
    
    elif audio_filter:
        cmd += ["-af", _append_tail(audio_filter, tail_s16)]

    else:
        cmd += ["-af", tail_s16]

    cmd += [
        "-acodec", "pcm_s16le",
        "-f", "s16le",
        "-",
    ]

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if p.returncode != 0 or not p.stdout:
        stderr_text = p.stderr.decode("utf-8", "ignore")
        if "No such filter:" in stderr_text:
            # 从报错中提取滤镜名
            # 典型行: [AVFilterGraph @ ...] No such filter: 'adeclip'
            missing = []
            for line in stderr_text.splitlines():
                if "No such filter:" in line:
                    # 简单提取 'xxx' 中的名
                    missing.append(line[line.find("No such filter:") + len("No such filter:"):].strip().strip("'\""))
            if missing:
                raise RuntimeError(
                    "ffmpeg 报错：缺少以下滤镜："
                    + ", ".join(missing)
                    + "\n"
                    "请尝试：\n"
                    "  1. 升级系统中的 ffmpeg 至较新版本；\n"
                    "  2. 或者修改/移除 --audio-filter 参数\n"
                    f"完整错误信息：\n{stderr_text}"
                )
        raise RuntimeError(f"ffmpeg 解码失败，输出为空: {file}\nstderr={stderr_text}")

    # s16le -> float32 [-1, 1)
    audio_np = (np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768)

    return torch.from_numpy(audio_np).unsqueeze(0), int((audio_np.size * 1000 + SAMPLERATE // 2) // SAMPLERATE) # [1, T]

def merge_short_segments_adaptive(segments, max_duration_ms):
    """
    把短于 1000ms 的片段合并到相邻较短的片段中，优先合并更短的一侧，并确保合并后的总时长不超过 max_duration_ms
    仅接受字典列表：
      {"start_ms","end_ms","keep_start_ms","keep_end_ms", ...}
    """
    if not segments:
        return []

    working_segments = [dict(s) for s in segments]
    result = []

    i = 0
    while i < len(working_segments):
        seg = working_segments[i]
        
        if (seg_duration_ms := seg["end_ms"] - seg["start_ms"]) <= 1000: # 不大于 1 秒才合并
            # 获取前后邻居
            prev_seg = result[-1] if result else None
            next_seg = working_segments[i + 1] if i + 1 < len(working_segments) else None

            dur_prev_ms = (prev_seg["end_ms"] - prev_seg["start_ms"]) if prev_seg else float("inf")
            dur_next_ms = (next_seg["end_ms"] - next_seg["start_ms"]) if next_seg else float("inf")

            # 向左合的新时长 = 当前结束 - 前段开始
            dur_if_left_ms = (seg["end_ms"] - prev_seg["start_ms"]) if prev_seg else float("inf")
            # 向右合的新时长 = 后段结束 - 当前开始
            dur_if_right_ms = (next_seg["end_ms"] - seg["start_ms"]) if next_seg else float("inf")

            target_side = None
            # 两边都合法，哪边合并后更短合哪边
            if dur_if_left_ms <= max_duration_ms and dur_if_right_ms <= max_duration_ms:
                target_side = "left" if dur_if_left_ms <= dur_if_right_ms else "right"
            # 只有左边合法（意味着右边超长或不存在）
            elif dur_if_left_ms <= max_duration_ms:
                target_side = "left"
            # 只有右边合法（意味着左边超长或不存在）
            elif dur_if_right_ms <= max_duration_ms:
                target_side = "right"

            if target_side == "left":
                logger.debug(f"【VAD】片段（{seg_duration_ms / 1000:.2f} 秒）{SRTWriter._format_time(seg['start_ms'] / 1000)} --> {SRTWriter._format_time(seg['end_ms'] / 1000)}，向左合并: 前段（{dur_prev_ms / 1000:.2f} 秒）{SRTWriter._format_time(prev_seg['start_ms'] / 1000)} -->{SRTWriter._format_time(prev_seg['end_ms'] / 1000)} 延长至 {(dur_prev_ms + seg_duration_ms) / 1000:.2f} 秒")
                prev_seg["end_ms"] = seg["end_ms"]
                prev_seg["keep_end_ms"] = seg["keep_end_ms"]
                i += 1
                continue

            elif target_side == "right":
                logger.debug(f"【VAD】片段（{seg_duration_ms / 1000.0:.2f} 秒）{SRTWriter._format_time(seg['start_ms'] / 1000.0)} --> {SRTWriter._format_time(seg['end_ms'] / 1000)}，向右合并: 后段（{dur_next_ms / 1000:.2f} 秒）{SRTWriter._format_time(next_seg['start_ms'] / 1000)} -->{SRTWriter._format_time(next_seg['end_ms'] / 1000)} 延长至 {(seg_duration_ms + dur_next_ms) / 1000:.2f} 秒")
                next_seg["start_ms"] = seg["start_ms"]
                next_seg["keep_start_ms"] = seg["keep_start_ms"]
                i += 1
                continue
            
            else:
                logger.debug(f"【VAD】片段（{seg_duration_ms / 1000.0:.2f} 秒）{SRTWriter._format_time(seg['start_ms'] / 1000.0)} --> {SRTWriter._format_time(seg['end_ms'] / 1000)} 合并后超过 {max_duration_ms / 1000:.2f} 秒，放弃")

        result.append(seg)
        i += 1

    return result

def open_folder(path):
    """
    根据不同操作系统，使用默认的文件浏览器打开指定路径的文件夹
    如果 path 是文件，则打开其所在文件夹并选中（高亮）该文件（仅限 Win/Mac）
    如果 path 是文件夹，则直接打开该文件夹
    """
    path = Path(path).resolve() # 确保是绝对路径
    if not path.exists():
        logger.warn(f"目录 '{path}' 不存在，无法自动打开")
        return
    try:
        if sys.platform == "win32":
            if path.is_file():
                # Windows 使用 explorer /select,"文件路径" 来高亮文件
                subprocess.run(["explorer", "/select,", str(path)])
            else:
                os.startfile(path)

        elif sys.platform == "darwin":  # macOS
            # macOS 使用 open -R "文件路径" 来在 Finder 中揭示文件
            subprocess.run(["open", "-R", str(path)])

        else:  # Linux and other Unix-like
            target = path.parent if path.is_file() else path
            subprocess.run(["xdg-open", str(target)])
        logger.info(f"已自动为您打开目录：{path}")
    except Exception as e:
        logger.warn(f"尝试自动打开目录失败：{e}，请手动访问：{path}")

def prepare_acoustic_features(waveform, speech_probs, frame_duration_ms, use_zcr):
    """计算并对齐声学特征（能量、ZCR），供智能切分使用"""

    # 计算帧长
    frame_length = int(round(frame_duration_ms * SAMPLERATE / 1000))
    
    # 计算能量 (Standard Deviation)
    # 逻辑：Var = E[x^2] - (E[x])^2，等同于对每一帧执行 torch.std(chunk, unbiased=False)
    logger.debug("正在计算声学特征（能量）……")
    
    # 计算 E[x]
    local_mean = torch.nn.functional.avg_pool1d(
        waveform, 
        kernel_size=frame_length, 
        stride=frame_length,
        ceil_mode=True
    )
    
    # 计算 E[x^2]
    local_sq_mean = torch.nn.functional.avg_pool1d(
        waveform.pow(2), 
        kernel_size=frame_length, 
        stride=frame_length,
        ceil_mode=True
    )
    
    # 得到能量数组 (std)
    # clamp(min=0) 防止浮点误差导致出现极小的负数
    # view(-1) 展平为 (NumFrames, )
    energy_array = torch.sqrt((local_sq_mean - local_mean.pow(2)).clamp(min=0)).view(-1).numpy()

    # 计算 ZCR 
    zcr_array = None
    if use_zcr:
        logger.debug("正在计算声学特征（ZCR）……")
        
        # 去直流，waveform shape: (1, T)
        waveform_centered = waveform - waveform.mean(dim=-1, keepdim=True)
        logger.debug("【ZCR】正在计算背景底噪水平……")   
        # 计算动态噪声门限，扫描全篇找每帧最大值
        frame_maxs_flat = torch.nn.functional.max_pool1d(
            # 取绝对值
            waveform_centered.abs().unsqueeze(0), 
            kernel_size=frame_length, 
            stride=frame_length,
            ceil_mode=True
        ).view(-1)# 输出形状 [1, 1, NumFrames]
        
        # 使用 10% 分位数，防止极静帧导致门限失效
        # 这能有效忽略偶尔出现的数字静音 (0值)，找到真正的“底噪层”
        # 乘以 2 倍作为安全门限，并设定 1e-6 的硬下限
        noise_threshold = max(torch.quantile(frame_maxs_flat, 0.10).item() * 2, 1e-6)
        logger.debug(f"【ZCR】底噪门限已设定为：{noise_threshold:.6f}")

        # 计算 ZCR
        # 判定大音量区域
        is_loud = (waveform_centered.abs() > noise_threshold)
        
        # 向量化计算过零
        # 逻辑：(过零) AND (至少一边是大音量)
        zcr_array = torch.nn.functional.avg_pool1d(
            # 向量化计算所有采样点的过零情况 (1, T-1)
            # 等价于 chunk[:-1] * chunk[1:]
            # 逻辑或: 左边响 OR 右边响
            # 得到“有效过零点”的布尔矩阵 (1, T-1)
            (((waveform_centered[:, :-1] * waveform_centered[:, 1:]) < 0) & (is_loud[:, :-1] | is_loud[:,1:])).float().unsqueeze(0), # 需要 (N, C, L)
            kernel_size=frame_length,
            stride=frame_length,
            ceil_mode=True # 保证处理尾部
        ).view(-1).numpy() # 使用 .view(-1) 将输出展平为一维向量

    # 对齐长度
    # 防止 pooling 的 ceil_mode 导致多出一帧，或者 VAD 模型输出少一帧的情况
    min_len = min(len(x) for x in [speech_probs, energy_array, zcr_array] if x is not None)
    
    # 执行截断对齐
    energy_array = energy_array[:min_len]
    speech_probs = speech_probs[:min_len]
    if zcr_array is not None:
        zcr_array = zcr_array[:min_len]
    
    logger.debug(f"声学特征计算完成，有效帧数：{min_len}")
    
    return speech_probs, energy_array, zcr_array

def refine_tail_timestamp(
    last_token_start_ms,   # 该段最后一个子词的开始时间
    rough_end_ms,          # 原段结束时间
    speech_probs,          # VAD 概率序列（get_speech_timestamps_onnx 返回的第二项）
    frame_duration_ms,
    frame_times_ms,        # 每帧对应的绝对时间（ms），长度与 speech_probs 对齐
    max_end_ms,            # 该段所属 VAD 大块的硬上限
    min_silence_duration_ms,   # 判定静音的最小时长
    lookahead_ms,          # 发现静音后再向后看一点，避免马上回到语音（clamp 在区间内）
    safety_margin_ms,      # 安全边距，避免切在突变点
    min_tail_keep_ms,      # 至少保留最后 token 后的这点时长，避免切得太硬
    percentile,            # 自适应阈值，值越大，越容易将高概率有语音区域判为静音
    offset,                # 在自适应阈值的基础上增加的固定偏移量，值越大，越容易将高概率区域有语音区域判为静音
    energy_array,
    energy_percentile,
    energy_offset,
    use_zcr,
    zcr_threshold,
    zcr_array,
    zcr_high_ratio,        # 高ZCR帧的占比阈值
    kernel
):
    """
    在 [last_token_start_ms, max_end_ms] 范围内做所有自适应统计（percentile 等），
    正常情况下只在这个范围内找切点
    如果在这段内出现 “概率突然掉、但能量或 ZCR 仍然偏高” 的可疑静音，
    则允许在额外的 1 秒尾巴内继续按同一阈值寻找真正静音切点。
    """
    # 仅在“最后子词之后”的尾窗内搜索（用 frame_times_ms 对齐索引，避免漂移）
    start_idx = int(np.searchsorted(frame_times_ms, last_token_start_ms, side="left"))
    end_idx = int(np.searchsorted(frame_times_ms, max_end_ms, side="right"))

    # 允许在 [max_end_ms, max_end_ms + 1s] 内额外搜索，但不用于统计阈值（可关闭）
    search_end_idx = int(np.searchsorted(frame_times_ms, max_end_ms + OVERLAP_MS, side="right"))

    # 至少要有 4 帧才能进行有效的统计学分析
    if start_idx >= len(speech_probs) or end_idx - start_idx <= 4:
        return min(rough_end_ms, max_end_ms)

    # 只在 [last_token_start_ms, max_end_ms] 这一截上做平滑和统计
    p_smooth = np.convolve(speech_probs[start_idx:end_idx], kernel, mode="same")
    e_smooth = np.convolve(energy_array[start_idx:end_idx], kernel, mode="same")

    # 局部自适应阈值percentile + offset
    # 限制阈值范围，防止极端情况导致逻辑失效
    # 0.10 保证底噪容忍度，0.95 保证不会因为全 1.0 的概率导致无法切割
    dyn_tau = np.clip(np.percentile(p_smooth, percentile) + offset, 0.10, 0.95)

    # 取尾段能量的低分位数，找“相对安静”的能量水平
    dyn_e_tau = max(np.percentile(e_smooth, energy_percentile) + energy_offset, 1e-6)

    min_silence_frames = math.ceil(min_silence_duration_ms / frame_duration_ms)

    def _is_sudden_drop_with_high_acoustic(i):
        """
        判定当前以 i 为起点、长度为 min_silence_frames 的窗口是否属于：
        “概率突然掉，但能量/ZCR 仍然偏高”的可疑静音
        只基于 [last_token_start_ms, max_end_ms] 内的数据进行判断
        """
        # 检测“概率突降 + 能量/ZCR 仍偏高”的窗口时，回看多长时间（秒 -> 帧）
        sudden_window_frames = max(4,
            int(min(250,
                    max(
                        80,
                        0.5 * min_silence_duration_ms
                    )) / frame_duration_ms))

        prev_end = i
        prev_start = max(0, prev_end - sudden_window_frames)
        # 如果切片为空（例如 i=0 时），说明没有“之前”的数据可供比较，直接返回 False
        if prev_end <= prev_start:
            return False

        # 获取“当前点之前”的一小段的最大概率
        prev_max_p = p_smooth[prev_start:prev_end].max()
        # 获取“当前点之后”的一小段（静音窗口）的平均概率
        curr_mean_p = p_smooth[i : i + min_silence_frames].mean()

        # 确保·前面是高置信度语音（显著高于静音阈值），再抬一点，防止把低置信噪声也当作“突降前的语音”
        if prev_max_p < dyn_tau + offset:
            return False

        # 确保当前窗口的概率相对前面明显“突降”
        if prev_max_p - curr_mean_p < 0.35:
            return False

        # 能量或 ZCR 仍然偏高，说明声学上还很“活跃”，不像真正静音
        high_energy = (
            np.mean(e_smooth[prev_start:prev_end]) > dyn_e_tau
            or np.mean(e_smooth[i : i + min_silence_frames]) > dyn_e_tau
        )

        high_zcr = False
        if use_zcr:
            z_start = start_idx + i
            z_end = z_start + min_silence_frames
            high_zcr = (
                np.mean(zcr_array[z_start:z_end] > zcr_threshold)
                > zcr_high_ratio
            )

        return high_energy or high_zcr

    def _is_stable_silence(i, p_smooth, e_smooth, global_base_idx):
        """
        判断当前片段是否为合格的静音切点
        i: 在当前片段 p_smooth 中的相对索引
        p_smooth: 当前分析的概率片段 (p_smooth 或 extra_p_smooth)
        e_smooth: 当前分析的能量片段
        global_base_idx: p_smooth 开头对应在原始大数组中的绝对索引 (用于取 ZCR)
        """
        # 概率是否持续低于自适应阈值
        if not np.all(p_smooth[i : i + min_silence_frames] < dyn_tau):
            return False

        # 能量是否也处于本段“相对静音”区间
        if not (np.mean(e_smooth[i : i + min_silence_frames]) < dyn_e_tau):
            return False

        # ZCR 保护 (检测清辅音)
        if use_zcr:
            z_start = global_base_idx + i
            z_end = z_start + min_silence_frames
            # 获取窗口内的 ZCR 数据
            # 计算窗口内超过 ZCR 阈值的帧的比例
            # np.mean(布尔数组) 相当于计算 True 的百分比
            high_zcr_ratio = np.mean(zcr_array[z_start:z_end] > zcr_threshold)
            # 只有当高 ZCR 帧的比例超过设定值（如 0.3）时，才触发保护
            if high_zcr_ratio > zcr_high_ratio:
                # 这里仍用 frame_duration_ms 做日志的近似展示（核心时间戳计算已改用 frame_times_ms）
                logger.debug(
                    f"【ZCR】疑似静音段 {SRTWriter._format_time((z_start * frame_duration_ms) / 1000)} --> {SRTWriter._format_time((z_end * frame_duration_ms) / 1000)} 内高频帧占比 {high_zcr_ratio:.2f} > {zcr_high_ratio}，视为清辅音，跳过切分"
                )
                return False

        # 滞回检查 —— 确认后面不会马上回到高概率语音（clamp 在区间内）
        lookahead_frames = math.ceil(lookahead_ms / frame_duration_ms)
        if i + min_silence_frames < min(len(p_smooth), i + min_silence_frames + lookahead_frames):
            if np.any(
                p_smooth[
                    i + min_silence_frames : min(len(p_smooth), i + min_silence_frames + lookahead_frames)
                ] >= min(0.98, dyn_tau + offset)
            ):
                return False

        return True

    def _calc_refined_time(idx_offset, limit_ms):
        """计算最终时间戳（ms），应用安全边距和最大限制；idx_offset 为全局帧索引"""
        return min(
            max(
                frame_times_ms[idx_offset] + safety_margin_ms,
                last_token_start_ms + min_tail_keep_ms
            ),
            limit_ms
        )

    # 标记：是否在分析区间内遇到过“概率突降 + 高能量/ZCR”的可疑静音
    allow_use_extra_tail = False

    # === 只在 [last_token_start_ms, max_end_ms] 内做切分 ===
    for i in range(len(p_smooth) - min_silence_frames + 1):
        # 检查这段是否属于「概率突降 + 高能量/ZCR」的可疑静音
        if _is_sudden_drop_with_high_acoustic(i):
            allow_use_extra_tail = True
            # 这类可疑静音本身不作为切点
            continue

        # 通用检查：是否稳定静音
        if _is_stable_silence(i, p_smooth, e_smooth, start_idx):
            return _calc_refined_time(start_idx + i, max_end_ms)

    # === 第二阶段：是否允许在 [max_end_ms, max_end_ms + 1s] 内继续找切点 ===
    if not allow_use_extra_tail:
        return min(rough_end_ms, max_end_ms)

    # 有可疑静音，可以利用已有阈值在多出的 1 秒内继续搜索
    # 只对额外 1s 做平滑，仍然使用之前在 [last_token_start_s, max_end_s] 内得到的 dyn_tau / dyn_e_tau。
    extra_p_smooth = np.convolve(speech_probs[end_idx:search_end_idx], kernel, mode="same")
    extra_e_smooth = np.convolve(energy_array[end_idx:search_end_idx], kernel, mode="same")

    for j in range(len(extra_p_smooth) - min_silence_frames + 1):
        if _is_stable_silence(j, extra_p_smooth, extra_e_smooth, end_idx):
            # 上限放宽到 max_end_s + OVERLAP_S
            return _calc_refined_time(end_idx + j, max_end_ms + OVERLAP_MS)

    # 兜底：没找到稳定静音，保留原结束时间
    return min(rough_end_ms, max_end_ms)

def save_tensor_as_wav(path, waveform):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes((np.clip(waveform.squeeze(0).detach().cpu().numpy(), -1.0, 1.0) * 32767).astype('<i2').tobytes())

def slice_waveform_ms(waveform, start_ms, end_ms):
    """
    高效 Tensor 切片与 Padding
    waveform: [1, T]
    """
    # 边界限制并切片，F.pad: (padding_left, padding_right)，pad 操作会产生副本
    return torch.nn.functional.pad(
        waveform[:, int(start_ms * SAMPLERATE // 1000):int(end_ms * SAMPLERATE // 1000)],
        (int(PAD_SECONDS * SAMPLERATE), int(PAD_SECONDS * SAMPLERATE))
        )

def transcribe_audio(model, audio, batch_size=1):
    """封装转录逻辑：调用模型 -> 解码 -> 返回子词列表"""
    return model.transcribe(audio=audio, batch_size=batch_size, return_hypotheses=True, verbose=False)

def main(argv=None):
    parser = arg_parser()

    # CLI 模式：argv=None，使用真实命令行参数；API 模式：argv 由 server.py 传入
    if argv is None:
        args = parser.parse_args()
        api_mode = False
    else:
        args = parser.parse_args(argv)
        api_mode = True

    # API 模式：在内存里累积输出；CLI 模式：写文件
    api_result = {} if api_mode else None

    if args.input_file is None:
        # CLI 模式没传参数，自动转为启动 Server
        import server 
        server.main()
        return  # 启动服务后直接结束 main 函数，防止往下执行报错

    # 配置全局日志等级
    logger.set_debug(args.debug and not api_mode)

    if args.audio_filter is not None and args.limiter_filter is not None:
        parser.error("【参数冲突】--audio-filter 和 --limiter-filter 不能同时使用")
                     
    # 校验 --no-chunk 和 VAD 参数的冲突
    # 只有当用户修改了默认值，但加了 --no-chunk 时才报错
    if args.no_chunk:    
        # 动态检查当前参数值是否等于定义时的 default 值
        # getattr(args, p) 获取当前解析到的值
        # parser.get_default(p) 获取定义时的默认值
        for p in [
            "vad_threshold",
            "vad_end_threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "keep_silence"
        ]:
            if getattr(args, p) != parser.get_default(p):
                parser.error(f"【参数错误】已添加 --no-chunk，不能设置参数 --{p}")

        # 校验 --no-chunk 和 --zcr 的冲突
        if args.zcr:
            parser.error("【参数冲突】使用--zcr（过零率检测）功能必须开启 VAD，因此不能与 --no-chunk 一起使用")

        # 校验 --no-chunk 和 --refine-tail 的冲突
        if args.refine_tail:
            parser.error("【参数冲突】使用--refine-tail（段尾精修）功能必须开启 VAD，因此不能与 --no-chunk 一起使用")

    # 校验未使用 --zcr 却指定了相关参数的情况
    # 只有当用户修改了默认值（即想要调整精修参数），但忘了加 --zcr 开关时才报错
    if not args.zcr:
        if getattr(args, "zcr_threshold") != parser.get_default("zcr_threshold"):
            parser.error(f"【参数错误】未添加 --zcr，不能设置参数 --zcr_threshold")
        if getattr(args, "tail_zcr_high_ratio") != parser.get_default("tail_zcr_high_ratio"):
            parser.error(f"【参数错误】未添加 --zcr，不能设置参数 --tail_zcr_high_ratio")
        if args.auto_zcr:
            parser.error(f"【参数错误】未添加 --zcr，不能设置参数 --auto_zcr")

    # 校验未使用 --refine-tail 却指定了相关参数的情况
    # 只有当用户修改了默认值（即想要调整精修参数），但忘了加 --refine-tail 开关时才报错
    if not args.refine_tail:
        # 动态检查当前参数值是否等于定义时的 default 值
        # getattr(args, p) 获取当前解析到的值
        # parser.get_default(p) 获取定义时的默认值
        for p in [
            "tail_percentile",
            "tail_offset",
            "tail_energy_percentile",
            "tail_energy_offset",
            "tail_lookahead_ms",
            "tail_safety_margin_ms",
            "tail_min_keep_ms",
            "tail_zcr_high_ratio"
        ]:
            if getattr(args, p) != parser.get_default(p):
                parser.error(f"【参数错误】未添加 --refine-tail，不能设置参数 --{p}")

    if not args.no_chunk:
        if not (local_onnx_model_path := Path(__file__).resolve().parent / "models" / "model_quantized.onnx").exists():
            logger.warn(
                f"未在 '{local_onnx_model_path}' 中找到 Pyannote-segmentation-3.0 模型"
            )
            logger.warn("请下载 model_quantized.onnx 并放入 models 文件夹")
            if api_mode:
                return {
                    "error": {
                        "message": "未找到 Pyannote-segmentation-3.0 模型，请下载 model_quantized.onnx 并放入 models 文件夹",
                        "type": "server_error",
                        "param": None,
                        "code": "pyannote_model_missing",
                    }
                }
            return

        # ==== VAD 阈值参数校验和默认 ====
        if args.vad_end_threshold is None:
            args.vad_end_threshold = max(0.05, args.vad_threshold - 0.15)
        if not (0.05 < args.vad_threshold <= 1.0):
            parser.error(f"vad_threshold 必须在（0.05-1）范围内，当前值错误")
        if not (0.05 <= args.vad_end_threshold <= 1.0):
            parser.error(f"vad_end_threshold 必须在（0.05-1）范围内，当前值错误")
        if args.vad_end_threshold > args.vad_threshold:
            parser.error(
                f"vad_end_threshold 不能大于 vad_threshold"
            )
    
        if args.refine_tail:
            if not (0.0 <= args.tail_percentile <= 100.0):
                parser.error(f"tail_percentile 必须在（0-100）范围内，当前值错误")
            if not (0.0 <= args.tail_energy_percentile <= 100.0):
                parser.error("tail_energy_percentile 必须在（0-100）范围内，当前值错误")
            if not (0.1 <= args.tail_zcr_high_ratio <= 0.5):
                parser.error("tail_zcr_high_ratio 必须在（0.1-0.5）范围内，当前值错误")

    # 启动内存 / 显存监控线程
    if args.debug and not api_mode:
        mem_stop_event = threading.Event()
        mem_thread = threading.Thread(
            target=_memory_monitor,
            args=(mem_stop_event, torch.cuda.current_device() if torch.cuda.is_available() else 0), 
            daemon=True,
        )
        mem_thread.start()

    # --- 准备路径和临时文件 ---
    input_path = Path(args.input_file)
    temp_full_wav_path = None
    temp_chunk_dir = None

    # --- 执行核心的语音识别流程 ---
    vad_model_load_start = 0
    vad_model_load_end = 0
    recognition_start_time = 0
    recognition_end_time = 0
    original_decoding_cfg = None
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        logger.info(f"正在转换输入文件 '{input_path}'……")
        waveform, total_audio_ms = load_audio_ffmpeg(input_path, args.audio_filter, args.limiter_filter)
        logger.info("转换完成")

        # 使用单例模式获取模型，避免重复加载
        model = get_asr_model()

        # 获取模型最大允许输入语音块长度
        MAX_SPEECH_DURATION_MS = int(round(model.cfg.train_ds.max_duration * 1000))

        # 设定 Batch Size
        if args.batch_size is None:
            if torch.cuda.is_available():
                logger.info(f"正在自动估算 Batch Size 的值……")
                BATCH_SIZE = auto_tune_batch_size(model, model.cfg.train_ds.max_duration)
                logger.info(f"自动估算 Batch Size 值为 {BATCH_SIZE}")
            else:
                BATCH_SIZE = 1
                logger.info("仅使用 CPU 时默认 Batch Size 值为 1")
        else:
            BATCH_SIZE = args.batch_size
            logger.info(f"用户指定 Batch Size 值为 {BATCH_SIZE}")

        if args.beam != model.cfg.decoding.beam.beam_size:
            # 使用深拷贝来完全复制原始配置，确保它不受任何后续修改的影响
            original_decoding_cfg = copy.deepcopy(model.cfg.decoding)
            
            # 同样使用深拷贝创建新配置，确保它与原始配置完全独立
            new_decoding_cfg = copy.deepcopy(model.cfg.decoding)
            new_decoding_cfg.beam.beam_size = args.beam
            
            # 在所有识别任务开始前，应用这个新的解码策略
            logger.info(f"正在应用新的解码策略：集束搜索宽度为 {args.beam} ……")
            model.change_decoding_strategy(new_decoding_cfg)

        # --- 逐块识别并校正时间戳 ---
        # --- 计时开始：核心识别流程 ---
        recognition_start_time = time.perf_counter()

        vad_chunk_end_times_ms = []  # 用于存储VAD块的结束时间
        
        # --- 批处理相关变量 ---
        batch_audio = [] # 待处理的音频数据队列
        batch_meta = [] # 存 {'index': chunk_index, 'offset': time_offset}
        # 预先为所有可能的一级块分配结果列表，稍微多一点空间也没事，后续按 append 顺序对应
        # 在 VAD 循环中动态添加
        all_chunk_subwords_collection = [] 

        def flush_batch():
            """内部闭包：执行批量推理并归位结果"""
            if not batch_audio:
                return

            # ---- OOM 检测辅助函数 ----
            def _is_oom_error(e: Exception):
                msg = str(e).lower()
                # 兼容多种形式：新版本的 OutOfMemoryError 类型 + 旧版本的 RuntimeError 文本
                return (
                    isinstance(e, torch.cuda.OutOfMemoryError)
                    or ("out of memory" in msg and ("cuda" in msg or "cudnn" in msg or "cublas" in msg))
                )
            hit_oom = False
            # ---- 第一次尝试：用当前 batch_size 一次性跑完 ----
            try:
                with torch.inference_mode():
                    hyps = transcribe_audio(model, batch_audio, len(batch_audio))[0]  # 取元组第 0 个元素，即结果列表
    
            except RuntimeError as e:
                if _is_oom_error(e):
                    hit_oom = True
                    logger.warn("显存不足，当前批次将回退为逐条推理……")
                else:
                    # 非 OOM 的错误照常抛出
                    raise
        
            # ---- OOM 回退路径：不在 except 里重跑，而是在外面重试 ----
            if hit_oom:
                if torch.cuda.is_available():
                    # 尽量释放一下显存碎片
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as e:
                        logger.debug(f"重置显存占用峰值统计失败：{e}")

        
                with torch.inference_mode():
                    # 回退为逐条推理（内部会顺序处理整个 batch_audio 列表）
                    hyps = transcribe_audio(model, batch_audio)[0]

            # 处理结果
            for hyp, meta in zip(hyps, batch_meta):
                if not hyp:
                    continue
                # 解码
                decoded_subwords = decode_hypothesis(model, hyp).subwords

                # 所有边界都统一到“整数微秒域”，加上时间偏移
                for sub in decoded_subwords:
                    # token 时间：统一用“整数微秒 floor”
                    sub_us = int(sub.seconds * 1_000_000) + int(meta["base_offset_ms"]) * 1000
                    # 用微秒判断归属（严格半开区间），start: floor（偏早），end: ceil（偏晚）
                    if int(meta["keep_start_ms"]) * 1000 <= sub_us < int(meta["keep_end_ms"]) * 1000:
                        # 保持 token 时间戳也从同一套 us 派生，写回 seconds
                        sub.seconds = sub_us / 1_000_000
                        all_chunk_subwords_collection[meta["chunk_index"]].append(sub)

            batch_audio.clear()
            batch_meta.clear()

        # --- 分支 A: 不分块 (No Chunk) ---
        if args.no_chunk:
            full_chunk = slice_waveform_ms(waveform, 0, total_audio_ms)
            if args.debug and not api_mode:
                fd, temp_full_wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                save_tensor_as_wav(temp_full_wav_path, full_chunk)

            logger.info("未使用VAD，一次性处理整个文件……")
            
            # 同样走 batch 流程（虽然只有 1 个）
            all_chunk_subwords_collection.append([]) # index 0
            
            batch_audio.append(full_chunk.squeeze(0))
            batch_meta.append({
                "chunk_index": 0,
                "base_offset_ms": 0,
                "keep_start_ms": 0,
                "keep_end_ms": int(total_audio_ms),
            })
            flush_batch()

        # --- 分支 B: 使用 VAD 分块 ---
        else:
            logger.info("【VAD】正在从本地路径加载 Pyannote-segmentation-3.0 模型……")
            vad_model_load_start = time.perf_counter()
            onnx_session = onnxruntime.InferenceSession(
                str(local_onnx_model_path), providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.perf_counter()
            logger.info(f"【VAD】模型加载完成")

            # 运行 VAD
            speeches, speech_probs, frame_duration_ms, frame_times_ms = get_speech_timestamps_onnx(
                waveform,
                onnx_session,
                args.vad_threshold,
                args.vad_end_threshold,
                args.min_speech_duration_ms,
                args.min_silence_duration_ms
            )

            if not speeches:
                logger.warn("【VAD】未侦测到语音活动")
                if api_mode:
                    return {
                        "error": {
                            "message": "VAD 未侦测到语音活动",
                            "type": "invalid_request_error",
                            "param": "file",
                            "code": "no_speech",
                        }
                    }
                return

            vad_chunk_end_times_ms.extend([seg[1] for seg in speeches])

            # 声学特征准备
            if args.auto_zcr or args.refine_tail:
                speech_probs, energy_array, zcr_array = prepare_acoustic_features(waveform, speech_probs, frame_duration_ms, args.zcr)
            # 对齐 frame_times_ms
            frame_times_ms = frame_times_ms[:len(speech_probs)]

            # ZCR 阈值校准 
            # 确定最终使用的 ZCR 阈值
            final_zcr_threshold = args.zcr_threshold
            # 只有开启了 ZCR 且开启了 Auto 才进行校准
            if args.auto_zcr:
                if (calibrated := calibrate_zcr_threshold(speech_probs, zcr_array)) is not None:
                    final_zcr_threshold = calibrated

            # 合并短块
            logger.debug("【VAD】正在合并短于1秒的语音块……")
            merged_ranges_s = assign_vad_ownership_keep_windows(
                merge_short_segments_adaptive(
                    [{"start_ms": s[0],"end_ms": s[1],"keep_start_ms": s[0],"keep_end_ms": s[1]}for s in speeches],
                    MAX_SPEECH_DURATION_MS
                ),
                args.keep_silence,
                total_audio_ms
            )
            if len(speeches) != len(merged_ranges_s):
                logger.debug(f"【VAD】原 {len(speeches)} 个语音块已合并为 {len(merged_ranges_s)} 个语音块")
            else:
                logger.debug(f"【VAD】没有需要合并的语音块")
            
            # 初始化结果收集器
            all_chunk_subwords_collection = [[] for _ in range(len(merged_ranges_s))]

            if args.debug and not api_mode:
                temp_chunk_dir = Path(tempfile.mkdtemp())

            # 遍历一级块
            for i, chunk_dict in enumerate(merged_ranges_s):
                # --- 准备一级语音块 ---
                # 计算包含静音保护的起止时间
                start_ms = max(0, chunk_dict["start_ms"] - args.keep_silence)
                end_ms = min(total_audio_ms, chunk_dict["end_ms"] + args.keep_silence)
                
                # 一级切片 (使用 Tensor 切片)
                chunk = slice_waveform_ms(waveform, start_ms, end_ms)

                # debug模式下为了检查才导出该块
                if args.debug and not api_mode:
                    save_tensor_as_wav(temp_chunk_dir / f"chunk_{i + 1}.wav", chunk)

                logger.info(
                    f"【VAD】正在处理语音块 {i + 1}/{len(merged_ranges_s)} （该块起止时间：{SRTWriter._format_time(chunk_dict['start_ms'] / 1000)} --> {SRTWriter._format_time(chunk_dict['end_ms'] / 1000)}，时长：{(chunk_dict['end_ms'] - chunk_dict['start_ms']) / 1000:.2f} 秒）",
                    end="", flush=True,
                    )

                # 判断短块还是长块
                if (chunk_dict["end_ms"] - chunk_dict["start_ms"]) <= MAX_SPEECH_DURATION_MS // 3:
                    # === 短块 ===
                    logger.info(" --> 短块，直接加入 Batch")
                    
                    # 加入 Batch
                    batch_audio.append(chunk.squeeze(0))
                    # 坐标变换：全局时间 = 一级块偏移 + 相对时间
                    # 短块的 keep 窗口就是它在全局时间轴上的起止时间
                    batch_meta.append({
                        "chunk_index": i,
                        "base_offset_ms": start_ms,
                        "keep_start_ms": chunk_dict["own_keep_start_ms"],
                        "keep_end_ms": chunk_dict["own_keep_end_ms"],
                    })
                
                else:
                    # === 长块 (二次切分) ===
                    # 运行局部 VAD
                    sub_speeches, sub_speech_probs, sub_frame_duration_ms, sub_frame_times_ms = get_speech_timestamps_onnx(
                        chunk,
                        onnx_session,
                        args.vad_threshold,
                        args.vad_end_threshold,
                        args.min_speech_duration_ms,
                        args.min_silence_duration_ms
                    )
                    
                    if not sub_speeches:
                        # 兜底策略：如果二次 VAD 没切出来（例如全是噪音），回退到整块识别
                        logger.warn("    【VAD】二次 VAD 未发现有效分割点，回退到整块识别")
                        # 构造一个覆盖整个 chunk 的虚拟 VAD 段
                        sub_speeches = [[0, int((chunk.shape[1] * 1000) // SAMPLERATE)]]

                    # 局部特征计算
                    sub_speech_probs, sub_energy_array, sub_zcr_array = prepare_acoustic_features(
                            chunk, sub_speech_probs, sub_frame_duration_ms, args.zcr
                    )

                    # 对齐 frame_times_ms
                    sub_frame_times_ms = sub_frame_times_ms[:len(sub_speech_probs)]
                    
                    # 智能切分
                    nonsilent_ranges_ms = [] 
                    for seg in sub_speeches:
                        # seg 是列表 [start, end]
                        # seg[1] 是相对于含 Padding 的 chunk 的时间
                        # start_ms 是 chunk 在原音频中的起始时间（含 keep_silence）
                        # PAD_MS 是 chunk 头部人为添加的静音
                        vad_chunk_end_times_ms.append(start_ms + seg[1] - PAD_MS)
                        nonsilent_ranges_ms.extend(global_smart_segmenter(
                                seg[0], 
                                seg[1],
                                sub_speech_probs,      # 局部概率
                                sub_energy_array,      # 局部能量
                                sub_frame_times_ms,
                                MAX_SPEECH_DURATION_MS,
                                args.vad_threshold,
                                final_zcr_threshold,   # 复用全局计算出的最佳阈值
                                sub_zcr_array         # 局部 ZCR
                            ))
                    logger.debug(
                            f"【VAD】侦测到 {len(nonsilent_ranges_ms)} 个子语音块"
                        )

                    # 再次合并子块
                    refined_sub_speeches = merge_short_segments_adaptive(nonsilent_ranges_ms, MAX_SPEECH_DURATION_MS)

                    if len(refined_sub_speeches) > 1:
                        logger.info(f"    --> Batch 中加入 {len(refined_sub_speeches)} 个子片段：")

                    # 遍历子块加入 Batch
                    for sub_idx, sub_seg in enumerate(refined_sub_speeches):
                        # sub_seg 是相对于 chunk (已含Padding) 的秒数
                        # 提取子块音频并加上静音保护
                        sub_chunk = slice_waveform_ms(chunk, sub_seg["start_ms"], sub_seg["end_ms"])
                        
                        # Debug 时导出临时子块文件
                        if args.debug and not api_mode:
                            save_tensor_as_wav(temp_chunk_dir / f"chunk_{i + 1}_sub_{sub_idx + 1}.wav", sub_chunk)

                        if len(refined_sub_speeches) != 1:
                            logger.info(f"第 {i + 1}-{sub_idx + 1} 段 {SRTWriter._format_time(sub_seg['start_ms'] / 1000)} --> {SRTWriter._format_time(sub_seg['end_ms'] / 1000)}，时长：{(sub_seg['end_ms'] - sub_seg['start_ms']) / 1000:.2f} 秒")
                        
                        batch_audio.append(sub_chunk.squeeze(0))
                        batch_meta.append({
                            "chunk_index": i,
                            # 最终时间 = 一级块全局偏移 + 二级块在一级块内的偏移 + 识别出的相对时间
                            # 因为 sub_seg[0] 是基于含填充的父chunk计算的，它包含了父chunk头部的 0.5s 静音，必须扣除
                            "base_offset_ms": start_ms + sub_seg["start_ms"] - PAD_MS,
                            "keep_start_ms": max(
                                                 start_ms + sub_seg["keep_start_ms"] - PAD_MS,
                                                 chunk_dict["own_keep_start_ms"],
                                             ),
                            "keep_end_ms": min(
                                               start_ms + sub_seg["keep_end_ms"] - PAD_MS,
                                               chunk_dict["own_keep_end_ms"],
                                           ),
                        })
                        
                        # 如果 batch 满了，立即执行
                        if len(batch_audio) >= BATCH_SIZE:
                            flush_batch()

                # 每一级块循环末尾，检查 batch 是否满了
                if len(batch_audio) >= BATCH_SIZE:
                    flush_batch()

            # 循环结束后，处理剩余的 batch
            flush_batch()
            
        raw_subwords = []
        if all_chunk_subwords_collection:
            for sub_list in all_chunk_subwords_collection:
                raw_subwords.extend(sub_list)
            logger.debug(f"【VAD】所有语音块处理完毕")

        # --- 计时结束：核心识别流程 ---
        recognition_end_time = time.perf_counter()

        # 如果整个过程下来没有任何识别结果，提前告知用户并退出，避免生成空文件
        if not raw_subwords:
            logger.info("=" * 70)
            logger.info("未识别到任何有效的语音内容")
            if api_mode:
                return {
                    "error": {
                        "message": "未识别到任何有效语音内容",
                        "type": "invalid_request_error",
                        "param": "file",
                        "code": "empty_result",
                    }
                }
            return

        logger.info("=" * 70)
        logger.info("正在根据子词和VAD边界生成精确文本片段……")

        all_segments, all_subwords = create_precise_segments_from_subwords(
            raw_subwords, sorted(set(vad_chunk_end_times_ms)), total_audio_ms, no_remove_punc=args.no_remove_punc
            )

        logger.info("文本片段生成完成")

        if args.refine_tail: # 只在启用精修且map存在时精修
            # 短窗平滑，5 是窗口大小
            kernel = np.ones(5, dtype=np.float32) / 5

            logger.info("【精修】正在修除每段的尾部静音……")
            for i, segment in enumerate(all_segments):
                # 遍历map，所以需要通过索引i来更新原始的all_segments列表
                segment.end_seconds = refine_tail_timestamp(
                    int(round(segment.subwords[-1].seconds * 1000)),
                    int(round(segment.end_seconds * 1000)),
                    speech_probs,
                    frame_duration_ms,
                    frame_times_ms,
                    int(round((min(segment.vad_limit, all_segments[i + 1].start_seconds) if i < len(all_segments) - 1 else segment.vad_limit) * 1000)),
                    args.min_silence_duration_ms,
                    args.tail_lookahead_ms,
                    args.tail_safety_margin_ms,
                    args.tail_min_keep_ms,
                    args.tail_percentile,
                    args.tail_offset,
                    energy_array,
                    args.tail_energy_percentile,
                    args.tail_energy_offset,
                    args.zcr,
                    final_zcr_threshold,
                    zcr_array,
                    args.tail_zcr_high_ratio,
                    kernel
                ) / 1000
            # 邻段防重叠微调
            for i in range(len(all_segments) - 1):
                if all_segments[i].end_seconds > all_segments[i + 1].start_seconds:
                    all_segments[i].end_seconds = all_segments[i + 1].start_seconds
            logger.info("【精修】结束时间戳精修完成")

        # --- 根据参数生成输出文件 ---
        logger.info("=" * 70)
        logger.info("识别完成，正在生成输出文件……")

        output_dir = input_path.parent.resolve()
        base_name = input_path.stem

        # 检查用户是否指定了任何一种文件输出格式
        if not (file_output_requested := any(
                (
                    args.text,
                    args.segment,
                    args.segment2srt,
                    args.segment2vtt,
                    args.segment2tsv,
                    args.subword,
                    args.subword2srt,
                    args.subword2json,
                    args.kass,
                )
            )) or args.text:
            full_text = " ".join(segment.text for segment in all_segments)

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested and not api_mode:
            logger.info("\n识别结果（完整文本）：")
            logger.info(full_text)
            logger.info("=" * 70)
            logger.info("未指定输出参数，结果将打印至控制台")
            logger.info("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            if api_mode:
                api_result["text"] = full_text
            else:
                with (output_path := output_dir / f"{base_name}.txt").open("w", encoding="utf-8") as f: 
                    f.write(full_text)
                logger.info(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            if api_mode:
                api_result["segment"] = [
                    {
                        "start": segment.start_seconds,
                        "end": segment.end_seconds,
                        "tokens_id": [sw.token_id for sw in segment.subwords],
                        "text": segment.text,
                    }
                    for segment in all_segments
                ]
            else:
                with (output_path := output_dir / f"{base_name}.segments.txt").open("w", encoding="utf-8") as f:
                    writer = TextWriter(f)
                    for segment in all_segments:
                        writer.write(segment)
                logger.info(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            def _write_segment2srt(f):
                writer = SRTWriter(f)
                for segment in all_segments:
                    writer.write(segment)
            if api_mode:
                buf = io.StringIO()
                _write_segment2srt(buf)
                api_result["segment2srt"] = buf.getvalue()
            else:
                with (output_path := output_dir / f"{base_name}.srt").open("w", encoding="utf-8") as f:
                    _write_segment2srt(f)
                logger.info(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.segment2vtt:
            def _write_segment2vtt(f):
                writer = VTTWriter(f)
                writer.write_header()
                for segment in all_segments:
                    writer.write(segment)
            if api_mode:
                buf = io.StringIO()
                _write_segment2vtt(buf)
                api_result["segment2vtt"] = buf.getvalue()
            else:
                with (output_path := output_dir / f"{base_name}.vtt").open("w", encoding="utf-8") as f:
                    _write_segment2vtt(f)
                logger.info(f"文本片段 WebVTT 字幕文件已保存为：{output_path}")

        if args.segment2tsv:
            def _write_segment2tsv(f):
                writer = TSVWriter(f)
                writer.write_header()
                for segment in all_segments:
                    writer.write(segment)
            if api_mode:
                buf = io.StringIO()
                _write_segment2tsv(buf)
                api_result["segment2tsv"] = buf.getvalue()
            else:
                with (output_path := output_dir / f"{base_name}.tsv").open("w", encoding="utf-8") as f:
                    _write_segment2tsv(f)
                logger.info(f"文本片段 TSV 文件已保存为：{output_path}")

        if args.subword:
            if api_mode:
                api_result["subword"] = [
                    {
                        "start": sub.seconds,
                        "end": sub.end_seconds,
                        "token": sub.token.replace(" ", ""),
                    }
                    for sub in all_subwords
                ]
            else:
                with (output_path := output_dir / f"{base_name}.subwords.txt").open("w", encoding="utf-8") as f:
                    for sub in all_subwords:
                        f.write(
                            f"[{SRTWriter._format_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                        )
                logger.info(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            def _write_subword2srt(f):
                for i, sub in enumerate(all_subwords):
                    f.write(f"{i + 1}\n")
                    f.write(f"{SRTWriter._format_time(sub.seconds)} --> {SRTWriter._format_time(sub.end_seconds)}\n")
                    f.write(f"{sub.token}\n\n")
            if api_mode:
                buf = io.StringIO()
                _write_subword2srt(buf)
                api_result["subword2srt"] = buf.getvalue()
            else:
                with (output_path := output_dir / f"{base_name}.subwords.srt").open("w", encoding="utf-8") as f:
                    _write_subword2srt(f)
                logger.info(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            def _write_subword2json():
                return [
                    {"token": sub.token, "timestamp": sub.seconds}
                    for sub in all_subwords
                ]
            if api_mode:
                api_result["subword2json"] = _write_subword2json()
            else:
                with (output_path := output_dir / f"{base_name}.subwords.json").open("w", encoding="utf-8") as f:
                    json.dump(_write_subword2json(), f, ensure_ascii=False, indent=4)
                logger.info(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            def _write_kass(f):
                # 使用 writer 生成标准文件头
                writer = ASSWriter(f)
                writer.write_header()

                for segment in all_segments:
                    karaoke_text = "".join(
                        f"{{\\k{math.ceil(round((sub.end_seconds - sub.seconds) * 100))}}}{sub.token}"
                        for sub in segment.subwords
                    )
                    f.write(f"Dialogue: 0,{ASSWriter._format_time(segment.start_seconds)},{ASSWriter._format_time(segment.end_seconds)},Default,,0,0,0,,{karaoke_text}\n")
            if api_mode:
                buf = io.StringIO()
                _write_kass(buf)
                api_result["kass"] = buf.getvalue()
            else:
                with (output_path := output_dir / f"{base_name}.ass").open("w", encoding="utf-8-sig") as f:
                    _write_kass(f)
                logger.info(f"卡拉OK式 ASS 字幕已保存为：{output_path}")

    finally:
        # 恢复原始解码策略，以防后续有其他操作
        if original_decoding_cfg is not None:
            logger.info("正在恢复原始解码策略……")
            model.change_decoding_strategy(original_decoding_cfg)

        logger.info("=" * 70)
        # 使用全局记录的加载耗时
        if _ASR_MODEL_LOAD_COST > 0:
            logger.info(
                f"ReazonSpeech模型加载耗时：{format_duration(_ASR_MODEL_LOAD_COST)}"
            )

        if recognition_end_time - recognition_start_time > 0:
            # 只有当VAD加载时间大于0时才打印，否则不显示
            if vad_model_load_end - vad_model_load_start > 0:
                logger.info(
                    f"Pyannote-segmentation-3.0 模型加载耗时：{format_duration(vad_model_load_end - vad_model_load_start)}"
                )
                logger.info(
                    f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time - (vad_model_load_end - vad_model_load_start))}"
                )
            else:
                logger.info(
                    f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time)}"
                )

        if args.debug and not api_mode:
            logger.debug("=" * 70)
            # 先停止监控线程
            try:
                mem_stop_event.set()
                mem_thread.join(timeout=1.0)
            except Exception as e:
                logger.debug(f"停止监控线程时发生异常：{e}")

            # 计算并打印内存 / 显存统计
            if MEM_SAMPLES["ram_gb"]:
                ram_mode, freq = _calc_mode(MEM_SAMPLES["ram_gb"])
                logger.debug(f"【内存监控】内存常规用量：{ram_mode:.2f} GB，峰值：{MEM_PEAKS['ram_gb']:.2f} GB（出现 {freq} 次）")
            else:
                logger.debug("【内存监控】未采集到内存占用数据")
    
            # GPU 显存两套信息：采样众数 + PyTorch 精确峰值
            if torch.cuda.is_available():
                # 采样数据的统计
                if MEM_SAMPLES["gpu_gb"]:
                    gpu_mode, freq = _calc_mode(MEM_SAMPLES["gpu_gb"])
                    logger.debug(f"【显存监控】显存常规用量：{gpu_mode:.2f} GB，峰值：{MEM_PEAKS['gpu_gb']:.2f} GB（出现 {freq} 次）")
    
                # PyTorch 记录的峰值（不依赖采样间隔）
                try:
                    
                    if (peak_gpu_bytes := torch.cuda.max_memory_allocated()) > 0:
                        logger.debug(
                            f"【显存监控】PyTorch 记录的显存占用峰值：{peak_gpu_bytes / (1024 ** 3):.2f} GB"
                        )
                except Exception as e:   
                    logger.debug(f"获取 PyTorch 显存占用峰值统计失败：{e}")

            logger.debug("=" * 70)
            logger.debug("调试模式已启用，保留临时文件和目录")
            if temp_full_wav_path:
                logger.debug(f"临时 WAV 文件位于: {temp_full_wav_path}")
                open_folder(temp_full_wav_path)
            if temp_chunk_dir:
                logger.debug(f"临时分块目录位于: {temp_chunk_dir}")
                open_folder(temp_chunk_dir) # 调用新函数打开文件夹

    if api_mode:
        api_result["duration"] = total_audio_ms / 1000
        return api_result            

if __name__ == "__main__":
    main()