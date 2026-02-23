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
from pathlib import Path

from reazonspeech.nemo.asr.transcribe import load_model
from reazonspeech.nemo.asr.audio import SAMPLERATE
from reazonspeech.nemo.asr.decode import PAD_SECONDS, SECONDS_PER_STEP, SUBWORDS_PER_SEGMENTS, PHONEMIC_BREAK, TOKEN_PUNC
from reazonspeech.nemo.asr.decode import find_end_of_segment_by_step, decode_hypothesis_to_subword_info
from reazonspeech.nemo.asr.interface import SubwordInfo, SegmentInfo, PreciseSubword, PreciseSegment
from reazonspeech.nemo.asr.writer import SRTWriter, ASSWriter, TextWriter, TSVWriter, VTTWriter

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

def assign_vad_ownership_keep_windows(vad_segments, keep_silence_sample, total_duration_sample):
    """
    输入 vad_segments: list[dict] 每个 dict 包含: start_sample/end_sample/keep_*
    输出增加 own_keep_start_sample/own_keep_end_sample
    """
    n = len(vad_segments)
    own_keep_start = [None] * n
    own_keep_end = [None] * n

    # 第一段：左侧最多扩 keep_silence_sample
    own_keep_start[0] = max(0, vad_segments[0]["start_sample"] - keep_silence_sample)

    for i in range(n - 1):
        e_i = vad_segments[i]["end_sample"]
        s_j = vad_segments[i + 1]["start_sample"]
        gap = s_j - e_i

        if gap < keep_silence_sample:
            # gap 全分配给后一段：边界取 e_i
            own_keep_end[i] = e_i
            own_keep_start[i + 1] = e_i
        else:
            # 首选中点
            mid = (e_i + s_j) // 2
            left_max = e_i + keep_silence_sample
            right_min = s_j - keep_silence_sample

            if right_min <= mid  <= left_max:
                # gap <= 2*keep_silence_sample：中点可行
                own_keep_end[i] = mid
                own_keep_start[i + 1] = mid
            else:
                # gap > 2*keep_silence_sample：各扩 keep_silence_sample，中间留空
                own_keep_end[i] = left_max
                own_keep_start[i + 1] = right_min

    # 最后一段：右侧最多扩 keep_silence_sample
    own_keep_end[-1] = min(total_duration_sample, vad_segments[-1]["end_sample"] + keep_silence_sample)

    # 补齐字段
    for i in range(n):
        vad_segments[i]["own_keep_start_sample"] = own_keep_start[i]
        vad_segments[i]["own_keep_end_sample"] = own_keep_end[i]

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
        result = max(1, torch.cuda.mem_get_info(device)[0] * 0.7 // per_sample // 2)

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

def create_precise_segments_from_subwords(raw_subwords, vad_chunk_end_samples, total_audio_samples, no_remove_punc):
    vad_end = list(vad_chunk_end_samples)
    vad_cursor = 0

    durations_sample = []

    # 计算自然结束时间，填充 vad_limit_sample / end_sample（用 next_start + clamp）
    for i, sw in enumerate(raw_subwords):
        if i < (len(raw_subwords) - 1):
            next_start = raw_subwords[i + 1].start_sample
        else:
            # 最后一个子词兜底
            next_start = sw.start_sample + STEP_SAMPLES * 2

        # 计算 VAD 限制 (VAD Limit)
        # 移动游标找到当前子词所属的 VAD 块
        # 利用短路逻辑：如果 vad_chunk_end_times_samples 为空，循环不会执行
        while vad_cursor < len(vad_end) and sw.start_sample >= vad_end[vad_cursor]:
            vad_cursor += 1

        if vad_cursor < len(vad_end):
            current_limit = vad_end[vad_cursor]
        else:
            # 最后一段没有vad边界，或者 no_chunk 模式下 vad_chunk_end_times_samples 列表为空，用整个音频的总时长作为边界
            current_limit = total_audio_samples

        sw.end_sample = min(next_start, current_limit)
        sw.vad_limit_sample = current_limit

        durations_sample.append(sw.end_sample - sw.start_sample)

    all_segments: list[SegmentInfo] = []
    all_subwords: list[SubwordInfo] = raw_subwords

    # 分组（按 vad_limit_sample 不变的连续段）+ find_end_of_segment_by_step
    start = 0
    while start < len(all_subwords):
        current_limit = all_subwords[start].vad_limit_sample
        # 直接在 all_subwords 中向后扫描，找到 vad_limit 变化的索引
        next_group_start_idx = start
        while next_group_start_idx < len(all_subwords) and all_subwords[next_group_start_idx].vad_limit_sample == current_limit:
            next_group_start_idx += 1

        # 确定当前 VAD 块内的搜索边界
        end_idx = min(
            next_group_start_idx - 1,
            find_end_of_segment_by_step(all_subwords, start, (int(PHONEMIC_BREAK * 1000) + STEP_MS - 1) // STEP_MS)
            )

        # 动态停顿阈值
        pause_split_indices = []
        # 基于语速/停顿的边界补充，只有当有足够多的数据点（例如超过20个子词间隔）时，才计算并启用阈值
        if end_idx - start + 1 > SUBWORDS_PER_SEGMENTS * 1.5:
            pause_threshold = np.percentile(durations_sample[start:end_idx], 90)
            pause_split_indices = [
                start + k for k, d in enumerate(durations_sample[start:end_idx])
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
            logger.debug(
                f"在 {fmt_time(samp_to_ms_floor(int(all_subwords[start].start_sample)) / 1000 + 1e-6)} "
                f"开始的长片段：“{''.join(preview_parts)}”内找到显著停顿点"
            )

        # 将片段的起始点和最终的结束点加入，形成完整的切分区间
        # start - 1 表示前一个片段的结束索引，作为新片段的基准
        # 为了让循环 for i in range(len(all_split_points) - 1): 能够从 start 索引开始处理第一个子片段
        all_split_points = [start - 1] + pause_split_indices + [end_idx]

        for i in range(len(all_split_points) - 1):
            # 提取当前片段的子词
            current_subwords = all_subwords[all_split_points[i] + 1 : all_split_points[i + 1] + 1]
            # 预先获取时间边界和文本，用于判断和日志
            curr_start = current_subwords[0].start_sample
            curr_end = current_subwords[-1].end_sample
            curr_limit = current_subwords[0].vad_limit_sample

            # 判断是否为纯标点/空格片段
            if all((s.token in TOKEN_PUNC or not s.token.strip()) for s in current_subwords):
                # === 分支 A：纯标点片段 ===
                curr_text_raw = "".join(s.token for s in current_subwords)
                
                if all_segments:
                    prev_seg = all_segments[-1]
                    # 此时 prev_seg 还没更新，curr_text_raw 是即将被合并（或丢弃）的标点
                    logger.debug((
                        f"{fmt_time(samp_to_ms_floor(int(prev_seg.start_sample)) / 1000 + 1e-6)} --> "
                        f"{fmt_time(samp_to_ms_ceil(int(prev_seg.end_sample)) / 1000 + 1e-6)}：{prev_seg.text}"
                        f" 和 "
                        f"{fmt_time(samp_to_ms_floor(int(curr_start)) / 1000 + 1e-6)} --> "
                        f"{fmt_time(samp_to_ms_ceil(int(curr_end)) / 1000 + 1e-6)}：{curr_text_raw}"
                        f" 已合并"
                    ))
    
                    # 无论是否保留标点，时间与vad边界都要吸收
                    prev_seg.end_sample = curr_end
                    prev_seg.vad_limit_sample = curr_limit

                    # 将上一段最后一个子词的结束时间也拉长
                    if no_remove_punc:
                        # 如果保留标点，合并文本和实体
                        prev_seg.subwords[-1].end_sample = curr_start
                        prev_seg.text += curr_text_raw
                        prev_seg.subwords.extend(current_subwords)
                    else:
                        prev_seg.subwords[-1].end_sample = curr_end
            else:
                # === 分支 B：含正文的片段 ===
                # 如果需要剔除句末标点，且最后一个子词是标点
                if not no_remove_punc and current_subwords[-1].token in TOKEN_PUNC:
                    removed_punc = current_subwords.pop()
                    logger.debug(
                        f"已剔除 {fmt_time(samp_to_ms_floor(int(removed_punc.start_sample)) / 1000 + 1e-6)} --> "
                        f"{fmt_time(samp_to_ms_ceil(int(removed_punc.end_sample)) / 1000 + 1e-6)}：{removed_punc.token}"
                    )
                    # 因为不是全标点句，pop 后列表一定不为空
                    # 填补时间空缺
                    current_subwords[-1].end_sample = removed_punc.end_sample
                    curr_end = removed_punc.end_sample
                text = "".join(s.token for s in current_subwords)

                all_segments.append(SegmentInfo(
                    start_sample=curr_start,
                    end_sample=curr_end,
                    text=text,
                    subwords=current_subwords,
                    chunk_index=current_subwords[0].chunk_index,
                    vad_limit_sample=curr_limit,
                ))

                if len(text) > SUBWORDS_PER_SEGMENTS * 2:
                    logger.warn(
                        f"{fmt_time(samp_to_ms_floor(int(curr_start)) / 1000 + 1e-6)} --> "
                        f"{fmt_time(samp_to_ms_ceil(int(curr_end)) / 1000 + 1e-6)} "
                        f"段字数超过 {SUBWORDS_PER_SEGMENTS * 2}，制作字幕时有可能溢出屏幕"
                    )

        # 更新外层循环游标
        start = end_idx + 1

    return all_segments, all_subwords

def enforce_monotonic_start_inplace(subs, total_audio_samples):
    """
    强制 start_sample 严格单调递增，避免排序/分段逻辑被相同或倒序时间戳破坏。
    （你后面会用 next_start 来推 end_sample，所以 monotonic 很重要）
    """
    if not subs:
        return
    prev = int(max(0, min(int(subs[0].start_sample), int(total_audio_samples))))
    subs[0].start_sample = prev
    for i in range(1, len(subs)):
        cur = int(subs[i].start_sample)
        cur = max(cur, prev + 1)
        if cur > total_audio_samples:
            cur = int(total_audio_samples)
        subs[i].start_sample = cur
        prev = cur

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

def fmt_time(seconds):
    h = int(seconds / 3600)
    m = int(seconds / 60) % 60
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return "%02i:%02i:%02i,%03i" % (h, m, s, ms)

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
    min_speech_duration_samples, #最小语音段时长
    min_silence_duration_samples #多长静音视为真正间隔
):
    """
    返回：
      speeches: List[[start_samples, end_samples], ...]
      speech_probs: np.ndarray                    # per-frame prob
      frame_hop_samples # 每帧大致跨度（sample）
      frame_times_sample # 每帧对应的“帧起点 sample”，与 speech_probs 对齐
    使用 segmentation-3.0 ONNX + ≤10 秒滑窗 + 双阈值后处理，
    将一段 waveform 转换为若干语音时间段
    """
    # 通过 ≤10 秒滑窗，拼出整段 speech_probs 和 frame_times_sample
    # 固定全局帧步长：10ms
    frame_hop_samples = ms_to_samp_round(10)

    total = waveform.shape[1]
    n_global = (total + frame_hop_samples - 1) // frame_hop_samples + 1
    accum = np.zeros(n_global, dtype=np.float32)
    weight = np.zeros(n_global, dtype=np.float32)

    start_sample = 0 

    while start_sample < total:
        end_sample = min(start_sample + 10 * SAMPLERATE, total)
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
        probs_full = torch.sigmoid(torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]).numpy()

        # 当前窗口的实际样本数
        window_samples = end_sample - start_sample

        # 生成 frame_times_sample：每帧对应“帧起点”的绝对 sample（np.int64）
        # 纯整数 floor 映射：保证落在 [start_sample, end_sample) 内
        times_full_sample = start_sample + (np.arange(len(probs_full), dtype=np.int64) * int(window_samples)) // len(probs_full)

        # Hamming 权重
        w = np.hamming(len(probs_full)).astype(np.float32)

        # warm-up 0.5s，对短窗做保护，warm-up trim：非首窗丢掉左 warm-up，非尾窗丢掉右 warm-up
        warm_up = min(0.5 * SAMPLERATE, window_samples // 4)  # 防短窗
        valid = np.ones(len(probs_full), dtype=bool)
        if not start_sample == 0:
            valid &= (times_full_sample >= start_sample + warm_up)
        if not end_sample == total:
            valid &= (times_full_sample <= end_sample - warm_up)

        # 映射到全局 10ms 网格做 overlap-add
        gidx = np.clip((times_full_sample + frame_hop_samples // 2) // frame_hop_samples, 0, n_global - 1)

        accum[gidx[valid]] += probs_full[valid] * w[valid]
        weight[gidx[valid]] += w[valid]

        if end_sample >= total:
            break

        # 10s 滑窗、8s 步长（2s overlap）
        start_sample += int(8 * SAMPLERATE)

    speech_probs = accum / np.maximum(weight, 1e-6)

    frame_times_sample = (np.arange(n_global, dtype=np.int64) * frame_hop_samples)
    # 去掉最后可能超出 total 的点
    keep = frame_times_sample < total
    speech_probs = speech_probs[keep]
    frame_times_sample = frame_times_sample[keep]

    # 双阈值 + 填平短静音 + 丢弃短语音 + 直接生成秒级区间
    # 最小语音段帧数（小于这个长度的语音段会被丢弃）
    min_speech_frames = (int(min_speech_duration_samples) + frame_hop_samples - 1) // frame_hop_samples
    # 最小静音段帧数（短于这个长度的静音视为“短静音”，会被填平）
    min_silence_frames = (int(min_silence_duration_samples) + frame_hop_samples - 1) // frame_hop_samples
    speeches = []    # 存 sample 结果 [[start_sample, end_sample], ...]
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
                        start_sample = int(frame_times_sample[seg_start_f])
                        end_sample = int(frame_times_sample[seg_end_f])
                        if end_sample > start_sample:
                            speeches.append([start_sample, end_sample])
                    # 重置状态，等待下一段
                    in_speech = False
                    seg_start_f = 0
                    silence_start_f = None
            else:
                # 概率又回到了 neg_threshold 以上，静音计数作废
                silence_start_f = None

    # 处理结尾残留：文件结束时仍在语音段中
    if in_speech and len(speech_probs) - seg_start_f >= min_speech_frames:
            start_sample = int(frame_times_sample[seg_start_f])
            end_sample = waveform.shape[1]
            if end_sample > start_sample:
                speeches.append([start_sample, end_sample])

    #  speeches: List[[start_sample, end_sample], ...]
    #  speech_probs: np.ndarray
    #  frame_hop_samples: int（用来把“样本时长阈值”换算成“帧数阈值”）
    #  frame_times_sample: np.ndarray(np.int64)，每帧对应的“帧起点 sample”
    return speeches, speech_probs, frame_hop_samples, frame_times_sample

_VAD_ONNX_SESSION = None
_VAD_ONNX_MODEL_LOAD_COST = 0
_VAD_ONNX_SESSION_LOCK = threading.Lock()
def get_vad_onnx_session(model_path):
    """
    获取 Pyannote-segmentation-3.0 的 onnxruntime.InferenceSession 单例，复用已加载 Session
    """
    global _VAD_ONNX_SESSION, _VAD_ONNX_MODEL_LOAD_COST

    with _VAD_ONNX_SESSION_LOCK:
        if _VAD_ONNX_SESSION is None:

            vad_model_load_start = time.perf_counter()
            logger.info("【VAD】正在从本地路径加载 Pyannote-segmentation-3.0 模型……")
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.perf_counter()
            logger.info(f"【VAD】模型加载完成")
    
            _VAD_ONNX_SESSION = session
            _VAD_ONNX_MODEL_LOAD_COST = vad_model_load_end - vad_model_load_start

        return _VAD_ONNX_SESSION

def global_smart_segmenter(
    start_sample, end_sample, speech_probs,
    energy_array, frame_times_sample, max_duration_sample,
    vad_threshold, zcr_threshold, zcr_array):
    """
    入口函数：分析一个长 VAD 段，返回切割好的子段列表。

    - 如果大于 max_duration_sample：
        * 在低 VAD 概率的局部极小值处生成候选点，并计算 cost
        * 从左到右贪心切分（必要时回退到硬切）
        * 相邻段推理范围重叠 2*SAMPLERATE，但提交窗口在 cut 处分割，保证不重复提交

    返回：List[dict]
    """
    def _frame_cost(frame_idx):
        """计算单帧作为切分点的代价（代价越低越适合切分）"""
        # 基础 VAD 概率成本
        if speech_probs[frame_idx] > vad_threshold:
            return 100  # 惩罚高概率点，强硬约束：确信是语音的地方绝不切

        # --- 动态抗噪过零率 (Robust ZCR) ---
        # 逻辑：(过零) AND (穿越点的幅度足够大，不仅是微小抖动)
        # 标准做法：check max(abs(x[n]), abs(x[n+1])) > th
        zcr_cost = 0
        if zcr_array is not None and zcr_array[frame_idx] > zcr_threshold:
            zcr_cost = zcr_array[frame_idx] * 50

        # 局部平滑度 (斜率)
        slope_cost = abs(speech_probs[frame_idx + 1] - speech_probs[frame_idx - 1]) * 2

        return speech_probs[frame_idx] + min(energy_array[frame_idx], 0.5) * 10 + slope_cost + zcr_cost

    # 时长不超长直接返回
    if (end_sample - start_sample) <= max_duration_sample:
        return [{
            "start_sample": start_sample,
            "end_sample": end_sample,
            "keep_start_sample": start_sample,
            "keep_end_sample": end_sample,
        }]

    # 生成候选点（筛选 VAD 概率 < max(0.15, vad_threshold - 0.15) 的局部低点）
    candidates = []
    # 边界保护：避免切在段首/段尾靠得太近
    # 用 frame_times_samples 反查 frame 索引
    search_start = int(np.searchsorted(frame_times_sample, start_sample + SAMPLERATE, side="left"))
    search_end = int(np.searchsorted(frame_times_sample, end_sample - SAMPLERATE, side="right"))

    # 提取搜索区间的概率
    # search_end > search_start + 2 是为了确保区间内至少有 3 帧数据
    # 寻找局部极小值需要比较：当前帧(curr) <= 前一帧(prev) 且 当前帧(curr) <= 后一帧(next)
    # 这对应了下方的切片逻辑：[1:-1] (curr), [:-2] (prev), [2:] (next)
    # 如果总帧数少于 3 (即 end - start <= 2)，切片后的维度将无法对齐或为空，因此跳过
    if search_end > search_start + 2:
        # 利用 numpy 寻找局部极小值且概率 < max(0.15, vad_threshold - 0.15) 的点
        # 逻辑：当前点 <= 前一点 AND 当前点 <= 后一点 AND 当前点 < max(0.15, vad_threshold - 0.15)
        p_curr = speech_probs[search_start + 1:search_end - 1] # 对应原数组的索引 i
        p_prev = speech_probs[search_start:search_end - 2] # 对应 i-1
        p_next = speech_probs[search_start + 2:search_end] # 对应 i+1
        # 生成布尔掩码
        mask = (p_curr < max(0.15, vad_threshold - 0.15)) & (p_curr <= p_prev) & (p_curr <= p_next)
        # 获取满足条件的相对索引
        # 还原绝对索引：+1 是因为 p_curr 从切片的第1个元素开始，+search_start 是切片偏移
        # 遍历筛选出的索引计算成本
        for idx in (np.where(mask)[0] + 1 + search_start):
            candidates.append({
                "frame": int(idx),
                "time_sample": int(frame_times_sample[idx]),
                "cost": _frame_cost(int(idx)),
            })

    segments = []
    curr_start = start_sample
    prev_cut = None
    # 初始化索引指针，指向 candidates 列表的开头
    cand_idx = 0
    while True:
        # 剩余长度已经不超过上限，直接收尾
        if (end_sample - curr_start) <= max_duration_sample:
            segments.append({
                "start_sample": curr_start,
                "end_sample": end_sample,
                "keep_start_sample": start_sample if prev_cut is None else prev_cut,
                "keep_end_sample": end_sample,
            })
            break

        # 向前移动指针：跳过窗口左侧的旧候选点（切点必须 > curr_start + OVERLAP_MS，才能保证 next_start > curr_start）
        while cand_idx < len(candidates) and candidates[cand_idx]["time_sample"] <= (curr_start + SAMPLERATE):
            cand_idx += 1

        # 收集当前窗口内的候选点
        cands = []
        # 使用一个临时游标 temp_idx 向后扫描，直到超出窗口右侧
        # 不修改 cand_idx，因为下一个窗口可能还会用到当前的起始点，如果有重叠或回退逻辑
        temp_idx = cand_idx
        while temp_idx < len(candidates):
            c = candidates[temp_idx]
            if c["time_sample"] > (curr_start + max_duration_sample - SAMPLERATE):
                # 因为列表是有序的，一旦超过右边界，后面的都无效
                break
            cands.append(c)
            temp_idx += 1

        if cands:
            # 直接选 cost 最小的
            cut_t = min(cands, key=lambda c: c["cost"])["time_sample"]
        else:
            # 当前窗口里没有任何合适候选，只能硬切
            cut_t = curr_start + max_duration_sample - SAMPLERATE

        segments.append({
            "start_sample": curr_start,
            "end_sample": cut_t + SAMPLERATE,
            "keep_start_sample": start_sample if prev_cut is None else prev_cut,
            "keep_end_sample": cut_t,
        })

        prev_cut = cut_t
        curr_start = cut_t - SAMPLERATE

    return segments

def load_audio_ffmpeg(file, audio_filter, limiter_filter):
    """
    使用 ffmpeg 将任何输入格式转为:
      - 单声道
      - 采样率 SAMPLERATE
      - 输出端量化为 16-bit PCM (s16le)，再在 Python 中转回 float32 [-1, 1)
      根据 audio_filter 和 limiter_filter 决定是否附加 -af 滤镜链，limiter_filter 会追加 alimiter
    然后转为 PyTorch Tensor 和总采样点数（samples）
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
        cmd += ["-af", _append_tail(
            limiter_filter,
            "alimiter=limit=0.98:level=disabled:attack=5:release=50:latency=1" + "," + tail_s16)]
    
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

    return torch.from_numpy(audio_np).unsqueeze(0), audio_np.size # [1, T], samples

def merge_short_segments_adaptive(segments, max_duration_sample):
    """
    把短于 1000ms 的片段合并到相邻较短的片段中（但内部坐标为 sample），优先合并更短的一侧，并确保合并后的总时长不超过 max_duration_sample
    仅接受字典列表：
      {"start_sample","end_sample","keep_start_sample","keep_end_sample", ...}
    """
    if not segments:
        return []

    working_segments = [dict(s) for s in segments]
    result = []

    i = 0
    while i < len(working_segments):
        seg = working_segments[i]
        seg_duration_sample = seg["end_sample"] - seg["start_sample"]
        if seg_duration_sample <= SAMPLERATE: # 不大于 1 秒才合并
            # 获取前后邻居
            prev_seg = result[-1] if result else None
            next_seg = working_segments[i + 1] if i + 1 < len(working_segments) else None

            dur_prev = (prev_seg["end_sample"] - prev_seg["start_sample"]) if prev_seg else math.inf
            dur_next = (next_seg["end_sample"] - next_seg["start_sample"]) if next_seg else math.inf

            # 向左合的新时长 = 当前结束 - 前段开始
            dur_if_left = (seg["end_sample"] - prev_seg["start_sample"]) if prev_seg else math.inf
            # 向右合的新时长 = 后段结束 - 当前开始
            dur_if_right = (next_seg["end_sample"] - seg["start_sample"]) if next_seg else math.inf

            target_side = None
            # 两边都合法，哪边合并后更短合哪边
            if dur_if_left <= max_duration_sample and dur_if_right <= max_duration_sample:
                target_side = "left" if dur_if_left <= dur_if_right else "right"
            # 只有左边合法（意味着右边超长或不存在）
            elif dur_if_left <= max_duration_sample:
                target_side = "left"
            # 只有右边合法（意味着左边超长或不存在）
            elif dur_if_right <= max_duration_sample:
                target_side = "right"

            if target_side == "left":
                logger.debug(
                    f"【VAD】片段（{samp_to_ms_floor(seg_duration_sample) / 1000:.2f} 秒）"
                    f"{fmt_time(samp_to_ms_floor(seg['start_sample']) / 1000)} --> "
                    f"{fmt_time(samp_to_ms_floor(seg['end_sample']) / 1000)}，"
                    f"向左合并: 前段（{samp_to_ms_floor(dur_prev) / 1000:.2f} 秒）"
                    f"{fmt_time(samp_to_ms_floor(prev_seg['start_sample']) / 1000)} -->"
                    f"{fmt_time(samp_to_ms_floor(prev_seg['end_sample']) / 1000)} "
                    f"延长至 {samp_to_ms_floor(dur_prev + seg_duration_sample) / 1000:.2f} 秒"
                )
                prev_seg["end_sample"] = seg["end_sample"]
                prev_seg["keep_end_sample"] = seg["keep_end_sample"]
                i += 1
                continue

            elif target_side == "right":
                logger.debug(
                    f"【VAD】片段（{samp_to_ms_floor(seg_duration_sample) / 1000:.2f} 秒）"
                    f"{fmt_time(samp_to_ms_floor(seg['start_sample']) / 1000)} --> "
                    f"{fmt_time(samp_to_ms_floor(seg['end_sample']) / 1000)}，"
                    f"向右合并: 后段（{samp_to_ms_floor(dur_next) / 1000:.2f} 秒）"
                    f"{fmt_time(samp_to_ms_floor(next_seg['start_sample']) / 1000)} -->"
                    f"{fmt_time(samp_to_ms_floor(next_seg['end_sample']) / 1000)} "
                    f"延长至 {samp_to_ms_floor(seg_duration_sample + dur_next) / 1000:.2f} 秒"
                )
                next_seg["start_sample"] = seg["start_sample"]
                next_seg["keep_start_sample"] = seg["keep_start_sample"]
                i += 1
                continue

        result.append(seg)
        i += 1

    return result

# === 纯整数：ms -> sample（四舍五入） ===
def ms_to_samp_round(milliseconds):
    return (milliseconds * SAMPLERATE + 500) // 1000

def open_folder(path):
    """
    根据不同操作系统，使用默认的文件浏览器打开指定路径的文件夹
    如果 path 是文件，则打开其所在文件夹并选中（高亮）该文件（仅限 Win/Mac）
    如果 path 是文件夹，则直接打开该文件夹
    """
    path = Path(path).resolve()
    if not path.exists():
        logger.warn(f"路径 '{path}' 不存在，无法自动打开")
        return

    opened_target = None # 记录“实际传给系统打开器”的对象

    try:
        # returncode == 0 通常表示命令被成功接收执行
        if sys.platform == "win32":
            if path.is_file():
                opened_target = path
                # Windows 使用 explorer /select,"文件路径" 来高亮文件
                cp = subprocess.run(["explorer", "/select,", str(path)], check=False)
                ok = (cp.returncode == 0)
            else:
                opened_target = path
                os.startfile(path)  # Windows 专用“用默认关联程序打开”，无 returncode
                ok = True

        elif sys.platform == "darwin":
            if path.is_file():
                opened_target = path
                # macOS 使用 open -R "文件路径" 来在 Finder 中揭示文件
                cp = subprocess.run(["open", "-R", str(path)], check=False)
            else:
                opened_target = path
                cp = subprocess.run(["open", str(path)], check=False)
            ok = (cp.returncode == 0)

        else:
            #如果传入的是文件：打开父目录；如果传入的是目录：直接打开目录
            opened_target = path.parent if path.is_file() else path
            cp = subprocess.run(["xdg-open", str(opened_target)], check=False)
            ok = (cp.returncode == 0)

        if ok:
            logger.info(f"已打开：{opened_target}")
        else:
            logger.warn(f"尝试打开目录失败（returncode={cp.returncode}），请手动访问：{opened_target}")

    except Exception as e:
        logger.warn(f"尝试自动打开目录失败：{e}，请手动访问：{opened_target or path}")

def prepare_acoustic_features(waveform, speech_probs, use_zcr):
    """计算并对齐声学特征（能量、ZCR），供智能切分使用"""

    # 计算帧长
    frame_length = max(1, (waveform.shape[1] + len(speech_probs) - 1) // len(speech_probs))
    
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
            # 需要 (N, C, L)
            (((waveform_centered[:, :-1] * waveform_centered[:, 1:]) < 0) & (is_loud[:, :-1] | is_loud[:,1:])).float().unsqueeze(0), 
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

def transform_segment_info(seg):
    return PreciseSegment(
        start_seconds=samp_to_ms_floor(seg.start_sample) / 1000 + 1e-6, # 安全缓冲
        end_seconds=samp_to_ms_ceil(seg.end_sample) / 1000 + 1e-6, # 安全缓冲
        text=seg.text,
        subwords=[transform_subword_info(sw) for sw in seg.subwords],
    )

def transform_subword_info(sw):
    return PreciseSubword(
        seconds=samp_to_ms_floor(sw.start_sample) / 1000 + 1e-6, # 安全缓冲
        end_seconds=samp_to_ms_ceil(sw.end_sample) / 1000 + 1e-6, # 安全缓冲
        token_id=sw.token_id,
        token=sw.token,
    )

def refine_tail_timestamp(
    last_token_start_sample, # 该段最后一个子词的开始时间
    rough_end_sample, # 原段结束时间
    speech_probs, # VAD 概率序列
    frame_times_sample, # np.ndarray[np.int64]，与 speech_probs 对齐
    frame_hop_samples, # 每帧大致跨度
    max_end_sample, # 该段所属 VAD 大块的硬上限
    min_silence_samples, # 判定静音的最小时长
    lookahead_samples, # 发现静音后再向后看一点，避免马上回到语音（clamp 在区间内）
    safety_margin_samples, # 安全边距，避免切在突变点
    min_tail_keep_samples, # 至少保留最后 token 后的这点时长，避免切得太硬
    percentile, # 自适应阈值，值越大，越容易将高概率有语音区域判为静音
    offset, # 在自适应阈值的基础上增加的固定偏移量，值越大，越容易将高概率区域有语音区域判为静音
    energy_array, # np.ndarray[float]，与 speech_probs 对齐
    energy_percentile,
    energy_offset,
    use_zcr,
    zcr_threshold,
    zcr_array,
    zcr_high_ratio, # 高ZCR帧的占比阈值
    kernel, # np.ndarray[float] 平滑核
):
    """
    在 [last_token_start_sample, max_end_sample] 范围内做所有自适应统计（percentile 等），
    正常情况下只在这个范围内找切点
    如果在这段内出现 “概率突然掉、但能量或 ZCR 仍然偏高” 的可疑静音，
    则允许在额外的 1 秒尾巴内继续按同一阈值寻找真正静音切点。
    """
    # 仅在“最后子词之后”的尾窗内搜索,用 frame_times_sample 做对齐索引
    start_idx = int(np.searchsorted(frame_times_sample, int(last_token_start_sample), side="left"))
    end_idx = int(np.searchsorted(frame_times_sample, int(max_end_sample), side="right"))
    # 允许在 [max_end_sample, max_end_sample + 1s] 内额外搜索，但不用于统计阈值（可关闭）
    search_end_idx = int(np.searchsorted(frame_times_sample, int(max_end_sample + SAMPLERATE), side="right"))

    # 至少要有 4 帧才能进行有效的统计学分析
    if start_idx >= len(speech_probs) or end_idx - start_idx <= 4:
        return min(rough_end_sample, max_end_sample)

    # 平滑（只在 [last_token_start_sample, max_end_sample] 内统计阈值）
    p_smooth = np.convolve(speech_probs[start_idx:end_idx], kernel, mode="same")
    e_smooth = np.convolve(energy_array[start_idx:end_idx], kernel, mode="same")

    # 局部自适应阈值percentile + offset
    # 限制阈值范围，防止极端情况导致逻辑失效
    # 0.10 保证底噪容忍度，0.95 保证不会因为全 1.0 的概率导致无法切割
    dyn_tau = np.clip(np.percentile(p_smooth, percentile) + offset, 0.10, 0.95)
    # 取尾段能量的低分位数，找“相对安静”的能量水平
    dyn_e_tau = max(np.percentile(e_smooth, energy_percentile) + energy_offset, 1e-6)

    # 额外统计：用于自适应“突降”阈值与滞回
    spread = max(float(np.percentile(p_smooth, 90)) - float(np.percentile(p_smooth, 10)), 1e-6)  

    # 跨 dyn_tau 的“像语音/像静音”两阈值（避免只看单点差值）
    hi = min(0.99, float(dyn_tau) + max(0.08, 0.25 * spread))   # 明确语音
    lo = max(0.01, float(dyn_tau) - max(0.03, 0.10 * spread))  # 明确静音

    # 自适应“突降幅度”基准阈值（方案A）
    drop_th_base = float(np.clip(0.6 * spread, 0.15, 0.45))

    # samples -> frames（纯整数 ceil）
    min_silence_frames = max(1, (int(min_silence_samples) + frame_hop_samples - 1) // frame_hop_samples)

    def _sudden_drop_hangover_frames(i):
        """
        若检测到“跨 hi/lo 的突降”且声学仍活跃（能量/ZCR高），返回建议的 hangover 帧数；
        否则返回 0。

        - drop 阈值随 spread 自适应
        - “跨 dyn_tau”三条件：prev>hi、curr<lo、drop>=drop_th
        - 声学强度越高，drop_th 越放宽、hangover 越长
        - WebRTC hangover 意图：触发后延迟释放，避免尾音/清辅音被切
        """
        sudden_window_frames = max(
            4,
            int(
                (
                    min(
                        ms_to_samp_round(250),
                        max(ms_to_samp_round(80), int(min_silence_samples) // 2),
                    )
                    + frame_hop_samples
                    - 1
                )
                // frame_hop_samples
            ),
        )
        prev_end = i
        prev_start = max(0, prev_end - sudden_window_frames)
        if prev_end <= prev_start:
            return 0

        prev_max_p = float(p_smooth[prev_start:prev_end].max())
        curr_mean_p = float(p_smooth[i : i + min_silence_frames].mean())

        # 必须跨越“明显语音 -> 明显静音”（Schmitt trigger）
        if prev_max_p < hi:
            return 0
        if curr_mean_p > lo:
            return 0

        # 声学活跃度
        high_energy = (
            float(np.mean(e_smooth[prev_start:prev_end])) > float(dyn_e_tau)
            or float(np.mean(e_smooth[i : i + min_silence_frames])) > float(dyn_e_tau)
        )

        high_zcr = False
        if use_zcr:
            z_start = start_idx + i
            z_end = z_start + min_silence_frames
            high_zcr = (
                float(np.mean(zcr_array[z_start:z_end] > zcr_threshold)) > float(zcr_high_ratio)
            )

        if not (high_energy or high_zcr):
            return 0

        acoustic_strength = (1 if high_energy else 0) + (1 if high_zcr else 0)

        # 自适应突降幅度阈值，并按声学证据放宽
        drop_th = float(np.clip(drop_th_base - 0.05 * acoustic_strength, 0.12, 0.45))
        if (prev_max_p - curr_mean_p) < drop_th:
            return 0

        # 声学证据越强，保护越久
        hangover_ms = int(np.clip(200 + 150 * acoustic_strength, 120, 600))
        hangover_frames = max(
            1,
            (ms_to_samp_round(hangover_ms) + int(frame_hop_samples) - 1) // int(frame_hop_samples),
        )
        return int(hangover_frames)

    def _is_stable_silence(i, p_arr, e_arr, global_base_idx):
        """
        判断当前片段是否为合格的静音切点
        i: 在当前片段 p_smooth 中的相对索引
        p_arr: 当前分析的概率片段 (p_smooth 或 extra_p_smooth)
        e_arr: 当前分析的能量片段
        global_base_idx: p_arr 开头对应在原始大数组中的绝对索引 (用于取 ZCR)
        """
        # 概率是否持续低于自适应阈值
        if not np.all(p_arr[i : i + min_silence_frames] < dyn_tau):
            return False

        # 能量是否也处于本段“相对静音”区间
        if not (np.mean(e_arr[i : i + min_silence_frames]) < dyn_e_tau):
            return False

        # ZCR 保护（清辅音）
        if use_zcr:
            z_start = int(global_base_idx + i)
            z_end = z_start + int(min_silence_frames)
            # 获取窗口内的 ZCR 数据
            # 计算窗口内超过 ZCR 阈值的帧的比例
            # np.mean(布尔数组) 相当于计算 True 的百分比
            high_zcr_ratio = np.mean(zcr_array[z_start:z_end] > zcr_threshold)
            # 只有当高 ZCR 帧的比例超过设定值（如 0.3）时，才触发保护
            if high_zcr_ratio > zcr_high_ratio:
                logger.debug(
                    f"【ZCR】疑似静音段 "
                    f"{fmt_time(samp_to_ms_floor(int(z_start) * int(frame_hop_samples)) / 1000)} --> "
                    f"{fmt_time(samp_to_ms_ceil(int(z_end) * int(frame_hop_samples)) / 1000)} "
                    f"内高频帧占比 {high_zcr_ratio:.2f} > {zcr_high_ratio}，视为清辅音，跳过切分"
                )
                return False

        # 滞回检查 —— 确认后面不会马上回到高概率语音（clamp 在区间内）
        lookahead_frames = max(1, (int(lookahead_samples) + frame_hop_samples - 1) // frame_hop_samples)
        la_start = i + min_silence_frames
        la_end = la_start + lookahead_frames
        if la_start < len(p_arr):
            if la_end > len(p_arr):
                la_end = len(p_arr)
            if la_end > la_start:
                if np.any(p_arr[la_start:la_end] >= min(0.98, hi)):
                    return False

        return True

    def _calc_refined_sample(idx_offset, limit_sample):
        """计算最终时间戳（ms），应用安全边距和最大限制；idx_offset 为全局帧索引"""
        s = int(frame_times_sample[idx_offset]) + int(safety_margin_samples)

        # 至少保留最后 token 后 min_tail_keep
        if s < int(last_token_start_sample) + int(min_tail_keep_samples):
            s = int(last_token_start_sample) + int(min_tail_keep_samples)

        # clamp 到上界
        if s > int(limit_sample):
            s = int(limit_sample)

        return s

    # 标记：是否在分析区间内遇到过“概率突降 + 高能量/ZCR”的可疑静音
    allow_use_extra_tail = False

    # === 只在 [last_token_start_sample, max_end_sample] 内找 ===
    hangover_left = 0
    for i in range(0, len(p_smooth) - min_silence_frames + 1):
        # 先尝试触发“可疑突降”保护：返回 hangover 帧数
        h = _sudden_drop_hangover_frames(i)
        if h > 0:
            allow_use_extra_tail = True
            hangover_left = max(hangover_left, h)

        # hangover 期间：强制跳过切点判定（避免尾音/清辅音被切）
        if hangover_left > 0:
            hangover_left -= 1
            continue

        if _is_stable_silence(i, p_smooth, e_smooth, start_idx):
            return _calc_refined_sample(start_idx + i, int(max_end_sample))

    # === 可疑突降触发时，允许在 [max_end_sample, max_end_sample+overlap] 内继续找 ===
    if not allow_use_extra_tail:
        return min(rough_end_sample, max_end_sample)

    # 有可疑静音，可以利用已有阈值在多出的 1 秒内继续搜索
    # 只对额外 1s 做平滑，仍然使用之前在 [last_token_start_s, max_end_s] 内得到的 dyn_tau / dyn_e_tau。
    extra_p_smooth = np.convolve(speech_probs[end_idx:search_end_idx], kernel, mode="same")
    extra_e_smooth = np.convolve(energy_array[end_idx:search_end_idx], kernel, mode="same")

    for j in range(0, len(extra_p_smooth) - min_silence_frames + 1):
        if _is_stable_silence(j, extra_p_smooth, extra_e_smooth, end_idx):
            # 上限放宽到 max_end_s + OVERLAP_S
            return _calc_refined_sample(end_idx + j, int(max_end_sample + SAMPLERATE))

    # 兜底：没找到稳定静音，保留原结束时间
    return min(rough_end_sample, max_end_sample)

def save_tensor_as_wav(path, waveform):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes((np.clip(waveform.squeeze(0).detach().cpu().numpy(), -1, 1) * 32767).astype('<i2').tobytes())

# === 纯整数：sample -> ms（向下/向上） ===
def samp_to_ms_floor(sample):
    return (sample * 1000) // SAMPLERATE
def samp_to_ms_ceil(sample):
    return (sample * 1000 + SAMPLERATE - 1) // SAMPLERATE

def slice_waveform_sample(waveform, start_sample, end_sample):
    """
    高效 Tensor 切片与 Padding
    waveform: [1, T]
    半开区间 [start_sample, end_sample)
    """
    chunk = waveform[:, start_sample:end_sample]
    # 边界限制并切片，F.pad: (padding_left, padding_right)，pad 操作会产生副本
    return torch.nn.functional.pad(chunk, (PAD_SAMPLES, PAD_SAMPLES))

def transcribe_audio(model, audio, batch_size=1):
    """封装转录逻辑：调用模型 -> 解码 -> 返回子词列表"""
    return model.transcribe(audio=audio, batch_size=batch_size, return_hypotheses=True, verbose=False)

# === 把秒常量变成毫秒整数 ===
STEP_MS = int(SECONDS_PER_STEP * 1000 + 0.5)  # 0.08 -> 80
STEP_SAMPLES = ms_to_samp_round(STEP_MS)
PAD_SAMPLES = int(PAD_SECONDS * SAMPLERATE)
MAX_SPEECH_DURATION_SAMPLES = 18 * SAMPLERATE

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
        if not (0.05 < args.vad_threshold <= 1):
            parser.error(f"vad_threshold 必须在（0.05-1）范围内，当前值错误")
        if not (0.05 <= args.vad_end_threshold <= 1):
            parser.error(f"vad_end_threshold 必须在（0.05-1）范围内，当前值错误")
        if args.vad_end_threshold > args.vad_threshold:
            parser.error(
                f"vad_end_threshold 不能大于 vad_threshold"
            )
    
        if args.refine_tail:
            if not (0 <= args.tail_percentile <= 100):
                parser.error(f"tail_percentile 必须在（0-100）范围内，当前值错误")
            if not (0 <= args.tail_energy_percentile <= 100):
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

    # --- 推理参数转换 ---
    keep_silence_samples = ms_to_samp_round(args.keep_silence)
    min_speech_samples = ms_to_samp_round(args.min_speech_duration_ms)
    min_silence_samples = ms_to_samp_round(args.min_silence_duration_ms)

    # --- 执行核心的语音识别流程 ---
    recognition_start_time = 0
    recognition_end_time = 0
    original_decoding_cfg = None
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        logger.info(f"正在转换输入文件 '{input_path}'……")
        waveform, total_audio_samples = load_audio_ffmpeg(input_path, args.audio_filter, args.limiter_filter)
        logger.info("转换完成")

        # 使用单例模式获取模型，避免重复加载
        model = get_asr_model()

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

        vad_chunk_end_samples = []  # 用于存储 VAD 块的结束 sample
        
        # --- 批处理相关变量 ---
        batch_audio = [] # 待处理的音频数据队列
        batch_meta = [] # 存 {'index': chunk_index, 'offset': time_offset}

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

            for number in batch_meta:
                logger.info(f"正在识别语音块 {number['display_number']}")

            hit_oom = False
            # ---- 第一次尝试：用当前 batch_size 一次性跑完 ----
            try:
                with torch.inference_mode():
                    hyps = transcribe_audio(model, batch_audio, len(batch_audio))
    
            except RuntimeError as e:
                if _is_oom_error(e):
                    hit_oom = True
                    logger.warn("显存不足，当前批次将回退为逐条识别……")
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
                    hyps = transcribe_audio(model, batch_audio)

            # 处理结果
            for hyp, meta in zip(hyps, batch_meta):
                if not hyp:
                    continue
                # 解码
                decoded = decode_hypothesis_to_subword_info(model, hyp)
                plan_idx = int(meta["plan_index"])

                for token_id, token, step_index in decoded:
                    # raw_local<0 时 clamp 到 0
                    local_start_sample = max(0, step_index * STEP_SAMPLES - PAD_SAMPLES)
                    t_sample = meta["base_offset_sample"] + local_start_sample

                    # clamp 到全局范围
                    if t_sample < 0:
                        t_sample = 0
                    elif t_sample > total_audio_samples:
                        t_sample = total_audio_samples

                    subwords_by_plan[plan_idx].append(SubwordInfo(
                        token_id=token_id,
                        token=token,
                        start_sample=t_sample,
                        end_sample=t_sample + STEP_SAMPLES,   # 临时值，后面会用 next_start/vad_limit 重算
                        step_index=int(step_index),
                        chunk_index=int(meta["chunk_index"]), # 仍保留“来自哪个 VAD 大块”的信息
                        vad_limit_sample=0,                   # 后续 create_precise_segments_from_subwords 会覆盖
                    ))

            batch_audio.clear()
            batch_meta.clear()

        # --- 分支 A: 不分块 (No Chunk) ---
        if args.no_chunk:
            logger.info("未使用VAD，一次性处理整个文件……")
            
            planned_segments = [{
                "plan_index": 0,
                "chunk_index": 0,
                "display_number": 1,
                "audio_start_sample": 0,
                "audio_end_sample": total_audio_samples,
                "base_offset_sample": 0,
                "own_keep_start_sample": 0,
                "own_keep_end_sample": total_audio_samples,
                "keep_start_sample": 0,
                "keep_end_sample": total_audio_samples,
            }]

            full_chunk = slice_waveform_sample(waveform, 0, total_audio_samples)
            if args.debug and not api_mode:
                fd, temp_full_wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                save_tensor_as_wav(temp_full_wav_path, full_chunk)

        # --- 分支 B: 使用 VAD 分块 ---
        else:
            onnx_session = get_vad_onnx_session(local_onnx_model_path)

            # 运行 VAD
            speeches, speech_probs, frame_hop_samples, frame_times_sample = get_speech_timestamps_onnx(
                waveform,
                onnx_session,
                args.vad_threshold,
                args.vad_end_threshold,
                min_speech_samples,
                min_silence_samples
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

            vad_chunk_end_samples = [seg[1] for seg in speeches]

            # 声学特征准备
            if args.auto_zcr or args.refine_tail:
                speech_probs, energy_array, zcr_array = prepare_acoustic_features(waveform, speech_probs, args.zcr)
            # 对齐 frame_times_sample
            frame_times_sample = frame_times_sample[:len(speech_probs)]

            # ZCR 阈值校准 
            # 确定最终使用的 ZCR 阈值
            final_zcr_threshold = args.zcr_threshold
            # 只有开启了 ZCR 且开启了 Auto 才进行校准
            if args.auto_zcr:
                if (calibrated := calibrate_zcr_threshold(speech_probs, zcr_array)) is not None:
                    final_zcr_threshold = calibrated

            # 合并短块
            logger.debug("【VAD】正在合并短于1秒的语音块……")
            nonsilent_speeches = assign_vad_ownership_keep_windows(
                merge_short_segments_adaptive(
                    [{
                        "start_sample": s,
                        "end_sample": e,
                        "keep_start_sample": s,
                        "keep_end_sample": e,
                    } for s, e in speeches],
                    MAX_SPEECH_DURATION_SAMPLES
                ),
                keep_silence_samples,
                total_audio_samples
            )
            if len(speeches) != len(nonsilent_speeches):
                logger.debug(f"【VAD】原 {len(speeches)} 个语音块已合并为 {len(nonsilent_speeches)} 个语音块")
            else:
                logger.debug(f"【VAD】没有需要合并的语音块")

            # 计划表：每个元素代表一次实际推理的音频窗（未 pad 的范围）及其 keep/own 信息
            planned_segments = []

            if args.debug and not api_mode:
                temp_chunk_dir = Path(tempfile.mkdtemp())

            # 遍历一级块
            for i, chunk_dict in enumerate(nonsilent_speeches):
                # --- 准备一级语音块 ---
                # 计算包含静音保护的起止时间
                start_sample = max(0, chunk_dict["start_sample"] - keep_silence_samples)
                end_sample = min(total_audio_samples, chunk_dict["end_sample"] + keep_silence_samples)
                
                # 一级切片 (使用 Tensor 切片)
                chunk = slice_waveform_sample(waveform, start_sample, end_sample)

                # debug模式下为了检查才导出该块
                if args.debug and not api_mode:
                    save_tensor_as_wav(temp_chunk_dir / f"chunk_{i + 1}.wav", chunk)

                logger.info(
                    f"【VAD】正在处理语音块 {i + 1}/{len(nonsilent_speeches)} （该块起止时间："
                    f"{fmt_time(samp_to_ms_floor(chunk_dict['start_sample']) / 1000)} --> "
                    f"{fmt_time(samp_to_ms_ceil(chunk_dict['end_sample']) / 1000)}，"
                    f"时长：{(chunk_dict['end_sample'] - chunk_dict['start_sample']) / SAMPLERATE:.2f} 秒）",
                    end="", flush=True,
                )

                # 判断短块还是长块
                if end_sample - start_sample <= 10 * SAMPLERATE:
                    # === 短块 ===
                    logger.info(" --> 短块，直接识别")
                    
                    audio_start_sample = max(0, int(start_sample) - int(ms_to_samp_round(1200)))
                    audio_end_sample = min(int(total_audio_samples), int(end_sample) + int(ms_to_samp_round(600)))
                    if audio_end_sample < audio_start_sample:
                        audio_end_sample = audio_start_sample
                    planned_segments.append({
                        "plan_index": len(planned_segments),

                        "chunk_index": i,
                        "display_number": i + 1,
                        # 推理时再从原始 waveform 切片并 pad（保证只 pad 一次）
                        "audio_start_sample": audio_start_sample,
                        "audio_end_sample": audio_end_sample,
                        "base_offset_sample": audio_start_sample,
                        # 归属：只看 own_keep
                        "own_keep_start_sample": chunk_dict["own_keep_start_sample"],
                        "own_keep_end_sample": chunk_dict["own_keep_end_sample"],
                        # 保留：只看 keep（短块没有二次切分窗口，keep=own_keep）
                        "keep_start_sample": chunk_dict["own_keep_start_sample"],
                        "keep_end_sample": chunk_dict["own_keep_end_sample"],
                    })

                
                else:
                    # === 长块 (二次切分) ===
                    logger.info("") # 断开上一行
                    # 运行局部 VAD
                    sub_speeches, sub_speech_probs, _, sub_frame_times_sample = get_speech_timestamps_onnx(
                        chunk,
                        onnx_session,
                        args.vad_threshold,
                        args.vad_end_threshold,
                        min_speech_samples,
                        min_silence_samples
                    )
                    
                    if not sub_speeches:
                        # 兜底策略：如果二次 VAD 没切出来（例如全是噪音），回退到整块识别
                        logger.warn("    【VAD】二次 VAD 未发现有效分割点，回退到整块识别")
                        # 构造一个覆盖整个 chunk 的虚拟 VAD 段
                        sub_speeches = [[0, chunk.shape[1]]]

                    # 局部特征计算
                    sub_speech_probs, sub_energy_array, sub_zcr_array = prepare_acoustic_features(
                            chunk, sub_speech_probs, args.zcr
                    )
                    
                    # 智能切分
                    nonsilent_sub_speeches = []
                    for seg in sub_speeches:
                        # seg 是列表 [start, end]
                        # seg[1] 是相对于含 Padding 的 chunk 的时间
                        # start_sample 是 chunk 在原音频中的起始时间（含 keep_silence）
                        # PAD_SAMPLES 是 chunk 头部人为添加的静音
                        vad_chunk_end_samples.append(start_sample + seg[1] - PAD_SAMPLES)

                        nonsilent_sub_speeches.extend(global_smart_segmenter(
                            seg[0],
                            seg[1],
                            sub_speech_probs,
                            sub_energy_array,
                            sub_frame_times_sample,
                            MAX_SPEECH_DURATION_SAMPLES,
                            args.vad_threshold,
                            final_zcr_threshold,
                            sub_zcr_array
                        ))
                    logger.debug(
                            f"【VAD】侦测到 {len(nonsilent_sub_speeches)} 个子语音块"
                        )

                    # 再次合并子块
                    refined_sub_speeches = merge_short_segments_adaptive(nonsilent_sub_speeches, MAX_SPEECH_DURATION_SAMPLES)

                    if len(refined_sub_speeches) > 1:
                        logger.info(f"    --> 拆为 {len(refined_sub_speeches)} 个子片段：")

                    # 遍历子块
                    for sub_idx, sub_seg in enumerate(refined_sub_speeches):
                        # 最终时间 = 一级块全局偏移 + 二级块在一级块内的偏移 + 识别出的相对时间
                        # 因为 sub_seg 是基于含填充的父chunk计算的，它包含了父chunk头部的 0.5s 静音，必须扣除
                        # 先把 sub_seg 的 keep 窗口映射到全局音频
                        keep_start_sample = start_sample + sub_seg["keep_start_sample"] - PAD_SAMPLES
                        keep_end_sample = start_sample + sub_seg["keep_end_sample"] - PAD_SAMPLES
                        # 推理窗同样映射到全局
                        audio_start_sample = start_sample + sub_seg["start_sample"] - PAD_SAMPLES
                        audio_end_sample = start_sample + sub_seg["end_sample"] - PAD_SAMPLES

                        # clamp 到 [0, total_audio_samples]，并确保 end >= start
                        keep_start_sample = max(0, min(keep_start_sample, total_audio_samples))
                        keep_end_sample = max(0, min(keep_end_sample, total_audio_samples))
                        if keep_end_sample < keep_start_sample:
                            keep_end_sample = keep_start_sample

                        audio_start_sample = max(0, min(audio_start_sample, total_audio_samples))
                        audio_end_sample = max(0, min(audio_end_sample, total_audio_samples))
                        if audio_end_sample < audio_start_sample:
                            audio_end_sample = audio_start_sample

                        audio_start_sample = max(0, int(audio_start_sample) - int(ms_to_samp_round(1200)))
                        audio_end_sample = min(int(total_audio_samples), int(audio_end_sample) + int(ms_to_samp_round(600)))
                        if audio_end_sample < audio_start_sample:
                            audio_end_sample = audio_start_sample
                        planned_segments.append({
                            "plan_index": len(planned_segments),
                            "chunk_index": i,
                            "display_number": f"{i+1}-{sub_idx+1}",
                            "audio_start_sample": audio_start_sample,
                            "audio_end_sample": audio_end_sample,
                            "base_offset_sample": audio_start_sample,
                            # 判断块归属使用 own_keep
                            "own_keep_start_sample": int(chunk_dict["own_keep_start_sample"]),
                            "own_keep_end_sample": int(chunk_dict["own_keep_end_sample"]),
                            # 保留、去重等使用 keep
                            "keep_start_sample": keep_start_sample,
                            "keep_end_sample": keep_end_sample,
                        })

                        # Debug 时导出临时子块文件，从 waveform 直接切（只 pad 一次）
                        if args.debug and not api_mode:
                            save_tensor_as_wav(
                                temp_chunk_dir / f"chunk_{i + 1}_sub_{sub_idx + 1}.wav",
                                slice_waveform_sample(waveform, audio_start_sample, audio_end_sample)
                                )

                        if len(refined_sub_speeches) != 1:
                            logger.info(
                                f"第 {i + 1}-{sub_idx + 1} 段 {fmt_time(samp_to_ms_floor(int(audio_start_sample)) / 1000)} --> {fmt_time(samp_to_ms_ceil(int(audio_end_sample)) / 1000)}，"
                                f"时长：{(int(audio_end_sample) - int(audio_start_sample)) / SAMPLERATE:.2f} 秒"
                            )

        # 初始化 subwords_by_plan
        subwords_by_plan = [[] for _ in range(len(planned_segments))]

        # 执行推理并收集
        for meta in planned_segments:
            batch_audio.append(
                slice_waveform_sample(
                    waveform,
                    meta["audio_start_sample"],
                    meta["audio_end_sample"],
                ).squeeze(0)
            )
            batch_meta.append(meta)

            if len(batch_audio) >= BATCH_SIZE:
                flush_batch()

        flush_batch()

        # 每个 plan 内先：排序 + 段首 prefix 去 clamp 重定位 + 单调化
        for meta in planned_segments:
            subs = subwords_by_plan[int(meta["plan_index"])]
            if not subs:
                continue
            """
            只对“段首被 clamp 的 prefix token”（raw_local<0 的连续前缀）做去 clamp 化重定位：
              raw_local = step_index * STEP_SAMPLES - PAD_SAMPLES
            将这段 prefix 的 start_sample 重定位到 keep_start_sample 附近，并做 1 sample 的递增展开避免同一时间戳堆叠。
            """
            # 以 step_index 为主排序，保证 prefix 的判断稳定
            subs.sort(key=lambda s: (s.step_index, s.start_sample))
        
            anchor = max(0, min(int(meta["keep_start_sample"]), int(total_audio_samples)))
        
            # 找连续 prefix：raw_local < 0
            p = 0
            for sw in subs:
                raw_local = int(sw.step_index) * STEP_SAMPLES - PAD_SAMPLES
                if raw_local < 0:
                    p += 1
                else:
                    break
        
            if p <= 0:
                continue
        
            # 重定位 prefix 到 keep_start 附近，并展开
            for j in range(p):
                subs[j].start_sample = min(anchor + j, int(total_audio_samples))
    
            enforce_monotonic_start_inplace(subs, int(total_audio_samples))

        # 相邻 plan 做 40 token 精确 token_id 重叠检测 
        merged = []
        prev_meta = None

        for meta in planned_segments:
            right = subwords_by_plan[int(meta["plan_index"])]
            if not right:
                continue

            if not merged:
                merged.extend(right)
                prev_meta = meta
                continue

            # 取尾/头最多 40 个 token_id 做精确匹配
            left_tail = merged[-40:] if len(merged) > 40 else merged
            right_head = right[:40] if len(right) > 40 else right
            """返回 k：left 的后 k 个 token_id == right 的前 k 个 token_id（精确匹配）"""
            left_ids = [sw.token_id for sw in left_tail]
            right_ids = [sw.token_id for sw in right_head]

            k = 0
            max_k = min(40, len(left_ids), len(right_ids))
            for kk in range(max_k, 1, -1):
                if left_ids[-kk:] == right_ids[:kk]:
                    k = kk
                    break

            if k <= 0:
                merged.extend(right)
                prev_meta = meta
                continue

            # 删左还是删右
            prev_keep_end = int(prev_meta["keep_end_sample"])
            curr_keep_start = int(meta["keep_start_sample"])
            Bmid = (prev_keep_end + curr_keep_start) // 2
            margin = int(ms_to_samp_round(200))

            left_overlap = merged[-k:]
            right_overlap = right[:k]

            tL = int(np.median([int(s.start_sample) for s in left_overlap]))
            tR = int(np.median([int(s.start_sample) for s in right_overlap]))

            right_clamped_like = (tR <= int(meta["audio_start_sample"]) + int(ms_to_samp_round(50)))

            if (tL < Bmid - margin) and (tR >= Bmid - margin) and (not right_clamped_like):
                # 删左保右
                del merged[-k:]
                merged.extend(right)
            else:
                # 删右保左
                merged.extend(right[k:])

            prev_meta = meta

        raw_subwords = merged
        enforce_monotonic_start_inplace(raw_subwords, int(total_audio_samples))
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

        all_segments_info, all_subwords_info = create_precise_segments_from_subwords(
            raw_subwords, sorted(set(vad_chunk_end_samples)), total_audio_samples, args.no_remove_punc
        )

        logger.info("文本片段生成完成")

        if args.refine_tail: # 只在启用精修且map存在时精修
            # 短窗平滑，5 是窗口大小
            kernel = np.ones(5, dtype=np.float32) / 5

            logger.info("【精修】正在修除每段的尾部静音……")
            for i, segment in enumerate(all_segments_info):
                # 遍历map，所以需要通过索引i来更新原始的all_segments_info列表
                segment.end_sample = refine_tail_timestamp(
                    segment.subwords[-1].start_sample,
                    segment.end_sample,
                    speech_probs,
                    frame_times_sample,
                    frame_hop_samples,
                    min(segment.vad_limit_sample, all_segments_info[i + 1].start_sample if i < len(all_segments_info) - 1 else segment.vad_limit_sample),
                    min_silence_samples,
                    ms_to_samp_round(args.tail_lookahead_ms),
                    ms_to_samp_round(args.tail_safety_margin_ms),
                    ms_to_samp_round(args.tail_min_keep_ms),
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
                )
            # 邻段防重叠微调
            for i in range(len(all_segments_info) - 1):
                if all_segments_info[i].end_sample > all_segments_info[i + 1].start_sample:
                    all_segments_info[i].end_sample = all_segments_info[i + 1].start_sample
            logger.info("【精修】结束时间戳精修完成")

        logger.info("=" * 70)
        logger.info("识别完成，正在生成输出文件……")

        # 为输出做准备
        output_dir = input_path.parent.resolve()
        base_name = input_path.stem
        precise_subwords = [transform_subword_info(sw) for sw in all_subwords_info]
        precise_segments = [transform_segment_info(seg) for seg in all_segments_info]

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
            full_text = " ".join(segment.text for segment in all_segments_info)

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
                    for segment in precise_segments
                ]
            else:
                with (output_path := output_dir / f"{base_name}.segments.txt").open("w", encoding="utf-8") as f:
                    writer = TextWriter(f)
                    for segment in precise_segments:
                        writer.write(segment)
                logger.info(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            def _write_segment2srt(f):
                writer = SRTWriter(f)
                for segment in precise_segments:
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
                for segment in precise_segments:
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
                for segment in precise_segments:
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
                    for sub in precise_subwords
                ]
            else:
                with (output_path := output_dir / f"{base_name}.subwords.txt").open("w", encoding="utf-8") as f:
                    for sub in precise_subwords:
                        f.write(
                            f"[{fmt_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                        )
                logger.info(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            def _write_subword2srt(f):
                for i, sub in enumerate(precise_subwords):
                    f.write(f"{i + 1}\n")
                    f.write(f"{fmt_time(sub.seconds)} --> {fmt_time(sub.end_seconds)}\n")
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
                    for sub in precise_subwords
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

                for segment in precise_segments:
                    karaoke_text = "".join(
                        f"{{\\k{math.ceil(round((sub.end_seconds - sub.seconds) * 100))}}}{sub.token}"
                        for sub in segment.subwords
                    )
                    f.write(
                        f"Dialogue: 0,"
                        f"{ASSWriter._format_time(segment.start_seconds)},"
                        f"{ASSWriter._format_time(segment.end_seconds)},"
                        f"Default,,0,0,0,,{karaoke_text}\n"
                    )
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
            if _VAD_ONNX_MODEL_LOAD_COST > 0:
                logger.info(
                    f"Pyannote-segmentation-3.0 模型加载耗时：{format_duration(_VAD_ONNX_MODEL_LOAD_COST)}"
                )
                logger.info(
                    f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time - _VAD_ONNX_MODEL_LOAD_COST)}"
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
        api_result["duration"] = samp_to_ms_ceil(total_audio_samples) / 1000
        return api_result            

if __name__ == "__main__":
    print("请勿直接运行本文件")
    sys.exit(1)