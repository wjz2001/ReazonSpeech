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
import json
import time
import numpy as np

# ONNX VAD 依赖
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# Unix 系统有resource，Windows 上可能 ImportError
# 仅 Windows 使用 WinAPI
if sys.platform == "win32":
    from ctypes import wintypes, windll
else:
    import resource  

from collections import Counter
from pathlib import Path
from pydub import AudioSegment
from omegaconf import open_dict
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
MEM_SAMPLES = {
    "ram_gb": [],
    "gpu_gb": [],
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
        if resource is not None:
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
    后台线程：周期性采样当前进程的 RAM 和 GPU 显存使用情况。
    interval: 采样间隔（秒），表示每几秒采样一次，可按需要调大/调小
    """
    while not stop_event.is_set():
        # 进程 RAM（GB）
        MEM_SAMPLES["ram_gb"].append(_get_ram_gb_sample())

        # 如果有 CUDA，就记录一次显存
        if torch.cuda.is_available():
            MEM_SAMPLES["gpu_gb"].append(torch.cuda.memory_allocated(device) / (1024 ** 3))

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
    return most_bin + bin_size / 2.0, freq
# === 内存 / 显存监控结束 ===

def auto_tune_batch_size(model, max_seg_sec):
    """
    根据显存自动估算 Batch Size
    """

    if not torch.cuda.is_available():
        return 1

    device = torch.cuda.current_device()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.inference_mode():
            _ = model.transcribe(
                # 构造 dummy 数据 (30s 空白音频)
                audio=[torch.zeros(int(max_seg_sec * SAMPLERATE), dtype=torch.float32)],
                batch_size=1, return_hypotheses=True, verbose=False
                )
    except RuntimeError:
        return 1 # OOM 或其他错误，回退到 1

    per_sample = max(torch.cuda.max_memory_allocated(device) - baseline, 1)
    torch.cuda.empty_cache()
    
    
    # 计算理论最大值，至少为 1，0.7和2是安全限制
    return max(1, int(torch.cuda.mem_get_info(device)[0] * 0.7) // per_sample // 2)

def calculate_frame_cost(frame_idx, speech_probs, energy_array, vad_threshold, zcr_threshold, zcr_array):
    """计算单帧作为切分点的代价（代价越低越适合切分）。
       ZCR 判定使用 max(|x1|, |x2|) > th。
    """
    # 基础 VAD 概率成本
    if speech_probs[frame_idx] > vad_threshold:
        return 100.0 # 惩罚高概率点，强硬约束,确信是语音的地方绝不切

    # --- 动态抗噪过零率 (Robust ZCR) ---
    # 逻辑：(过零) AND (穿越点的幅度足够大，不仅是微小抖动)
    # 标准做法：check max(abs(x[n]), abs(x[n+1])) > th
    zcr_cost = 0.0
    if zcr_array is not None:
        if zcr_array[frame_idx] > zcr_threshold: 
            zcr_cost = zcr_array[frame_idx] * 50.0 

    # 局部平滑度 (斜率)
    slope_cost = abs(speech_probs[frame_idx + 1] - speech_probs[frame_idx - 1]) * 2.0

    return speech_probs[frame_idx] + min(energy_array[frame_idx], 0.5) * 10.0 + slope_cost + zcr_cost

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

def convert_audio_to_tensor(initial_audio):
    """
    将 Pydub AudioSegment 转换为 PyTorch Tensor (float32, [-1, 1])。
    使用 np.frombuffer 读 int16，再用 torch.tensor 拷贝到 GPU/CPU 张量
    """
    # torch.tensor(...) 总是拷贝数据，不依赖 numpy 的可写性
    # [T] float32
    # 归一化到 [-1, 1]
    # 升维 [1, T]
    return torch.tensor(np.frombuffer(initial_audio.raw_data, dtype=np.int16), dtype=torch.float32).div_(32768.0).unsqueeze(0)

def create_precise_segments_from_subwords(raw_subwords, vad_chunk_end_times_s, total_duration_s, no_remove_punc):

    all_subwords = []
    durations = [] 
    
    vad_cursor = 0
    for i, sub in enumerate(raw_subwords):
        # 计算自然结束时间 (Next Start)
        next_start = raw_subwords[i + 1].seconds if i < len(raw_subwords) - 1 else sub.seconds + SECONDS_PER_STEP # 最后一个子词，用 step 兜底
        
        # 计算 VAD 限制 (VAD Limit)
        # 移动游标找到当前子词所属的 VAD 块
        # 利用短路逻辑：如果 vad_chunk_end_times_s 为空，循环不会执行
        while vad_cursor < len(vad_chunk_end_times_s) and sub.seconds > vad_chunk_end_times_s[vad_cursor]:
            vad_cursor += 1
        # 最后一段没有vad边界，用整个音频的总时长作为边界
        current_vad_limit = vad_chunk_end_times_s[vad_cursor] if vad_cursor < len(vad_chunk_end_times_s) else total_duration_s

        # 计算最终结束时间逻辑：取 (开始+step) 和 (min(下一个开始, VAD限制)) 中的较大者
        # 即保证不短于 step，且不超过 VAD 边界和下一个词的开始
        end_seconds = max(
            sub.seconds + SECONDS_PER_STEP,
            min(next_start, current_vad_limit)
        )

        # 收集 duration 用于后续停顿阈值计算
        durations.append(next_start - sub.seconds)

        all_subwords.append(PreciseSubword(
            seconds=sub.seconds,
            token_id=sub.token_id,
            token=sub.token,
            end_seconds=end_seconds,
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
        # 只有当有足够多的数据点（例如超过20个子词间隔）时，才计算并启用阈值
        if len(durations[start:end_idx]) > SUBWORDS_PER_SEGMENTS * 1.5:
            # 基于语速/停顿的边界补充
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
            curr_text_raw = "".join(s.token for s in current_subwords) # 原始文本
    
            # 判断是否为纯标点/空格片段
            if all((s.token in TOKEN_PUNC or not s.token.strip()) for s in current_subwords):
                # === 分支 A：纯标点片段 ===
                if all_segments:
                    prev_seg = all_segments[-1]
                    
                    if logger.debug_mode:
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
                text = curr_text_raw
                # 如果需要剔除句末标点，且最后一个子词是标点
                if not no_remove_punc and current_subwords[-1].token in TOKEN_PUNC:
                    removed_punc = current_subwords.pop()

                    if logger.debug_mode:
                        logger.debug(f"已剔除 {SRTWriter._format_time(removed_punc.seconds)} --> {SRTWriter._format_time(removed_punc.end_seconds)}：{removed_punc.token}")

                    # 因为不是全标点句，pop 后列表一定不为空
                    # 填补时间空缺
                    current_subwords[-1].end_seconds = removed_punc.end_seconds
            
                    # 重新生成已剔除标点最终文本
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

def find_optimal_splits_dp_strict(candidates, num_cuts, start_frame, end_frame, frame_duration_s, max_duration_s):
    """严格硬约束的动态规划算法。"""
    max_frames = int(max_duration_s / frame_duration_s)
    min_frames = int(1.0 / frame_duration_s) 
    
    dp = [[(float("inf"), []) for _ in range(len(candidates))] for _ in range(num_cuts + 1)]

    # 第一刀
    for j in range(len(candidates)):
        if min_frames <= (cut_frame := candidates[j]["frame"]) - start_frame <= max_frames:
            dp[1][j] = (candidates[j]["cost"], [cut_frame])

    # 后续每一刀
    for i in range(2, num_cuts + 1):
        for j in range(i - 1, len(candidates)):
            curr_frame = candidates[j]["frame"]
            for k in range(i - 2, j):
                if min_frames <= curr_frame - candidates[k]["frame"] <= max_frames:
                    prev_cost, prev_path = dp[i-1][k]
                    if prev_cost != float("inf"):
                        new_total_cost = prev_cost + candidates[j]["cost"]
                        if new_total_cost < dp[i][j][0]:
                            dp[i][j] = (new_total_cost, prev_path + [curr_frame])

    # 最终检查
    final_best_cost = float("inf")
    final_best_path = []
    for j in range(num_cuts - 1, len(candidates)):
        if min_frames <= end_frame - candidates[j]["frame"] <= max_frames:
            cost, path = dp[num_cuts][j]
            if cost < final_best_cost:  # ✅ 比较成本
                final_best_cost = cost
                final_best_path = path

    return final_best_path

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
def get_asr_model():
    """
    获取 Reazonspeech 模型的单例
    如果模型未加载，则加载并缓存；如果已加载，则直接返回
    """
    global _ASR_MODEL, _ASR_MODEL_LOAD_COST
    if _ASR_MODEL is None:
        logger.info("正在加载 Reazonspeech ……")
        asr_model_load_start = time.time()  # <--- 计时开始
        # 调用原有的 load_model 函数
        _ASR_MODEL = load_model()
        asr_model_load_end = time.time()  # <--- 计时结束
        logger.info(f"Reazonspeech 加载完成")
        _ASR_MODEL_LOAD_COST = asr_model_load_end - asr_model_load_start
    return _ASR_MODEL

# --- ONNX VAD 辅助函数 ---
def get_speech_timestamps_onnx(
    waveform,
    onnx_session,
    threshold,
    neg_threshold, #语音结束阈值（低阈值）
    min_speech_duration_ms, #最小语音段时长
    min_silence_duration_ms #多长静音视为真正间隔
):
    """使用 ONNX 模型和后处理来获取语音时间戳"""
    # ONNX 模型需要特定的输入形状
    # 运行 ONNX 模型
    #  ONNX 输出 (batch, T, classes)，取 [0] 变成 (T, classes)，再取[0]
    logits = torch.from_numpy(
        onnx_session.run(
            None,
            {"input_values": waveform.unsqueeze(0).numpy()}
        )[0]
        )[0]
    # 聚合语音组能量 (Index 1-6)
    # torch.logsumexp 默认 keepdim=False，输入 (T,6) -> 输出 (T)
    # 获取静音能量 (Index 0) -> (T)
    # 竞争: 语音 - 静音，再Sigmoid
    # 转回 Numpy 以便进入 Python 循环
    # logits[:, 0] > 静音维度，logits[:,1:] > 语音维度
    speech_probs = torch.sigmoid(torch.logsumexp(logits[:,1:], dim=1) - logits[:, 0]).numpy()

    frame_duration_s = (waveform.shape[1] / SAMPLERATE) / logits.shape[0]

    speeches = []
    current_speech_start = None
    triggered = False
    temp_silence_start_frame = None
    min_silence_frames = max(1, int(min_silence_duration_ms / (frame_duration_s * 1000.0)))

    for i, prob in enumerate(speech_probs):
        # --- 语音开始逻辑（高阈值） ---
        if not triggered:
            if prob >= threshold:
                triggered = True
                current_speech_start = i * frame_duration_s
            continue

        # --- 处于语音段中时，跟踪静音区域 ---
        if prob < neg_threshold:
            # 低于结束阈值，可能是静音的开始
            if temp_silence_start_frame is None:
                temp_silence_start_frame = i
        else:
            # 回到语音，清掉静音起点
            temp_silence_start_frame = None

        # --- 按静音长度结束当前段 ---
        if temp_silence_start_frame is not None:
            # ==== 将 ms/s 转为帧数 ====
            if (i - temp_silence_start_frame) >= min_silence_frames:
                # 访问列表索引 0 即start
                if ((end_time := temp_silence_start_frame * frame_duration_s) - current_speech_start) * 1000.0 >= min_speech_duration_ms:
                    speeches.append([current_speech_start, end_time])
                    # 收尾并准备下一段
                triggered = False
                current_speech_start = None
                temp_silence_start_frame = None

    # --- 处理结尾残留的语音段 ---
    if triggered and current_speech_start is not None:
        if ((end_time := len(speech_probs) * frame_duration_s) - current_speech_start) * 1000.0 >= min_speech_duration_ms:
            speeches.append([current_speech_start, end_time])

    # 完整的语音概率数组 speech_probs 和每一帧的持续时间 frame_duration_s
    return speeches, speech_probs, frame_duration_s

def global_smart_segmenter(start_s, end_s, speech_probs, energy_array, frame_duration_s, max_duration_s, vad_threshold, zcr_threshold, zcr_array, overlap_s):
    """
    入口函数：分析一个长 VAD 段，返回切割好的子段列表。
    """

    # 计算必须切几刀
    if (num_cuts := max(0, math.ceil((end_s - start_s) / (max_duration_s - overlap_s)) - 1)) == 0:
        return [[start_s, end_s]]

    start_frame = int(start_s / frame_duration_s)
    end_frame = int(end_s / frame_duration_s)

    # 生成候选点 (筛选 VAD 概率 < max(0.15, vad_threshold - 0.1) 的局部低点)
    candidates = []
    # 边界保护
    search_start = start_frame + int(1.0 / frame_duration_s)
    search_end = min(len(speech_probs), end_frame - int(1.0 / frame_duration_s))

    # 提取搜索区间的概率
    # search_end > search_start + 2 是为了确保区间内至少有 3 帧数据
    # 寻找局部极小值需要比较：当前帧(curr) <= 前一帧(prev) 且 当前帧(curr) <= 后一帧(next)
    # 这对应了下方的切片逻辑：[1:-1] (curr), [:-2] (prev), [2:] (next)
    # 如果总帧数少于 3 (即 end - start <= 2)，切片后的维度将无法对齐或为空，因此跳过
    if search_end > search_start + 2:
        # 利用 numpy 寻找局部极小值且概率 < max(0.15, vad_threshold - 0.1) 的点
        # 逻辑：当前点 <= 前一点 AND 当前点 <= 后一点 AND 当前点 < max(0.15, vad_threshold - 0.1)
        p_curr = speech_probs[search_start+1:search_end-1] # 对应原数组的索引 i
        p_prev = speech_probs[search_start:search_end-2] # 对应 i-1
        p_next = speech_probs[search_start+2:search_end] # 对应 i+1

        # 生成布尔掩码
        mask = (p_curr < max(0.15, vad_threshold - 0.1)) & (p_curr <= p_prev) & (p_curr <= p_next)
        
        # 获取满足条件的相对索引
        # 还原绝对索引：+1 是因为 p_curr 从切片的第1个元素开始，+search_start 是切片偏移
        # 遍历筛选出的索引计算成本
        for idx in (np.where(mask)[0] + 1 + search_start):
            cost = calculate_frame_cost(idx, speech_probs, energy_array, vad_threshold, zcr_threshold, zcr_array)
            candidates.append({"frame": idx, "cost": cost})
    
    # 执行 DP
    best_cuts_frames = []
    if len(candidates) >= num_cuts:
        best_cuts_frames = find_optimal_splits_dp_strict(
            candidates, num_cuts, start_frame, end_frame, frame_duration_s, (max_duration_s - overlap_s)
        )
    
    # 构建结果
    if not best_cuts_frames:
        # 如果找不到最佳切点，直接返回原始的长段落，把切分决策交给后续流程
        return [[start_s, end_s]]
        
    final_segments = []
    segment_start_s = start_s
    for cut_frame in best_cuts_frames:
        final_segments.append([segment_start_s, (split_s := cut_frame * frame_duration_s)])
        segment_start_s = split_s - overlap_s
    final_segments.append([segment_start_s, end_s])

    return final_segments

def merge_overlap_dedup(chunk_results):
    """重叠区域：文本相同则去重，否则保留双方并打印提示"""
    def _overlap_interval(range_a, range_b):
        s, e = max(range_a[0], range_b[0]), min(range_a[1], range_b[1])
        return (s, e) if e > s else (None, None)
    
    def _is_essentially_empty(chars):
        """判断 chars 列表是否实质上为空（即为空列表，或仅包含TOKEN_PUNC）"""
        # 将所有 chars 拼接后检查，如果只有空格或空字符串，视为空
        text = "".join(chars).strip()
        # 检查是否所有字符都在TOKEN_PUNC集合中
        return not text or all(char in TOKEN_PUNC for char in text)

    merged = list(chunk_results[0]["subwords"])

    for i in range(1, len(chunk_results)):
        curr_subs = chunk_results[i]["subwords"]
        # chunk_ranges_s 必须按 end 单调递增
        s_ov, e_ov = _overlap_interval(chunk_results[i-1]["range"], chunk_results[i]["range"])

        # 如果没有重叠，直接连接
        if s_ov is None:
            merged.extend(curr_subs)
            continue

        # --- 有重叠区域 ---
        # 旧块中不属于重叠区的子词
        prev_non_overlap = [s for s in merged if s.seconds < s_ov]
        # 旧块中属于重叠区的子词
        prev_overlap = [s for s in merged if s_ov <= s.seconds < e_ov]
        # 提取旧块重叠区的 token 文本
        prev_tokens = [s.token for s in prev_overlap]
        
        # 新块中不属于重叠区的子词
        curr_non_overlap = [s for s in curr_subs if s.seconds >= e_ov]
        # 新块中属于重叠区的子词
        curr_overlap = [s for s in curr_subs if s_ov <= s.seconds < e_ov]
        # 提取新块重叠区的 token 文本
        curr_tokens = [s.token for s in curr_overlap]

        # 判定：是否完全一致
        if prev_tokens == curr_tokens:
            # 完全一致，直接合并（去重），只保留新块的重叠部分
            final_overlap = curr_overlap
            if prev_tokens and curr_tokens: #不为空
                logger.debug(f"已合并重叠区域，该区域起止时间：{SRTWriter._format_time(s_ov)} --> {SRTWriter._format_time(e_ov)}")
        # 旧块是空的（或纯标点），不管新块有没有内容 -> 信任新块（覆盖旧的）
        # 这涵盖了：旧无新有（用新）、旧无新无（用新，即更新标点）、旧标点新内容（用新）
        elif _is_essentially_empty(prev_tokens):
            final_overlap = curr_overlap
            
        # 旧块有实质内容，但新块是空的（或纯标点） -> 信任旧块
        # 这涵盖了：旧有新无（保留旧的，丢弃新产生的纯标点噪音）
        elif _is_essentially_empty(curr_tokens):
            final_overlap = prev_overlap
        else:
            # 旧块和新块的重叠部分都保留，按时间排序
            final_overlap = prev_overlap + curr_overlap
            final_overlap.sort(key=lambda x: x.seconds)
            
            logger.info(f"【提示】发现有效重叠区域，该区域起止时间：{SRTWriter._format_time(s_ov)} --> {SRTWriter._format_time(e_ov)}")
            logger.info(f"    --> 前段为: {''.join(prev_tokens)}")
            logger.info(f"    --> 后段为: {''.join(curr_tokens)}")

        merged = prev_non_overlap + final_overlap + curr_non_overlap

    return merged

def merge_short_segments_adaptive(segments, max_duration):
    """
    把短于 1 秒的片段合并到相邻较短的片段中，
    优先合并更短的一侧，并确保合并后的总时长不超过 max_duration
    """
    if not segments:
        return []

    # 创建副本以避免修改原始数据
    working_segments = [list(s) for s in segments]
    result = []
    
    i = 0
    while i < len(working_segments):
        seg = working_segments[i]

        if (seg_duration := seg[1] - seg[0]) <= 1: # 短于 1 才合并
            # 获取前后邻居
            prev_seg = result[-1] if result else None
            next_seg = working_segments[i + 1] if i + 1 < len(working_segments) else None
            
            dur_prev = (prev_seg[1] - prev_seg[0]) if prev_seg else float('inf')
            dur_next = (next_seg[1] - next_seg[0]) if next_seg else float('inf')

            # 向左合的新时长 = 当前结束 - 前段开始
            dur_if_left = (seg[1] - prev_seg[0]) if prev_seg else float('inf')
            # 向右合的新时长 = 后段结束 - 当前开始
            dur_if_right = (next_seg[1] - seg[0]) if next_seg else float('inf')

            target_side = None # "left" or "right"
            # 两边都合法，哪边合并后更短合哪边
            if dur_if_left <= max_duration and dur_if_right <= max_duration:
                target_side = "left" if dur_if_left <= dur_if_right else "right"
            # 只有左边合法（意味着右边超长或不存在）
            elif dur_if_left <= max_duration:
                target_side = "left"
            # 只有右边合法（意味着左边超长或不存在）
            elif dur_if_right <= max_duration:
                target_side = "right"

            if target_side == "left":
                logger.debug(f"【VAD】片段（{seg_duration:.2f} 秒）{SRTWriter._format_time(seg[0])} --> {SRTWriter._format_time(seg[1])}，向左合并: 前段（{dur_prev:.2f} 秒）{SRTWriter._format_time(prev_seg[0])} -->{SRTWriter._format_time(prev_seg[1])} 延长至 {dur_prev + seg_duration:.2f} 秒")
                prev_seg[1] = seg[1]
                i += 1
                continue
            elif target_side == "right":
                logger.debug(f"【VAD】片段（{seg_duration:.2f} 秒）{SRTWriter._format_time(seg[0])} --> {SRTWriter._format_time(seg[1])}，向右合并: 后段（{dur_next:.2f} 秒）{SRTWriter._format_time(next_seg[0])} -->{SRTWriter._format_time(next_seg[1])} 延长至 {seg_duration + dur_next:.2f} 秒")
                next_seg[0] = seg[0]
                i += 1
                continue
            else:
                logger.debug(f"【VAD】片段（{seg_duration:.2f} 秒）{SRTWriter._format_time(seg[0])} --> {SRTWriter._format_time(seg[1])} 合并后超过 {max_duration} 秒，放弃")

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

def prepare_acoustic_features(waveform, speech_probs, frame_duration_s, use_zcr):
    """计算并对齐声学特征（能量、ZCR），供智能切分使用"""

    # 计算帧长
    frame_length = int(round(frame_duration_s * SAMPLERATE))
    
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
        # 乘以 2 倍作为安全门限，并设定 1e-4 的硬下限
        noise_threshold = max(torch.quantile(frame_maxs_flat, 0.10).item() * 2, 1e-4)
        logger.debug(f"【ZCR】底噪门限已设定为：{noise_threshold:.6f}")

        # 计算 ZCR
        # 判定大音量区域
        is_loud = (waveform_centered.abs() > noise_threshold)
        
        # 向量化计算过零
        # 逻辑：(过零) AND (至少一边是大音量)
        zcr_tensor = torch.nn.functional.avg_pool1d(
            # 向量化计算所有采样点的过零情况 (1, T-1)
            # 等价于 chunk[:-1] * chunk[1:]
            # 逻辑或: 左边响 OR 右边响
            # 得到“有效过零点”的布尔矩阵 (1, T-1)
            (((waveform_centered[:, :-1] * waveform_centered[:, 1:]) < 0) & (is_loud[:, :-1] | is_loud[:,1:])).float().unsqueeze(0), # 需要 (N, C, L)
            kernel_size=frame_length,
            stride=frame_length,
            ceil_mode=True # 保证处理尾部
        ).view(-1) # 使用 .view(-1) 将输出展平为一维向量
        
        zcr_array = zcr_tensor.numpy()

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

def refine_tail_end_timestamp(
    last_token_start_s,   # 该段最后一个子词的开始时间
    rough_end_s,          # 原段结束时间
    speech_probs,         # VAD 概率序列（get_speech_timestamps_onnx 返回的第二项）
    frame_duration_s,
    max_end_s,            # 该段所属 VAD 大块的硬上限
    min_silence_duration_ms,   # 判定静音的最小时长
    lookahead_ms,         # 发现静音后再向后看一点，避免马上回到语音
    safety_margin_ms,     # 安全边距，避免切在突变点
    min_tail_keep_ms,     # 至少保留最后 token 后的这点时长，避免切得太硬
    percentile,           # 自适应阈值，值越大，越容易将高概率有语音区域判为静音
    offset,               # 在自适应阈值的基础上增加的固定偏移量，值越大，越容易将高概率区域有语音区域判为静音
    use_zcr,
    zcr_threshold,
    zcr_array,
    tail_zcr_high_ratio        # 高ZCR帧的占比阈值
):
    # 仅在“最后子词之后”的尾窗内搜索
    start_idx = int(last_token_start_s / frame_duration_s)
    end_idx = min(len(speech_probs), int(np.ceil(max_end_s / frame_duration_s)))
    # 至少要有 4 帧才能进行有效的统计学分析
    if start_idx >= len(speech_probs) or end_idx - start_idx <= 4:
        return min(rough_end_s, max_end_s)

    # 短窗平滑，5 是窗口大小
    p_smooth = np.convolve(speech_probs[start_idx:end_idx], np.ones(5, dtype=np.float32) / 5.0, mode="same")

    # 局部自适应阈值percentile + offset
    # 限制阈值范围，防止极端情况导致逻辑失效
    # 0.10 保证底噪容忍度，0.95 保证不会因为全 1.0 的概率导致无法切割
    dyn_tau = np.clip(float(np.percentile(p_smooth, percentile)) + offset, 0.10, 0.95)

    min_silence_frames = max(1, int(min_silence_duration_ms / 1000.0 / frame_duration_s))
    # 连续静音 + 滞回
    for i in range(0, len(p_smooth) - min_silence_frames):
        if np.all(p_smooth[i : i + min_silence_frames] < dyn_tau):
            # --- ZCR 校验 ---
            if use_zcr:
                z_start = start_idx + i
                z_end = z_start + min_silence_frames
                
                # 获取窗口内的 ZCR 数据
                # 计算窗口内超过 ZCR 阈值的帧的比例
                # np.mean(布尔数组) 相当于计算 True 的百分比
                high_zcr_ratio = np.mean(zcr_array[z_start : z_end] > zcr_threshold)
                
                # 只有当高 ZCR 帧的比例超过设定值（如 0.3）时，才触发保护
                if high_zcr_ratio > tail_zcr_high_ratio:
                    if logger.debug_mode:
                        logger.debug(f"【ZCR】疑似静音段 {SRTWriter._format_time(z_start * frame_duration_s)} --> {SRTWriter._format_time(z_end * frame_duration_s)} 内高频帧占比 {high_zcr_ratio:.2f} > {tail_zcr_high_ratio}，视为清辅音，跳过切分")
                    continue 

            if not np.any(p_smooth[
                i + min_silence_frames : 
                min(len(p_smooth), i + min_silence_frames + int(lookahead_ms / 1000.0 / frame_duration_s))
                ] >= min(0.98, dyn_tau + 0.05) ):  # 滞回
                return min(
                    max(
                        (start_idx + i) * frame_duration_s + safety_margin_ms / 1000.0,
                        last_token_start_s + min_tail_keep_ms / 1000.0,
                    ),
                    max_end_s
                )

    # 兜底：没找到稳定静音，保留原结束时间
    return min(rough_end_s, max_end_s)

def slice_waveform_ms(waveform, start_ms, end_ms):
    """
    高效 Tensor 切片与 Padding
    waveform: [1, T]
    """
    def ms_to_index(ms):
        """【新增】毫秒 -> 采样点下标"""
        return max(0, int(round(ms * SAMPLERATE / 1000.0)))

    # 边界限制
    start_idx = ms_to_index(start_ms)
    end_idx = ms_to_index(end_ms)

    # 切片 (View 操作，零内存拷贝)
    # Padding (如果需要)
    # F.pad: (padding_left, padding_right)
    chunk = torch.nn.functional.pad(
        waveform[:, start_idx:end_idx], (ms_to_index(int(PAD_SECONDS * 1000)), ms_to_index(int(PAD_SECONDS * 1000)))
        )

    return chunk

def main():
    OVERLAP_S = 1.0  # 此处定义重叠时长
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型识别语音，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题"
    )

    # 音频/视频文件路径
    parser.add_argument(
        "input_file",
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
        help="设置集束搜索（Beam Search）宽度，范围为 4 到 257 之间的整数，默认值是 4 ，更大的值可能更准确但更慢",
        )

    parser.add_argument(
        "--no-remove-punc",
        action="store_true",
        help="禁止自动剔除句末标点，保留原始识别结果",
    )

    # --- VAD核心参数 ---
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="禁用智能分块功能，一次性处理整个音频文件",
    )

    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.4,
        metavar="[0.05-1]",
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

    # --- 段尾精修参数（必须先使用VAD） ---
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
        help="【精修】在自适应阈值的基础上增加的固定偏移量。值越大，越容易将高概率区域语音区域判为静音",
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

    args = parser.parse_args()

    # 配置全局日志等级
    logger.set_debug(args.debug)

    # 校验 --no-chunk 和 VAD 参数的冲突
    if args.no_chunk:
    # 只有当用户修改了默认值，但加了 --no-chunk 时才报错
        
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
            "tail_lookahead_ms",
            "tail_safety_margin_ms",
            "tail_min_keep_ms",
            "tail_zcr_high_ratio"
        ]:
            if getattr(args, p) != parser.get_default(p):
                parser.error(f"【参数错误】未添加 --refine-tail，不能设置参数 --{p}")

    if not args.no_chunk:
        if onnxruntime is None:
            logger.warn("缺少 onnxruntime，请运行 'pip install onnxruntime'")
            return

        if not (local_onnx_model_path := Path(__file__).resolve().parent / "models" / "model_quantized.onnx").exists():
            logger.warn(
                f"未在 '{local_onnx_model_path}' 中找到 Pyannote-segmentation-3.0 模型"
            )
            logger.warn("请下载 model_quantized.onnx 并放入 models 文件夹")
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
            if not (0.1 <= args.tail_zcr_high_ratio <= 0.5):
                parser.error("tail_zcr_high_ratio 必须在（0.1-0.5）范围内，当前值错误")

    # 启动内存 / 显存监控线程
    if args.debug:
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
        # 转换为单声道，16kHz采样率，16bit，这是ASR模型的标准格式
        audio = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(SAMPLERATE).set_sample_width(2)
        total_audio_ms = len(audio)
        waveform = convert_audio_to_tensor(audio)  # [1, T]
        logger.info("转换完成")

        if args.debug:
            # Debug 模式下保留 silence 用于导出
            # 使用 pydub 创建静音段用于填充
            # PAD_SECONDS 是以秒为单位，pydub 需要毫秒
            silence_padding = AudioSegment.silent(duration=int(PAD_SECONDS * 1000), frame_rate=SAMPLERATE)
        else:
            del audio

        # 使用单例模式获取模型，避免重复加载
        model = get_asr_model()

        # 获取模型最大允许输入语音块长度
        MAX_SPEECH_DURATION_S = model.cfg.train_ds.max_duration

        # 设定 Batch Size
        if args.batch_size is None:
            if torch.cuda.is_available():
                logger.info(f"正在自动估算 Batch Size 的值……")
                BATCH_SIZE = auto_tune_batch_size(model, MAX_SPEECH_DURATION_S)
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
            with open_dict(new_decoding_cfg): # 使用 open_dict 使配置可修改
                new_decoding_cfg.beam.beam_size = args.beam
            
            # 在所有识别任务开始前，应用这个新的解码策略
            logger.info(f"正在应用新的解码策略：集束搜索宽度为 {args.beam} ……")
            model.change_decoding_strategy(new_decoding_cfg)

        # --- 逐块识别并校正时间戳 ---
        # --- 计时开始：核心识别流程 ---
        recognition_start_time = time.time()

        vad_chunk_end_times_s = []  # 用于存储VAD块的结束时间
        chunk_results = [] # 最终结果容器
        
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
                    hyps = model.transcribe(
                        audio=batch_audio,
                        batch_size=len(batch_audio),
                        return_hypotheses=True,
                        verbose=False,
                    )[0]  # 取元组第 0 个元素，即结果列表
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
                    except Exception:
                        pass
        
                with torch.inference_mode():
                    # 回退为逐条推理（内部会顺序处理整个 batch_audio 列表）
                    hyps = model.transcribe(
                        audio=batch_audio,
                        batch_size=1,
                        return_hypotheses=True,
                        verbose=False,
                    )[0]

            # 处理结果
            for hyp, meta in zip(hyps, batch_meta):
                if not hyp: continue
                # 解码
                decoded_subwords = decode_hypothesis(model, hyp).subwords
                # 加上时间偏移
                for sub in decoded_subwords:
                    sub.seconds += meta['base_offset']
                
                # 放入对应的一级块结果桶中
                all_chunk_subwords_collection[meta['chunk_index']].extend(decoded_subwords)

            batch_audio.clear()
            batch_meta.clear()

        # --- 分支 A: 不分块 (No Chunk) ---
        if args.no_chunk:
            if args.debug:
                fd, temp_full_wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                # 使用 pydub 导出
                (silence_padding + audio + silence_padding).export(Path(temp_full_wav_path), format="wav")

            logger.info("未使用VAD，一次性处理整个文件……")
            
            # 同样走 batch 流程（虽然只有 1 个）
            all_chunk_subwords_collection.append([]) # index 0
            
            batch_audio.append(
                torch.nn.functional.pad(waveform, (int(PAD_SECONDS * SAMPLERATE), int(PAD_SECONDS * SAMPLERATE))).squeeze(0)
                )
            batch_meta.append({'chunk_index': 0, 'base_offset': 0.0})
            flush_batch()
            
            if all_chunk_subwords_collection[0]:
                chunk_results.append({
                    "subwords": all_chunk_subwords_collection[0],
                    "range": (0.0, total_audio_ms / 1000.0)
                })

        # --- 分支 B: 使用 VAD 分块 ---
        else:
            logger.info("【VAD】正在从本地路径加载 Pyannote-segmentation-3.0 模型……")
            vad_model_load_start = time.time()
            onnx_session = onnxruntime.InferenceSession(
                str(local_onnx_model_path), providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.time()
            logger.info(f"【VAD】模型加载完成")

            # 运行 VAD (代码逻辑保持不变)
            speeches, speech_probs, frame_duration_s = get_speech_timestamps_onnx(
                waveform,
                onnx_session,
                args.vad_threshold,
                args.vad_end_threshold,
                args.min_speech_duration_ms,
                args.min_silence_duration_ms
            )

            if not speeches:
                logger.warn("【VAD】未侦测到语音活动")
                return # 这里如果在 try 块内 return，finally 块依然会执行

            # 声学特征准备
            if args.auto_zcr or args.refine_tail:
                speech_probs, _, zcr_array = prepare_acoustic_features(waveform, speech_probs, frame_duration_s, args.zcr)

            # ZCR 阈值校准 
            # 确定最终使用的 ZCR 阈值
            final_zcr_threshold = args.zcr_threshold
            # 只有开启了 ZCR 且开启了 Auto 才进行校准
            if args.auto_zcr:
                if (calibrated := calibrate_zcr_threshold(speech_probs, zcr_array)) is not None:
                    final_zcr_threshold = calibrated

            # 合并短块
            logger.debug("【VAD】正在合并短于1秒的语音块……")
            merged_ranges_s = merge_short_segments_adaptive(
                speeches, 
                MAX_SPEECH_DURATION_S
            )
            if len(speeches) != len(merged_ranges_s):
                logger.debug(f"【VAD】原 {len(speeches)} 个语音块已合并为 {len(merged_ranges_s)} 个语音块")
            else:
                logger.debug(f"【VAD】没有需要合并的语音块")
            
            # 初始化结果收集器
            all_chunk_subwords_collection = [[] for _ in range(len(merged_ranges_s))]
            chunk_ranges_log = [] # 记录每一块的时间范围，用于最后的合并

            if args.debug:
                temp_chunk_dir = Path(tempfile.mkdtemp())

            # 遍历一级块
            for i, (unpadded_start_s, unpadded_end_s) in enumerate(merged_ranges_s):
                # --- 准备一级语音块 ---
                # 计算包含静音保护的起止时间
                start_ms = max(0, int(unpadded_start_s * 1000) - args.keep_silence)
                end_ms = min(total_audio_ms, int(unpadded_end_s * 1000) + args.keep_silence)
                
                chunk_ranges_log.append((start_ms / 1000.0, end_ms / 1000.0))
                
                # 一级切片 (使用 Tensor 切片)
                chunk = slice_waveform_ms(waveform, start_ms, end_ms)

                # debug模式下为了检查才导出该块
                if args.debug:
                    # 定义路径对象
                    chunk_audio = silence_padding + audio[start_ms:end_ms] + silence_padding
                    chunk_audio.export(temp_chunk_dir / f"chunk_{i + 1}.wav", format="wav")

                logger.info(
                    f"【VAD】正在处理语音块 {i + 1}/{len(merged_ranges_s)} （该块起止时间：{SRTWriter._format_time(unpadded_start_s)} --> {SRTWriter._format_time(unpadded_end_s)}，时长：{(unpadded_end_s - unpadded_start_s):.2f} 秒）",
                    end="", flush=True,
                    )

                # 判断短块还是长块
                if unpadded_end_s - unpadded_start_s <= MAX_SPEECH_DURATION_S / 3.0:
                    # === 短块 ===
                    logger.info(" --> 短块，直接加入 Batch")
                    # 短块没有运行二次VAD，直接使用一级VAD的原始结束时间
                    vad_chunk_end_times_s.append(unpadded_end_s)
                    
                    # 加入 Batch
                    batch_audio.append(chunk.squeeze(0))
                    # 坐标变换：全局时间 = 一级块偏移 + 相对时间
                    batch_meta.append({
                        'chunk_index': i,
                        'base_offset': start_ms / 1000.0
                    })
                
                else:
                    # === 长块 (二次切分) ===
                    # 运行局部 VAD
                    sub_speeches, sub_speech_probs, sub_frame_duration_s = get_speech_timestamps_onnx(
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
                        sub_speeches = [[0.0, chunk.shape[1] / SAMPLERATE]]

                    # 局部特征计算
                    sub_speech_probs, sub_energy_array, sub_zcr_array = prepare_acoustic_features(
                            chunk, sub_speech_probs, sub_frame_duration_s, args.zcr
                    )
                    
                    # 智能切分
                    nonsilent_ranges_s = [] 
                    for seg in sub_speeches:
                        # seg 是列表 [start, end]
                        # seg[1] 是相对于含 Padding 的 chunk 的时间
                        # start_ms 是 chunk 在原音频中的起始时间（含 keep_silence）
                        # PAD_SECONDS 是 chunk 头部人为添加的静音
                        vad_chunk_end_times_s.append((start_ms / 1000.0) + seg[1] - PAD_SECONDS)
                        nonsilent_ranges_s.extend(global_smart_segmenter(
                                seg[0], 
                                seg[1],
                                sub_speech_probs,      # 局部概率
                                sub_energy_array,      # 局部能量
                                sub_frame_duration_s,  # 局部帧长
                                MAX_SPEECH_DURATION_S,
                                args.vad_threshold,
                                final_zcr_threshold,   # 复用全局计算出的最佳阈值
                                sub_zcr_array,         # 局部 ZCR
                                OVERLAP_S
                            ))
                    logger.debug(
                            f"【VAD】侦测到 {len(sub_speeches)} 个子语音块，保守拆分超过 {MAX_SPEECH_DURATION_S} 秒的部分"
                        )

                    # 再次合并子块
                    refined_sub_speeches = []
                    for seg in merge_short_segments_adaptive(
                            nonsilent_ranges_s, 
                            MAX_SPEECH_DURATION_S
                        ):
                        if (seg[1] - seg[0]) <= MAX_SPEECH_DURATION_S:
                            # 长度正常，直接加入
                            refined_sub_speeches.append(seg)
                        else:
                            # 长度依然超标，执行带重叠的强制硬切分
                            curr = seg[0]
                            while curr < seg[1]:
                                # 计算这一刀的结束点
                                # 如果这一刀切到了末尾之后，就直接用末尾
                                if (next_cut := curr + MAX_SPEECH_DURATION_S) >= seg[1]:
                                    refined_sub_speeches.append([curr, seg[1]])
                                    break
                                
                                # 加入当前硬切分段
                                refined_sub_speeches.append([curr, next_cut])
                                
                                # 移动游标，回退 overlap_s 以形成重叠
                                curr = next_cut - OVERLAP_S

                    if len(refined_sub_speeches) > 1:
                        logger.info(f"    --> Batch 中加入 {len(refined_sub_speeches)} 个子片段：")

                    # 遍历子块加入 Batch
                    for sub_idx, sub_seg in enumerate(refined_sub_speeches):
                        # sub_seg 是相对于 chunk (已含Padding) 的秒数
                        # 提取子块音频并加上静音保护
                        sub_chunk = slice_waveform_ms(chunk, int(sub_seg[0] * 1000), int(sub_seg[1] * 1000))
                        
                        # Debug 时导出临时子块文件
                        if args.debug:
                            (silence_padding + chunk_audio[int(sub_seg[0] * 1000):int(sub_seg[1] * 1000)] + silence_padding).export(
                                (temp_chunk_dir / f"chunk_{i + 1}_sub_{sub_idx + 1}.wav"), format="wav"
                                )

                        if len(refined_sub_speeches) != 1:
                            logger.info(f"第 {i + 1}-{sub_idx + 1} 段 {SRTWriter._format_time(sub_seg[0])} --> {SRTWriter._format_time(sub_seg[1])}，时长：{sub_seg[1] - sub_seg[0]:.2f} 秒")
                        
                        batch_audio.append(sub_chunk.squeeze(0))
                        batch_meta.append({
                            'chunk_index': i,
                            # 最终时间 = 一级块全局偏移 + 二级块在一级块内的偏移 + 识别出的相对时间
                            # 因为 sub_seg[0] 是基于含填充的父chunk计算的，它包含了父chunk头部的 0.5s 静音，必须扣除
                            'base_offset': (start_ms / 1000.0 + sub_seg[0] - PAD_SECONDS)
                        })
                        
                        # 如果 batch 满了，立即执行
                        if len(batch_audio) >= BATCH_SIZE:
                            flush_batch()

                # 每一级块循环末尾，检查 batch 是否满了
                if len(batch_audio) >= BATCH_SIZE:
                    flush_batch()

            # 循环结束后，处理剩余的 batch
            flush_batch()

            # 整理结果结构 (Chunk Results)
            for i, subwords in enumerate(all_chunk_subwords_collection):
                if subwords:
                    chunk_results.append({
                        "subwords": subwords,
                        "range": chunk_ranges_log[i]
                    })
            
            # 去重
            if chunk_results:
                # 所有块处理完后，执行去重，如果只有一个块，无需合并
                logger.debug("【VAD】所有语音块处理完毕，正在去重……")
                if len(chunk_results) > 1:
                    raw_subwords = merge_overlap_dedup(chunk_results)
                else:
                    raw_subwords = chunk_results[0]["subwords"]
                logger.debug(f"【VAD】语音块去重完毕")
            else:
                raw_subwords = []

        # --- 计时结束：核心识别流程 ---
        recognition_end_time = time.time()

        # 如果整个过程下来没有任何识别结果，提前告知用户并退出，避免生成空文件
        if not raw_subwords:
            logger.info("=" * 70)
            logger.info("未识别到任何有效的语音内容")
            return

        logger.info("=" * 70)
        logger.info("正在根据子词和VAD边界生成精确文本片段……")

        all_segments, all_subwords = create_precise_segments_from_subwords(
            raw_subwords, vad_chunk_end_times_s, total_audio_ms / 1000.0, no_remove_punc=args.no_remove_punc
            )

        logger.info("文本片段生成完成")

        if args.refine_tail: # 只在启用精修且map存在时精修
            logger.info("【精修】正在修除每段的尾部静音……")

            for i, segment in enumerate(all_segments):
                # 遍历map，所以需要通过索引i来更新原始的all_segments列表
                segment.end_seconds = refine_tail_end_timestamp(
                    segment.subwords[-1].seconds,
                    segment.end_seconds,
                    speech_probs,
                    frame_duration_s,
                    min(segment.vad_limit, all_segments[i + 1].start_seconds) if i < len(all_segments) - 1 else segment.vad_limit,
                    args.min_silence_duration_ms,
                    args.tail_lookahead_ms,
                    args.tail_safety_margin_ms,
                    args.tail_min_keep_ms,
                    args.tail_percentile,
                    args.tail_offset,
                    args.zcr,
                    final_zcr_threshold,
                    zcr_array,
                    args.tail_zcr_high_ratio
                )

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
        if not file_output_requested:
            logger.info("\n识别结果（完整文本）：")
            logger.info(full_text)
            logger.info("=" * 70)
            logger.info("未指定输出参数，结果将打印至控制台")
            logger.info("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            # Path 对象自带 open 方法
            with (output_path := output_dir / f"{base_name}.txt").open("w", encoding="utf-8") as f: 
                f.write(full_text)
            logger.info(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            with (output_path := output_dir / f"{base_name}.segments.txt").open("w", encoding="utf-8") as f:
                writer = TextWriter(f)
                for segment in all_segments:
                    writer.write(segment)
            logger.info(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            with (output_path := output_dir / f"{base_name}.srt").open("w", encoding="utf-8") as f:
                writer = SRTWriter(f)
                for segment in all_segments:
                    writer.write(segment)
            logger.info(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.segment2vtt:
            with (output_path := output_dir / f"{base_name}.vtt").open("w", encoding="utf-8") as f:
                writer = VTTWriter(f)
                writer.write_header()
                for segment in all_segments:
                    writer.write(segment)
            logger.info(f"文本片段 WebVTT 字幕文件已保存为：{output_path}")

        if args.segment2tsv:
            with (output_path := output_dir / f"{base_name}.tsv").open("w", encoding="utf-8") as f:
                writer = TSVWriter(f)
                writer.write_header()
                for segment in all_segments:
                    writer.write(segment)
            logger.info(f"文本片段 TSV 文件已保存为：{output_path}")

        if args.subword:
            with (output_path := output_dir / f"{base_name}.subwords.txt").open("w", encoding="utf-8") as f:
                for sub in all_subwords:
                    f.write(
                        f"[{SRTWriter._format_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                    )
            logger.info(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            with (output_path := output_dir / f"{base_name}.subwords.srt").open("w", encoding="utf-8") as f:
                for i, sub in enumerate(all_subwords):
                    f.write(f"{i + 1}\n")
                    f.write(f"{SRTWriter._format_time(sub.seconds)} --> {SRTWriter._format_time(sub.end_seconds)}\n")
                    f.write(f"{sub.token}\n\n")

            logger.info(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            with (output_path := output_dir / f"{base_name}.subwords.json").open("w", encoding="utf-8") as f:
                json.dump(
                    [{"token": sub.token, "timestamp": sub.seconds} for sub in all_subwords],
                    f, ensure_ascii=False, indent=4
                )
            logger.info(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            with (output_path := output_dir / f"{base_name}.ass").open("w", encoding="utf-8-sig") as f:

                # 使用 writer 生成标准文件头
                writer = ASSWriter(f)
                writer.write_header()

                for segment in all_segments:
                    karaoke_text = ""
                    for sub in segment.subwords:
                        karaoke_text += f"{{\\k{max(1, round((sub.end_seconds - sub.seconds) * 100))}}}{sub.token}"
                    f.write(f"Dialogue: 0,{ASSWriter._format_time(segment.start_seconds)},{ASSWriter._format_time(segment.end_seconds)},Default,,0,0,0,,{karaoke_text}\n")

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

        if args.debug:
            logger.debug("=" * 70)
            # 先停止监控线程
            try:
                mem_stop_event.set()
                mem_thread.join(timeout=1.0)
            except Exception:
                pass
    
            # 计算并打印内存 / 显存统计
            if MEM_SAMPLES["ram_gb"]:
                max_ram = max(MEM_SAMPLES["ram_gb"])
                ram_mode, freq = _calc_mode(MEM_SAMPLES["ram_gb"])
                logger.debug(f"【内存监控】内存常规用量：{ram_mode:.2f} GB，峰值：{max_ram:.2f} GB（出现 {freq} 次）")
            else:
                logger.debug("【内存监控】未采集到内存占用数据")
    
            # GPU 显存两套信息：采样众数 + PyTorch 精确峰值
            if torch.cuda.is_available():
                # 采样数据的统计
                if MEM_SAMPLES["gpu_gb"]:
                    max_gpu_sample = max(MEM_SAMPLES["gpu_gb"])
                    gpu_mode, freq = _calc_mode(MEM_SAMPLES["gpu_gb"])
                    logger.debug(f"【显存监控】显存常规用量：{gpu_mode:.2f} GB，峰值：{max_gpu_sample:.2f} GB（出现 {freq} 次）")
    
                # PyTorch 记录的峰值（不依赖采样间隔）
                try:
                    
                    if (peak_gpu_bytes := torch.cuda.max_memory_allocated()) > 0:
                        logger.debug(
                            f"【显存监控】PyTorch 记录的显存占用峰值：{peak_gpu_bytes / (1024 ** 3):.2f} GB"
                        )
                except Exception:
                    pass

            logger.debug("=" * 70)
            logger.debug("调试模式已启用，保留临时文件和目录")
            if temp_full_wav_path:
                logger.debug(f"临时 WAV 文件位于: {temp_full_wav_path}")
                open_folder(temp_full_wav_path)
            if temp_chunk_dir:
                logger.debug(f"临时分块目录位于: {temp_chunk_dir}")
                open_folder(temp_chunk_dir) # 调用新函数打开文件夹

if __name__ == "__main__":
    main()