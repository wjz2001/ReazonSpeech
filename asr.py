import argparse
import copy
import heapq
import math
import os
import tempfile
import shutil
import sys
import subprocess
import torch
import torchaudio
import json
import time
import numpy as np
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from pydub import AudioSegment
from omegaconf import open_dict
from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.audio import SAMPLERATE
from reazonspeech.nemo.asr.decode import find_end_of_segment, decode_hypothesis, PAD_SECONDS, SECONDS_PER_STEP, SUBWORDS_PER_SEGMENTS
from reazonspeech.nemo.asr.interface import Segment
from reazonspeech.nemo.asr.writer import (
    SRTWriter,
    ASSWriter,
    TextWriter,
    TSVWriter,
    VTTWriter,
)

# ONNX VAD 依赖
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

def calculate_dynamic_noise_threshold(waveform, samplerate, frame_duration_s):
    """
    计算动态噪声门限。
    使用 10% 分位数代替 min，防止极静帧导致门限失效。
    去除直流偏移。
    """
    frame_length = max(1, int(frame_duration_s * samplerate))

    # 扫描全篇找每帧最大值
    # ceil_mode=True 保证处理尾部数据
    # 展平以便计算分位数
    frame_maxs_flat = torch.nn.functional.max_pool1d(
        # 去除直流偏移 (Center the waveform)
        # 取绝对值
        (waveform - waveform.mean(dim=-1, keepdim=True)).abs().unsqueeze(0), 
        kernel_size=frame_length, 
        stride=frame_length,
        ceil_mode=True
    ).view(-1) # 输出形状 [1, 1, NumFrames]

    if frame_maxs_flat.numel() > 0:
        # 使用 10% 分位数 (Quantile) 代替 Min
        # 这能有效忽略偶尔出现的数字静音 (0值)，找到真正的“底噪层”
        noise_floor = torch.quantile(frame_maxs_flat.float(), 0.10).item()
        
        # 乘以 2 倍作为安全门限，并设定 1e-4 的硬下限
        return max(noise_floor * 2, 1e-4)
    
    return 0.005 # 兜底

def calculate_frame_cost(frame_idx, speech_probs, waveform, samplerate, frame_duration_s, vad_threshold, noise_threshold, use_zcr, zcr_threshold):
    """计算单帧作为切分点的代价（代价越低越适合切分）。
       边界情况返回高代价。
       ZCR 判定使用 max(|x1|, |x2|) > th。
    """
    # 基础 VAD 概率成本
    prob = speech_probs[frame_idx]
    if prob > vad_threshold:
        return 100.0 # 惩罚高概率点，强硬约束,确信是语音的地方绝不切
        
    # 能量成本 (避免切在清音/呼吸声上)
    # 边界检查
    chunk = waveform[
        0, 
        int(frame_idx * frame_duration_s * samplerate) : 
        min(int((frame_idx + 1) * frame_duration_s * samplerate), waveform.shape[1])
    ]
    # [修复] 边界情况不能返回0 (会被误认为是极佳切点)，应返回中性偏高代价
    if chunk.numel() <= 1:
        return prob + 10.0 

    # --- 去直流 ---
    chunk = chunk - chunk.mean()

    # --- 能量成本 ---
    energy_cost = min(
        torch.sqrt(torch.mean(chunk**2)).item(),
        0.5
        ) * 10.0

    # --- 动态抗噪过零率 (Robust ZCR) ---
    # 逻辑：(过零) AND (穿越点的幅度足够大，不仅是微小抖动)
    # 标准做法：check max(abs(x[n]), abs(x[n+1])) > th
    zcr_cost = 0.0
    # 符号相反
    # 任意一侧幅度超过噪声门限 (表示穿越了噪声带)
    if use_zcr:
        zcr_rate = compute_chunk_zcr(chunk, noise_threshold)

        if zcr_rate > zcr_threshold: 
            zcr_cost = zcr_rate * 50.0 

    # 局部平滑度 (斜率)
    slope_cost = 0.0
    if 0 < frame_idx < len(speech_probs) - 1:
        slope_cost = abs(speech_probs[frame_idx + 1] - speech_probs[frame_idx - 1]) * 2.0

    return prob + energy_cost + slope_cost + zcr_cost

def calibrate_zcr_threshold(waveform, speech_probs, noise_threshold, frame_duration_s, samplerate):
    """
    自适应 ZCR 阈值校准函数 (Method A: Percentile).
    """
    print("正在校准自适应 ZCR 阈值……")
    
    # 结果是一个 Tensor，长度与 speech_probs 基本一致
    zcr_array = compute_global_zcr_vectorized(waveform, samplerate, frame_duration_s, noise_threshold).numpy()

    # 对齐长度 (因为池化 padding 可能导致长度差 1)
    min_len = min(len(zcr_array), len(speech_probs))
    zcr_array = zcr_array[:min_len]
    vad_array = speech_probs[:min_len]

    # 分群
    # 资料建议：VAD > 0.8 为浊音(voiced)，VAD < 0.2 为非语音/清音背景(unvoiced)
    # 注意：这里的 unvoiced 其实包含静音和清音，但这不影响我们找下界
    voiced_mask = vad_array > 0.8
    unvoiced_mask = vad_array < 0.2
    
    if np.sum(voiced_mask) < 10 or np.sum(unvoiced_mask) < 10:
        print("浊音/静音样本区分度不足，回退默认值")
        return None

    # 统计分位数
    # 资料建议：τ_v = P90 (浊音上限), τ_u = P10 (非语音下限)
    tau_v = np.percentile(zcr_array[voiced_mask], 90)
    tau_u = np.percentile(zcr_array[unvoiced_mask], 10) # 实际上这里应该找高ZCR群体的下限，但资料里的 P10 是指 unvoiced 分布的左侧
    
    # 计算自适应阈值
    adaptive_th = 0.5 * (tau_v + tau_u)
    
    print(f"统计结果：浊音 P90={tau_v:.3f}，非语音 P10={tau_u:.3f}，计算阈值={adaptive_th:.3f}")

    # 安全检查：如果计算出的阈值太极端，说明数据分布有问题
    if 0.05 < adaptive_th < 0.5:
        return adaptive_th
    else:
        print("计算阈值超出安全范围（0.05~0.5），回退默认值")
        return None

def compute_chunk_zcr(chunk, noise_threshold):
    """
    【核心原子函数】计算单个音频片段的 ZCR 率。
    被 calculate_frame_cost, refine_tail_end_timestamp 调用。
    确保所有地方的算法完全一致。
    """
    if chunk.numel() <= 1:
        return 0.0

    # 去直流 (DC Removal)
    chunk = chunk - chunk.mean()

    # 计算过零且幅度足够大
    # (穿越零点) AND (两点中任意一点幅度 > 噪声门限)
    # 返回比率 (0.0 - 1.0)   
    # 找出所有过零点
    zero_crossings = (chunk[:-1] * chunk[1:] < 0)
    
    # 找出所有能量足够大的点对
    sufficient_energy = ((chunk[:-1].abs() > noise_threshold) | (chunk[1:].abs() > noise_threshold))
    
    # 对两个布尔张量进行逻辑与操作，然后求和
    valid_crossings = torch.logical_and(zero_crossings, sufficient_energy).sum().item()
    
    # 计算比率
    return valid_crossings / (chunk.numel() - 1)

def compute_global_zcr_vectorized(waveform, samplerate, frame_duration_s, noise_threshold):
    """
    【向量化全篇计算】使用 PyTorch 卷积/池化操作一次性计算全篇 ZCR，速度比 for 循环快几百倍
    """
    # 预处理：去直流
    # waveform: (1, T)
    waveform_centered = waveform - waveform.mean(dim=-1, keepdim=True)
    
    # 等价于 abs > threshold
    is_loud = (waveform_centered.abs() > noise_threshold)
    
    # 使用 Average Pooling 计算每一帧的 ZCR 率
    # 这相当于对每一帧做 sum() / count
    frame_length = int(frame_duration_s * samplerate)
    
    # 必须保证 stride 和 frame_length 一致，才能和 VAD 帧对齐
    # 输出形状: (Num_Frames, )
    return torch.nn.functional.avg_pool1d(
        # 向量化计算所有采样点的过零情况 (1, T-1)
        # 等价于 chunk[:-1] * chunk[1:]
        # 逻辑或: 左边响 OR 右边响
        # 得到“有效过零点”的布尔矩阵 (1, T-1)
        (((waveform_centered[:, :-1] * waveform_centered[:, 1:]) < 0) & (is_loud[:, :-1] | is_loud[:, 1:])).float().unsqueeze(0), # 需要 (N, C, L)
        kernel_size=frame_length,
        stride=frame_length,
        ceil_mode=True # 保证处理尾部
    ).view(-1) # 使用 .view(-1) 将输出展平为一维向量

def create_precise_segments_from_subwords(
    all_subwords, vad_chunk_end_times_s = None, no_chunk = False, debug = False
):
    # 初始化一个新的空列表
    if vad_chunk_end_times_s is None:
        vad_chunk_end_times_s = []

    if not all_subwords:
        return [], [], []

    # 预处理：为每个子词计算其VAD块的结束时间
    if no_chunk or not vad_chunk_end_times_s:
        subword_vad_end_times = [float("inf")] * len(all_subwords)
    else:
        subword_vad_end_times = []
        vad_cursor = 0
        for sub in all_subwords:
            # 移动游标找到当前子词所属的VAD块
            while (vad_cursor < len(vad_chunk_end_times_s) and
                   sub.seconds > vad_chunk_end_times_s[vad_cursor]):
                vad_cursor += 1
            
            subword_vad_end_times.append(
                vad_chunk_end_times_s[vad_cursor] if vad_cursor < len(vad_chunk_end_times_s) else float("inf")
            )

    # 计算子词平均持续时长和子词结束时间
    average_duration = SECONDS_PER_STEP
    durations = []

    if len(all_subwords) > 1:
        durations = [
            all_subwords[i + 1].seconds - all_subwords[i].seconds
            for i in range(len(all_subwords) - 1)
        ]

    positive_durations = [d for d in durations if d > 0]
    if positive_durations:
        average_duration = sum(positive_durations) / len(positive_durations)

    # --- 动态计算停顿阈值 ---
    pause_threshold = float('inf') # 默认阈值为无穷大，即默认禁用此功能
    # 只有当有足够多的数据点（例如超过20个子词间隔）时，才计算并启用阈值
    MIN_SAMPLES_FOR_THRESHOLD = SUBWORDS_PER_SEGMENTS * 2 
    if len(durations) > MIN_SAMPLES_FOR_THRESHOLD and positive_durations:
        pause_threshold = np.percentile(positive_durations, 95) 

    subword_end_seconds = []
    for i, sub in enumerate(all_subwords):
        # 统一计算潜在的结束时间
        # 直接从缓存中读取VAD边界，不再需要游标
        end_time = min(
            all_subwords[i + 1].seconds if i < len(all_subwords) - 1
            else sub.seconds + average_duration,
            subword_vad_end_times[i]
        )
    
        # 最后修正：确保结束时间总是在开始时间之后
        subword_end_seconds.append(max(end_time, sub.seconds + SECONDS_PER_STEP))

    # 使用 VAD 优先、find_end_of_segment 其次、基于语速/停顿边界补充的逻辑生成片段
    all_segments = []
    segment_to_subword_map = []
    start = 0
    while start < len(all_subwords):
        # 获取当前需要遵守的 VAD 边界的时间戳
        # 在 subword_vad_end_times 中查找。
        # bisect_right 会返回在保持列表有序的前提下，插入subword_vad_end_times[start]的最右位置
        # lo=start 是为了优化性能，告诉函数从 start 位置开始往后找，不要从头找。
        next_group_start_idx = bisect_right(subword_vad_end_times, subword_vad_end_times[start], lo=start)

        # 这一组的最后一个索引，就是下一组起始索引减 1
        # 在VAD确定的范围内，使用启发式规则寻找
        # 确保更早的断点不会超过 VAD 的硬边界
        end_idx = min(next_group_start_idx - 1, find_end_of_segment(all_subwords, start))

        # 基于语速/停顿的边界 (最低, 作为补充)
        # 只有当前片段依然很长 (例如超过 N 个子词)，并且没有被前两种规则切分时，才考虑此规则
        pause_split_indices = [] 
        # 在 find_end_of_segment 划定的长片段内部，寻找所有显著的停顿点
        if (end_idx - start + 1) > MIN_SAMPLES_FOR_THRESHOLD:
            pause_split_indices = [
                i for i in range(start, end_idx)
                if durations[i] > pause_threshold # 记录所有停顿点的索引
            ]
        
        # 将片段的起始点和最终的结束点加入，形成完整的切分区间
        # start - 1 是为了让循环 for i in range(len(all_split_points) - 1): 能够从 start 索引开始处理第一个子片段
        all_split_points = [start - 1] + pause_split_indices + [end_idx]

        if pause_split_indices and debug:
            # --- 构建带有 || 分隔符的完整片段预览 ---
            preview_parts = []
            # 将切分点索引放入一个集合中，以提高查找效率，set使平均时间复杂度从 O(N) 降低到 O(1)
            split_points_set = set(pause_split_indices)
            # 遍历当前长片段（从 start 到 end_idx）的所有子词
            for i in range(start, end_idx + 1):
                # 添加子词本身
                preview_parts.append(all_subwords[i].token)
                # 如果当前子词的索引是一个切分点，则在它后面添加分隔符
                if i in split_points_set:
                    preview_parts.append(" || ")
            # 将所有部分连接成一个字符串
            print(f"在 {SRTWriter._format_time(all_subwords[start].seconds)} 开始的长片段：“{''.join(preview_parts)}”内找到显著停顿点")
        
        # 遍历所有切分点，生成多个片段
        for i in range(len(all_split_points) - 1):
            segment_start_idx = all_split_points[i] + 1
            segment_end_idx = all_split_points[i + 1]

            # 提取当前片段的子词和索引
            current_subwords = all_subwords[segment_start_idx:segment_end_idx + 1]

            # 创建 Segment 对象
            text = "".join(s.token for s in current_subwords)
    
            if text:
                new_segment = Segment(
                    start_seconds=current_subwords[0].seconds,
                    end_seconds=subword_end_seconds[segment_end_idx],
                    text=text
                )
                all_segments.append(new_segment)
                segment_to_subword_map.append(
                    (
                        new_segment,
                        current_subwords,
                        range(segment_start_idx, segment_end_idx + 1)
                    )
                )

        start = end_idx + 1

    return all_segments, segment_to_subword_map, subword_end_seconds

def find_optimal_splits_dp_strict(candidates, num_cuts, start_frame, end_frame, frame_duration_s, max_duration_s):
    """严格硬约束的动态规划算法。"""
    n = len(candidates)
    max_frames = int(max_duration_s / frame_duration_s)
    min_frames = int(3.0 / frame_duration_s) 
    
    dp = [[(float('inf'), []) for _ in range(n)] for _ in range(num_cuts + 1)]

    # 第一刀
    for j in range(n):
        cut_frame = candidates[j]['frame']
        segment_len = cut_frame - start_frame
        if min_frames <= segment_len <= max_frames:
            dp[1][j] = (candidates[j]['cost'], [cut_frame])

    # 后续每一刀
    for i in range(2, num_cuts + 1):
        for j in range(i - 1, n):
            curr_frame = candidates[j]['frame']
            for k in range(i - 2, j):
                prev_frame = candidates[k]['frame']
                segment_len = curr_frame - prev_frame
                
                if min_frames <= segment_len <= max_frames:
                    prev_cost, prev_path = dp[i-1][k]
                    if prev_cost != float('inf'):
                        new_total_cost = prev_cost + candidates[j]['cost']
                        if new_total_cost < dp[i][j][0]:
                            dp[i][j] = (new_total_cost, prev_path + [curr_frame])

    # 最终检查
    final_min_cost = float('inf')
    final_best_path = []
    for j in range(num_cuts - 1, n):
        last_cut_frame = candidates[j]['frame']
        last_segment_len = end_frame - last_cut_frame
        if min_frames <= last_segment_len <= max_frames:
            cost, path = dp[num_cuts][j]
            if cost < final_min_cost:
                final_min_cost = cost
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

# --- ONNX VAD 辅助函数 ---
def get_speech_timestamps_onnx(
    waveform,
    onnx_session,
    threshold,
    sampling_rate,
    neg_threshold, #语音结束阈值（低阈值）
    min_speech_duration_ms, #最小语音段时长
    min_silence_duration_ms #多长静音视为真正间隔
):
    """使用 ONNX 模型和后处理来获取语音时间戳"""
    # ONNX 模型需要特定的输入形状

    waveform = waveform.unsqueeze(0)

    # 运行 ONNX 模型
    #  (1, T, 7), 取 [0] 变成 (T, 7)
    logits = torch.from_numpy(
        onnx_session.run
        (
            None,
            {"input_values": waveform.numpy()}
        )[0]
        )[0]
    # 聚合语音组能量 (Index 1-6)
    # torch.logsumexp 默认 keepdim=False，输入 (T,6) -> 输出 (T)
    # 获取静音能量 (Index 0) -> (T)
    # 竞争: 语音 - 静音，再Sigmoid
    # 转回 Numpy 以便进入 Python 循环
    speech_probs = torch.sigmoid(torch.logsumexp(logits[:,1:], dim=1) - logits[:, 0]).numpy()

    frame_duration_s = (waveform.shape[2] / sampling_rate) / logits.shape[0]

    speeches = []
    current_speech = None
    triggered = False
    temp_silence_start_frame = None

    for i, prob in enumerate(speech_probs):
        # --- 语音开始逻辑（高阈值） ---
        if not triggered:
            if prob >= threshold:
                triggered = True
                current_speech = {'start': i * frame_duration_s}
                temp_silence_start_frame = None
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
            if (i - temp_silence_start_frame) >= max(1, int(min_silence_duration_ms / (frame_duration_s * 1000.0))):
                end_time = temp_silence_start_frame * frame_duration_s
                if (end_time - current_speech["start"]) * 1000.0 >= min_speech_duration_ms:
                    current_speech["end"] = end_time
                    speeches.append(current_speech)
                    # 收尾并准备下一段
                triggered = False
                current_speech = None
                temp_silence_start_frame = None

    # --- 处理结尾残留的语音段 ---
    if triggered and current_speech:
        end_time = len(speech_probs) * frame_duration_s
        if (end_time - current_speech['start']) * 1000.0 >= min_speech_duration_ms:
            current_speech['end'] = end_time
            speeches.append(current_speech)

    # 完整的语音概率数组 speech_probs 和每一帧的持续时间 frame_duration_s
    return speeches, speech_probs, frame_duration_s

def global_smart_segmenter(long_segment, speech_probs, waveform, samplerate, frame_duration_s, max_duration_s, vad_threshold, noise_threshold, use_zcr, zcr_threshold, overlap_s):
    """
    入口函数：分析一个长 VAD 段，返回切割好的子段列表。
    """
    start_s = long_segment['start']
    end_s = long_segment['end']

    # 计算必须切几刀
    num_cuts = max(0, math.ceil((end_s - start_s) / max_duration_s) - 1)
    if num_cuts == 0:
        return [long_segment]

    start_frame = int(start_s / frame_duration_s)
    end_frame = int(end_s / frame_duration_s)

    # 生成候选点 (筛选 VAD 概率 < max(0.15, vad_threshold - 0.1) 的局部低点)
    candidates = []
    search_start = start_frame + int(1.0 / frame_duration_s)
    search_end = end_frame - int(1.0 / frame_duration_s)

    # 边界保护
    search_start = max(0, search_start)
    search_end = min(len(speech_probs), search_end)

    # 提取搜索区间的概率
    if search_end > search_start + 2:
        segment_probs = speech_probs[search_start:search_end]
        
        # 利用 numpy 寻找局部极小值且概率 < max(0.15, vad_threshold - 0.1) 的点
        # 逻辑：当前点 <= 前一点 AND 当前点 <= 后一点 AND 当前点 < max(0.15, vad_threshold - 0.1)
        # 注意：切片操作 [1:-1] 对应原数组的索引 i, [:-2] 对应 i-1, [2:] 对应 i+1
        
        # 这里的 segment_probs 是一个切片视图
        p_curr = segment_probs[1:-1]
        p_prev = segment_probs[:-2]
        p_next = segment_probs[2:]

        # 生成布尔掩码
        mask = (p_curr < max(0.15, vad_threshold - 0.1)) & (p_curr <= p_prev) & (p_curr <= p_next)
        
        # 获取满足条件的相对索引
        relative_indices = np.where(mask)[0]
        
        # 还原绝对索引：+1 是因为 p_curr 从切片的第1个元素开始，+search_start 是切片偏移
        candidate_indices = relative_indices + 1 + search_start

        # 遍历筛选出的索引计算成本 (此时循环次数大大减少)
        for idx in candidate_indices:
            cost = calculate_frame_cost(idx, speech_probs, waveform, samplerate, frame_duration_s, vad_threshold, noise_threshold, use_zcr, zcr_threshold)
            candidates.append({'frame': idx, 'cost': cost})
    
    # 执行 DP
    best_cuts_frames = []
    if len(candidates) >= num_cuts:
        best_cuts_frames = find_optimal_splits_dp_strict(
            candidates, num_cuts, start_frame, end_frame, frame_duration_s, (max_duration_s - overlap_s)
        )
    
    # 构建结果
    if not best_cuts_frames:
        # 如果找不到最佳切点，直接返回原始的长段落，把切分决策交给后续流程
        return [long_segment]
        
    final_segments = []
    current_start_s = start_s
    for cut_frame in best_cuts_frames:
        split_s = cut_frame * frame_duration_s
        final_segments.append({'start':current_start_s, 'end': split_s})
        current_start_s = split_s - overlap_s
    final_segments.append({'start':current_start_s, 'end': end_s})

    return final_segments

def merge_overlapping_intervals(intervals):
    """仅合并物理上重叠的区间。"""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        
        if current_start < merged[-1][1]: 
            merged[-1][1] = max(merged[-1][1], current_end)
        else:
            merged.append([current_start, current_end])
            
    return merged

def merge_overlap_dedup(chunk_subwords_list, chunk_ranges_s):
    """
    合并来自不同块的子词列表，并处理重叠区域的重复子词。

    Args:
        chunk_subwords_list (list[list[Subword]]): 每个块识别出的子词列表的列表。
        chunk_ranges_s (list[tuple[float, float]]): 每个块在原始音频中的（开始，结束）时间。

    Returns:
        list[Subword]: 合并并去重后的最终子词列表。
    """
    def _overlap_interval(a, b):
        # a, b 是 (start_s, end_s)
        s, e = max(a[0], b[0]), min(a[1], b[1])
        return (s, e) if e > s else (None, None)

    def _is_conflict(subword1, subword2):
        """判断两个子词是否冲突（token相同且时间相近）"""
        return (subword1.token_id == subword2.token_id and
                abs(subword1.seconds - subword2.seconds) <= SECONDS_PER_STEP * 1.5)
                
    if not chunk_subwords_list:
        return []
    # 初始 merged 列表就是第一个块的结果，它已经是排序的
    merged = chunk_subwords_list[0]

    for i in range(1, len(chunk_subwords_list)):
        curr_subs = chunk_subwords_list[i]
        s_ov, e_ov = _overlap_interval(chunk_ranges_s[i-1], chunk_ranges_s[i])

        # 如果没有重叠，直接使用高效的归并
        if s_ov is None:
            # heapq.merge 要求输入是已排序的，我们的数据满足这个条件
            merged = list(heapq.merge(merged, curr_subs, key=lambda x: x.seconds))
            continue

        # --- 有重叠区域，执行更复杂的合并去重逻辑 ---
        
        # 将旧结果（merged）和新结果（curr_subs）划分为重叠区和非重叠区
        prev_non_overlap = [s for s in merged if not (s_ov <= s.seconds <= e_ov)]
        prev_overlap = [s for s in merged if s_ov <= s.seconds <= e_ov]
        
        curr_non_overlap = [s for s in curr_subs if not (s_ov <= s.seconds <= e_ov)]
        curr_overlap = [s for s in curr_subs if s_ov <= s.seconds <= e_ov]

        # 处理核心的重叠区域
        final_overlap = []
        # 使用集合来存储那些因为冲突而被“淘汰”的旧 subword 的 id (内存地址)
        discarded_old_ids = set()
        
        # 为了高效查找，将 prev_overlap 的 token 存入字典
        # 键是 token_id，值是该 token_id 对应的所有 subword 列表
        prev_overlap_map = defaultdict(list)
        for sub in prev_overlap:
            prev_overlap_map[sub.token_id].append(sub)
            
        # 遍历新的重叠区，决定每个 subword 的去留
        for s_new in curr_overlap:
            for s_old in prev_overlap_map.get(s_new.token_id, []):
                if _is_conflict(s_new, s_old):
                    # 我们标记旧的 subword 为 None，表示它可能被替换
                    # 记录要丢弃的旧 subword 的 id
                    discarded_old_ids.add(id(s_old))
                    break
            
            # 将新的 subword 添加到重叠区（冲突的旧 subword 会在后面被过滤掉）
            final_overlap.append(s_new)

        # 把旧的、未被标记为删除的 subword 添加回来
        for s_old in prev_overlap:
            if id(s_old) not in discarded_old_ids:
                final_overlap.append(s_old)
        
        # 对刚刚合并的重叠区进行一次排序（这部分数据量很小）
        final_overlap.sort(key=lambda x: x.seconds)

        # 3. 使用 heapq.merge 高效合并所有部分
        # 此时 prev_non_overlap, final_overlap, curr_non_overlap 都是已排序列表
        merged = list(heapq.merge(prev_non_overlap, final_overlap, curr_non_overlap, key=lambda x: x.seconds))

    return merged

def open_folder(path):
    """根据不同操作系统，使用默认的文件浏览器打开指定路径的文件夹"""
    if not path.is_dir():
        print(f"【调试模式】目录 '{path}' 不存在，无法自动打开")
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and other Unix-like
            try:
                subprocess.run(["xdg-open", path])
            except FileNotFoundError:
                print(f"【警告】无法自动打开文件夹，系统可能缺少 xdg-open")
        print(f"【调试模式】已自动为您打开临时分块目录：{path}")
    except Exception as e:
        print(f"【警告】尝试自动打开目录失败：{e}，请手动访问：{path}")

def refine_tail_end_timestamp(
    last_token_start_s,   # 该段最后一个子词的开始时间
    rough_end_s,          # 原段结束时间（来自你的组段逻辑）
    speech_probs,         # VAD 概率序列（get_speech_timestamps_onnx 返回的第二项）
    waveform, samplerate, noise_threshold,
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
    debug = False
):
    # 仅在“最后子词之后”的尾窗内搜索
    start_idx = int(last_token_start_s / frame_duration_s)
    # 尾部最多多看 1s
    end_idx = min(len(speech_probs), int(np.ceil(min(max_end_s, rough_end_s + 1) / frame_duration_s)))
    if end_idx - start_idx <= 3:
        return min(rough_end_s, max_end_s)

    # 短窗平滑
    p = np.asarray(speech_probs[start_idx:end_idx], dtype=np.float32)
    if len(p) >= 5:
        p_smooth = np.convolve(p, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    else:
        p_smooth = p

    # 局部自适应阈值percentile + offset
    dyn_tau = float(np.percentile(p_smooth, percentile)) + offset

    # 连续静音 + 滞回
    min_silence_frames = max(1, int(min_silence_duration_ms / 1000.0 / frame_duration_s))
    for i in range(0, len(p_smooth) - min_silence_frames):
        if np.all(p_smooth[i : i + min_silence_frames] < dyn_tau):
            # --- ZCR 校验 ---
            w_start = int((start_idx + i) * frame_duration_s * samplerate)
            w_end = min(w_start + int(0.05 * samplerate), waveform.shape[1])
            chunk = waveform[0, w_start:w_end]
            
            # 只有当 chunk 足够长(>10ms)才进行 ZCR 计算，避免除零或无效计算
            if use_zcr and chunk.numel() > int(0.01 * samplerate):
                # ZCR 判定
                zcr_value = compute_chunk_zcr(chunk, noise_threshold)
                if zcr_value > zcr_threshold:
                    if debug:
                        print(f"【ZCR】检测到清音！时间点：{SRTWriter._format_time((start_idx + i) * frame_duration_s)}，ZCR值：{zcr_value:.3f}（阈值: {zcr_threshold}）")
                    continue # 包含清音，直接在此处 continue

            if np.all(
                p_smooth[
                    i + min_silence_frames:min(len(p_smooth),
                    i + min_silence_frames + max(0, int(lookahead_ms / 1000.0 / frame_duration_s)))
                    ] < dyn_tau + 0.02
                ):  # 滞回
                return min(
                    max(
                        (start_idx + i) * frame_duration_s + safety_margin_ms / 1000.0,
                        last_token_start_s + min_tail_keep_ms / 1000.0,
                    ),
                    max_end_s
                )

    # 兜底：没找到稳定静音，保留原结束（不超过 VAD 上限）
    return min(rough_end_s, max_end_s)

def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型识别语音，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题"
    )

    # 必须：音频/视频文件路径
    parser.add_argument(
        "input_file",
        help="需要识别语音的音频/视频文件路径",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，处理结束后不删除临时文件，并自动打开临时分块目录",
    )

    def beam_size_type(value):
        """自定义 argparse 类型函数，用于验证 beam_size 的范围"""
        try:
            ivalue = int(value)
            if not (4 <= ivalue <= 64):
                raise ValueError() # 故意引发一个值错误，被下面的 except 捕获
        except ValueError:
            # 如果无法转换为整数，则引发错误
            raise argparse.ArgumentTypeError(f"beam_size 必须为 4 到 64 之间的整数，您提供的 {value} 不正确")
        # 如果验证通过，返回整数值
        return ivalue

    parser.add_argument(
            "--beam",
            type=beam_size_type,
            default=4,
            help="设置集束搜索（Beam Search）宽度，范围为 4 到 64 之间的整数，默认值是 4 ，更大的值可能更准确但更慢",
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
        help="【VAD】判断为语音的置信度阈值（0-1）",
    )
    # VAD 结束阈值（双阈值滞回）
    parser.add_argument(
        "--vad_end_threshold",
        type=float,
        default=None,
        help="【VAD】判断为语音结束后静音的置信度阈值（0-1），默认值是vad_threshold的值减去0.15",
    )
    parser.add_argument(
        "--min_speech_duration_ms",
        type=float,
        default=100,
        help="【VAD】移除短于此时长（毫秒）的语音块",
    )
     # 静音最小时长，用于智能合并/分段
    parser.add_argument(
        "--min_silence_duration_ms",
        type=float,
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
        default=30.0,
        help="【精修】自适应阈值（0-100）。值越大，越容易将高概率语音区域判为静音",
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
    # 校验 --no-chunk 和 VAD 参数的冲突
    if args.no_chunk:
    # 只有当用户修改了默认值，但加了 --no-chunk 时才报错
        # 定义所有 VAD 相关的参数名
        vad_params = [
            "vad_threshold",
            "vad_end_threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "keep_silence"
        ]
        
        # 动态检查当前参数值是否等于定义时的 default 值
        # getattr(args, p) 获取当前解析到的值
        # parser.get_default(p) 获取定义时的默认值
        for p in vad_params:
            if getattr(args, p) != parser.get_default(p):
                parser.error(f"【参数错误】已添加 --no-chunk，不能设置参数 --{p}")

        # 校验 --no-chunk 和 --zcr 的冲突
        if args.zcr:
            parser.error("【参数冲突】使用--zcr（过零率检测）功能必须开启 VAD，因此不能与 --no-chunk 同时使用")

        # 校验 --no-chunk 和 --refine-tail 的冲突
        if args.refine_tail:
            parser.error("【参数冲突】使用--refine-tail（段尾精修）功能必须开启 VAD，因此不能与 --no-chunk 同时使用")

    # 校验未使用 --zcr 却指定了相关参数的情况
    # 只有当用户修改了默认值（即想要调整精修参数），但忘了加 --zcr 开关时才报错
    if not args.zcr:
        if getattr(args, "zcr_threshold") != parser.get_default("zcr_threshold"):
            parser.error(f"【参数错误】未添加 --zcr，不能设置参数 --zcr_threshold")
        if args.auto_zcr:
            parser.error(f"【参数错误】未添加 --zcr，不能设置参数 --auto_zcr")

    # 校验未使用 --refine-tail 却指定了相关参数的情况
    # 只有当用户修改了默认值（即想要调整精修参数），但忘了加 --refine-tail 开关时才报错
    if not args.refine_tail:
        # 定义所有精修相关的参数名
        tail_params = [
            "tail_percentile",
            "tail_offset",
            "tail_lookahead_ms",
            "tail_safety_margin_ms",
            "tail_min_keep_ms"
        ]
        
        # 动态检查当前参数值是否等于定义时的 default 值
        # getattr(args, p) 获取当前解析到的值
        # parser.get_default(p) 获取定义时的默认值
        for p in tail_params:
            if getattr(args, p) != parser.get_default(p):
                parser.error(f"【参数错误】未添加 --refine-tail，不能设置参数 --{p}")

    if not args.no_chunk:
        if onnxruntime is None:
            print("【错误】缺少 onnxruntime，请运行 'pip install onnxruntime'")
            return

        local_onnx_model_path = Path(__file__).resolve().parent / "models" / "model_quantized.onnx"

        if not local_onnx_model_path.exists():
            print(
                f"【错误】Pyannote-segmentation-3.0 模型未在 '{local_onnx_model_path}' 中找到"
            )
            print("请下载 model_quantized.onnx 并放入 models 文件夹")
            return

        # ==== VAD 阈值参数校验和默认 ====
        if args.vad_end_threshold is None:
            args.vad_end_threshold = max(0, args.vad_threshold - 0.15)
        if not (0.0 <= args.vad_threshold <= 1.0):
            parser.error(f"vad_threshold 必须在（0-1）范围内，当前值错误")
        if not (0.0 <= args.vad_end_threshold <= 1.0):
            parser.error(f"vad_end_threshold 必须在（0-1）范围内，当前值错误")
        if args.vad_end_threshold > args.vad_threshold:
            parser.error(
                f"vad_end_threshold 不能大于 vad_threshold"
            )
    
        if args.refine_tail and not (0.0 <= args.tail_percentile <= 100.0):
            parser.error(f"tail_percentile 必须在（0-100）范围内，当前值错误")

    # --- 准备路径和临时文件 ---
    input_path = Path(args.input_file)
    output_dir = input_path.parent.resolve()
    base_name = input_path.stem

    # 创建一个临时的 WAV 文件
    # delete=False 确保在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = Path(temp_wav_file.name)
    temp_wav_file.close()  # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = None

    # --- 执行核心的语音识别流程 ---
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件……")
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(SAMPLERATE)
        audio.export(temp_wav_path, format="wav")
        print("转换完成")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型……")
        asr_model_load_start = time.time()  # <--- 计时开始
        model = load_model()
        asr_model_load_end = time.time()  # <--- 计时结束
        print(f"模型加载完成")

        # 获取模型最大允许输入语音块长度
        MAX_SPEECH_DURATION_S = model.cfg.train_ds.max_duration

        if args.beam != model.cfg.decoding.beam.beam_size:
            # 使用深拷贝来完全复制原始配置，确保它不受任何后续修改的影响
            original_decoding_cfg = copy.deepcopy(model.cfg.decoding)
            
            # 同样使用深拷贝创建新配置，确保它与原始配置完全独立
            new_decoding_cfg = copy.deepcopy(model.cfg.decoding)
            with open_dict(new_decoding_cfg): # 使用 open_dict 使配置可修改
                new_decoding_cfg.beam.beam_size = args.beam
            
            # 在所有识别任务开始前，应用这个新的解码策略
            print(f"正在应用新的解码策略：集束搜索宽度为 {args.beam} ……")
            model.change_decoding_strategy(new_decoding_cfg)

        # --- 逐块识别并校正时间戳 ---
        all_subwords = []
        vad_chunk_end_times_s = []  # 用于存储VAD块的结束时间

        # --- 计时开始：核心识别流程 ---
        recognition_start_time = time.time()

        # 使用 pydub 创建静音段用于填充
        # PAD_SECONDS 是以秒为单位，pydub 需要毫秒
        silence_padding = AudioSegment.silent(duration=PAD_SECONDS * 1000, frame_rate=SAMPLERATE)

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("未使用VAD，一次性处理整个文件……")
            (silence_padding + audio + silence_padding).export(temp_wav_path, format="wav")
            hyp, _ = model.transcribe(
                [str(temp_wav_path)],
                return_hypotheses=True,
                verbose=True,
            )
            if hyp and hyp[0]:
                ret = decode_hypothesis(model, hyp[0])
                all_subwords = ret.subwords

        else:
            temp_chunk_dir = Path(tempfile.mkdtemp())
            print("正在从本地路径加载 Pyannote-segmentation-3.0 模型……")
            vad_model_load_start = time.time()  # <--- 计时开始
            onnx_session = onnxruntime.InferenceSession(
                str(local_onnx_model_path), providers=["CPUExecutionProvider"]
            )
            vad_model_load_end = time.time()  # <--- 计时结束
            print(f"Pyannote-segmentation-3.0 模型加载完成，将在 CPU 上运行")
            print("正在使用 Pyannote-segmentation-3.0 侦测语音活动……")

            # ==== 先做 VAD（双阈值 + 静音 + min_speech），不在此强制 30s ====
            # ==== 获取VAD分块，并缓存完整的语音概率 ====
            waveform = torchaudio.load(temp_wav_path)[0]
            speeches, speech_probs, frame_duration_s = get_speech_timestamps_onnx(
                waveform,
                onnx_session,
                threshold=args.vad_threshold,
                sampling_rate=SAMPLERATE,
                neg_threshold=args.vad_end_threshold,
                min_speech_duration_ms=int(args.min_speech_duration_ms),
                min_silence_duration_ms=int(args.min_silence_duration_ms)
            )

            if not speeches:
                print("【警告】未侦测到语音活动")
                recognition_end_time = time.time()
                return

            print("正在计算全局背景底噪水平……")    
            global_noise_threshold = calculate_dynamic_noise_threshold(waveform, SAMPLERATE, frame_duration_s)
            print(f"全局底噪门限已设定为：{global_noise_threshold:.6f}")

            # 确定最终使用的 ZCR 阈值
            final_zcr_threshold = args.zcr_threshold # 默认为手动值
            
            if args.zcr and args.auto_zcr:
                # 只有开启了 ZCR 且开启了 Auto 才进行校准
                calibrated_val = calibrate_zcr_threshold(
                    waveform, speech_probs, global_noise_threshold, frame_duration_s, SAMPLERATE
                )
                if calibrated_val is not None:
                    final_zcr_threshold = calibrated_val
                    print(f"【ZCR】ZCR 自适应阈值已调整为：{final_zcr_threshold:.3f}")
                else:
                    print(f"【ZCR】自适应阈值校准失败，使用值 {final_zcr_threshold}")

            # 先把 VAD 段转成毫秒区间（不做硬切分）
            base_ranges_ms = [
                [int(seg["start"] * 1000.0), int(seg["end"] * 1000.0)]
                for seg in speeches
            ]

            # 合并所有重叠的 VAD 段
            print("正在合并重叠的 VAD 语音块……")
            merged_ranges_ms = merge_overlapping_intervals(base_ranges_ms)
            if len(base_ranges_ms) != len(merged_ranges_ms):
                print(f"原 {len(base_ranges_ms)} 个 VAD 语音块已合并为 {len(merged_ranges_ms)} 个 VAD 语音块")
            else:
                print(f"没有需要合并的 VAD 语音块")

            # 记录这个VAD块在原始音频中的精确结束时间
            vad_chunk_end_times_s = [end_ms / 1000.0 for _, end_ms in merged_ranges_ms]

            nonsilent_ranges_ms = []
            
            # 遍历每一个合并后的大段（可能是 5秒，也可能是 80秒）
            for start_ms, end_ms in merged_ranges_ms:
                long_segment = {
                    'start': start_ms / 1000.0,
                    'end': end_ms / 1000.0
                }
                
                # 调用新的全局智能切分器
                OVERLAP_S = 1.0  # 此处定义重叠时长
                sub_segments = global_smart_segmenter(
                    long_segment,
                    speech_probs,
                    waveform, # 传入波形算能量
                    SAMPLERATE,
                    frame_duration_s,
                    MAX_SPEECH_DURATION_S, # 模型上限 (通常30s)
                    args.vad_threshold,
                    global_noise_threshold,
                    args.zcr,
                    final_zcr_threshold,
                    OVERLAP_S
                )
                
                # 将切好的子段加入处理队列
                for sub in sub_segments:
                    nonsilent_ranges_ms.append([
                        int(sub['start'] * 1000), 
                        int(sub['end'] * 1000)
                    ])

            print(
                f"VAD 侦测到 {len(merged_ranges_ms)} 个语音块，保守拆分超过 {MAX_SPEECH_DURATION_S} 秒的部分后，实际需要处理 {len(nonsilent_ranges_ms)} 个语音块"
            )
            if not nonsilent_ranges_ms:
                print("【警告】经过过滤后无有效语音块")
                recognition_end_time = time.time()
                return

            chunk_ranges_s = []
            chunk_subwords_list = []

            for i, (unpadded_start_ms, unpadded_end_ms) in enumerate(nonsilent_ranges_ms):
                # --- 准备一级语音块 ---
                # 计算包含静音保护的起止时间
                start_ms = max(0, unpadded_start_ms - args.keep_silence)
                end_ms = min(len(audio), unpadded_end_ms + args.keep_silence)
                
                # 导出当前的一级块
                chunk_path = temp_chunk_dir / f"chunk_{i + 1}.wav"
                # 先赋值给 chunk 变量，以便后续二次切分时使用，再导出文件
                chunk = silence_padding + audio[start_ms:end_ms] + silence_padding
                chunk.export(chunk_path, format="wav")
                
                print(
                    f"正在处理语音块 {i + 1}/{len(nonsilent_ranges_ms)} （该块起止时间：{SRTWriter._format_time(unpadded_start_ms / 1000.0)} --> {SRTWriter._format_time(unpadded_end_ms / 1000.0)}，时长：{(unpadded_end_ms - unpadded_start_ms) / 1000.0:.2f} 秒）",
                    end="", flush=True,
                    )

                final_subwords_in_this_chunk = []
                
                # 设定阈值：如果块时长超过模型最大允许时长的 1/3，则视为“长块”，需要二次拆解
                # 计算当前块的净时长（包含padding）
                if ((end_ms - start_ms) / 1000.0) <= MAX_SPEECH_DURATION_S / 3.0:
                    # === 分支 A：短块，直接识别 ===
                    print(" --> 短块，直接识别")
                    hyp, _ = model.transcribe(
                        [str(chunk_path)],
                        return_hypotheses=True,
                        verbose=False,
                    )
                    if hyp and hyp[0]:
                        ret = decode_hypothesis(model, hyp[0])
                        if ret.subwords:
                            # 坐标变换：全局时间 = 一级块偏移 + 相对时间
                            for sub in ret.subwords:
                                sub.seconds += start_ms / 1000.0
                                final_subwords_in_this_chunk.append(sub)
                else:
                    # === 分支 B：长块，执行二次 VAD 拆分后再识别 ===
                    print(" --> 长块，执行二次 VAD 拆分")
                    
                    # 加载一级块的波形
                    chunk_waveform, _ = torchaudio.load(chunk_path)
                    
                    # 运行局部 VAD
                    sub_speeches, _, _ = get_speech_timestamps_onnx(
                        chunk_waveform,
                        onnx_session,
                        threshold=args.vad_threshold,
                        sampling_rate=SAMPLERATE,
                        neg_threshold=args.vad_end_threshold,
                        min_speech_duration_ms=int(args.min_speech_duration_ms),
                        min_silence_duration_ms=int(args.min_silence_duration_ms)
                    )
                    
                    if not sub_speeches:
                        # 兜底策略：如果二次 VAD 没切出来（例如全是噪音），回退到整块识别
                        print("    【提示】二次 VAD 未发现有效分割点，回退到整块识别")
                        # 构造一个覆盖整个 chunk 的虚拟 VAD 段
                        # chunk 是 pydub 对象，len(chunk) 返回毫秒数，需转为秒
                        sub_speeches = [{'start': 0.0, 'end': len(chunk) / 1000.0}]

                    refined_sub_speeches = []
                    for seg in sub_speeches:
                        if (seg['end'] - seg['start']) <= MAX_SPEECH_DURATION_S:
                            # 长度正常，直接加入
                            refined_sub_speeches.append(seg)
                        else:
                            # 长度依然超标，执行带重叠的强制硬切分
                            curr = seg['start']
                            while curr < seg['end']:
                                # 计算这一刀的结束点
                                next_cut = curr + MAX_SPEECH_DURATION_S
                                
                                # 如果这一刀切到了末尾之后，就直接用末尾
                                if next_cut >= seg['end']:
                                    refined_sub_speeches.append({'start': curr, 'end': seg['end']})
                                    break
                                
                                # 加入当前硬切分段
                                refined_sub_speeches.append({'start': curr, 'end': next_cut})
                                
                                # 移动游标，回退 overlap_s 以形成重叠
                                curr = next_cut - OVERLAP_S

                    if len(refined_sub_speeches) > 1:
                        print(f"    --> 拆分并逐个识别 {len(refined_sub_speeches)} 个子片段")
                        
                    # 遍历二次拆分出的子块
                    for sub_idx, sub_seg in enumerate(refined_sub_speeches):
                        # 计算子块在一级块内部的起止时间 (ms)
                        # sub_seg['start'] 是相对于 chunk_path (0s) 的时间
                        sub_start_ms = int(sub_seg['start'] * 1000)
                        sub_end_ms = int(sub_seg['end'] * 1000)
                        
                        # 提取子块音频并加上静音保护
                        # 导出临时子块文件
                        sub_chunk_path = temp_chunk_dir / f"chunk_{i + 1}_sub_{sub_idx + 1}.wav"
                        (silence_padding + chunk[sub_start_ms:sub_end_ms] + silence_padding).export(sub_chunk_path,format="wav")

                        if len(refined_sub_speeches) != 1:
                            print(f"第 {i + 1}-{sub_idx + 1} 段 {(sub_end_ms - sub_start_ms) / 1000.0:.2f} 秒：")
                        
                        # 识别子块内容
                        hyp, _ = model.transcribe(
                            [str(sub_chunk_path)],
                            return_hypotheses=True,
                            verbose=False,
                        )
                        
                        if hyp and hyp[0]:
                            ret = decode_hypothesis(model, hyp[0])
                            if ret.subwords:
                                # 最终时间 = 一级块全局偏移 + 二级块在一级块内的偏移 + 识别出的相对时间
                                # 因为 sub_seg['start'] 是基于含填充的父chunk计算的，它包含了父chunk头部的 0.5s 静音，必须扣除
                                base_offset = start_ms / 1000.0 + sub_seg['start'] - PAD_SECONDS
                                for sub in ret.subwords:
                                    sub.seconds += base_offset
                                    final_subwords_in_this_chunk.append(sub)

                # --- 结果收集 ---
                if final_subwords_in_this_chunk:
                    chunk_subwords_list.append(final_subwords_in_this_chunk)
                    chunk_ranges_s.append((start_ms / 1000.0, end_ms / 1000.0))

            if chunk_subwords_list:
                # 所有块处理完后，再执行一次合并，如果只有一个块，无需合并
                print("所有语音块处理完毕，正在去重……")
                all_subwords = merge_overlap_dedup(chunk_subwords_list, chunk_ranges_s) if len(chunk_subwords_list) > 1 else chunk_subwords_list[0]
                print(f"语音块去重完毕")
            else:
                all_subwords = []

        # --- 计时结束：核心识别流程 ---
        recognition_end_time = time.time()

        # 如果整个过程下来没有任何识别结果，提前告知用户并退出，避免生成空文件
        if not all_subwords:
            print("=" * 70)
            print("【信息】未识别到任何有效的语音内容")
            return

        print("=" * 70)
        print("正在根据子词和VAD边界生成精确文本片段……")

        all_segments, segment_to_subword_map, subword_end_seconds = (
            create_precise_segments_from_subwords(all_subwords, vad_chunk_end_times_s, args.no_chunk, args.debug)
        )

        if not all_segments:
            print("【错误】未能生成任何文本片段，程序结束")
            return

        print("文本片段生成完成")

        if not args.no_chunk and args.refine_tail and segment_to_subword_map: # 只在VAD模式且启用精修且map存在时精修
            print("正在修除每段的尾部静音……")

            # 初始化 VAD 游标
            vad_cursor = 0
            num_vad_chunks = len(vad_chunk_end_times_s)

            for i, (seg, _, indices) in enumerate(segment_to_subword_map):
                if not indices:
                    continue
        
                # 如果当前 VAD 块的时间小于当前片段开始时间，游标后移
                # vad_chunk_end_times_s 是 VAD 块的“结束”时间
                # 要找第一个结束时间 > seg.start_seconds 的块
                while (vad_cursor < num_vad_chunks and 
                       vad_chunk_end_times_s[vad_cursor] <= seg.start_seconds):
                    vad_cursor += 1
                
                # 获取 max_end_s
                max_end_s = vad_chunk_end_times_s[vad_cursor] if vad_cursor < num_vad_chunks else seg.end_seconds

                # 遍历map，所以需要通过索引i来更新原始的all_segments列表
                all_segments[i].end_seconds = refine_tail_end_timestamp(
                    last_token_start_s=all_subwords[indices[-1]].seconds,# 直接从indices列表拿到最后一个子词的全局索引
                    rough_end_s=seg.end_seconds,
                    speech_probs=speech_probs,
                    waveform=waveform, 
                    samplerate=SAMPLERATE,
                    noise_threshold=global_noise_threshold,
                    frame_duration_s=frame_duration_s,
                    max_end_s=max_end_s,
                    min_silence_duration_ms=int(args.min_silence_duration_ms),
                    percentile=args.tail_percentile,
                    offset=args.tail_offset,
                    lookahead_ms=args.tail_lookahead_ms,
                    safety_margin_ms=args.tail_safety_margin_ms,
                    min_tail_keep_ms=args.tail_min_keep_ms,
                    use_zcr=args.zcr,
                    zcr_threshold=final_zcr_threshold,
                    debug = args.debug
                )

            # 邻段防重叠微调（保持你原来的逻辑）
            for i in range(len(all_segments) - 1):
                if all_segments[i].end_seconds > all_segments[i + 1].start_seconds:
                    all_segments[i].end_seconds = all_segments[i + 1].start_seconds
            
            print("结束时间戳精修完成")

        # --- 根据参数生成输出文件 ---
        print("=" * 70)
        print("识别完成，正在生成输出文件……")

        # 检查用户是否指定了任何一种文件输出格式
        file_output_requested = any(
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
        )

        if args.text or not file_output_requested:
            full_text = " ".join(seg.text for seg in all_segments)

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested:
            print("\n识别结果（完整文本）：")
            print(full_text)
            print("=" * 70)
            print("提示：未指定输出参数，结果将打印至控制台")
            print("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            output_path = output_dir / f"{base_name}.txt"
            with output_path.open("w", encoding="utf-8") as f: # Path 对象自带 open 方法
                f.write(full_text)
            print(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            output_path = output_dir / f"{base_name}.segments.txt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = TextWriter(f)
                writer.write_header()  # 虽然为空，但保持接口一致性
                for seg in all_segments:
                    writer.write(seg)
            print(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            output_path = output_dir / f"{base_name}.srt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = SRTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.segment2vtt:
            output_path = output_dir / f"{base_name}.vtt"
            with output_path.open("w", encoding="utf-8") as f:
                writer = VTTWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 WebVTT 字幕文件已保存为：{output_path}")

        if args.segment2tsv:
            output_path = output_dir / f"{base_name}.tsv"
            with output_path.open("w", encoding="utf-8") as f:
                writer = TSVWriter(f)
                writer.write_header()
                for seg in all_segments:
                    writer.write(seg)
            print(f"文本片段 TSV 文件已保存为：{output_path}")

        if args.subword:
            output_path = output_dir / f"{base_name}.subwords.txt"
            with output_path.open("w", encoding="utf-8") as f:
                for sub in all_subwords:
                    f.write(
                        f"[{SRTWriter._format_time(sub.seconds)}] {sub.token.replace(' ', '')}\n"
                    )
            print(f"带时间戳的所有子词信息已保存为：{output_path}")

        if args.subword2srt:
            output_path = output_dir / f"{base_name}.subwords.srt"
            with output_path.open("w", encoding="utf-8") as f:
                if subword_end_seconds:
                    for i, sub in enumerate(all_subwords):
                        f.write(f"{i + 1}\n")
                        f.write(f"{SRTWriter._format_time(sub.seconds)} --> {SRTWriter._format_time(subword_end_seconds[i])}\n")
                        f.write(f"{sub.token}\n\n")

            print(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            output_path = output_dir / f"{base_name}.subwords.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(
                    [{"token": sub.token, "timestamp": sub.seconds} for sub in all_subwords],
                    f, ensure_ascii=False, indent=4
                )
            print(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            output_path = output_dir / f"{base_name}.ass"
            with output_path.open("w", encoding="utf-8-sig") as f:

                # 使用 writer 生成标准文件头
                writer = ASSWriter(f)
                writer.write_header()

                dialogue_lines = []

                # 【直接遍历预先计算好的映射关系
                for seg, segment_subwords, segment_indices in segment_to_subword_map:

                    if not segment_subwords:
                        continue

                    karaoke_text = ""
                    for i, sub in enumerate(segment_subwords):
                        global_index = segment_indices[i]
                        karaoke_text += f"{{\\k{max(1, round((subword_end_seconds[global_index] - sub.seconds) * 100))}}}{sub.token}"

                    # 格式化 Dialogue 行，使用 writer 的时间格式化函数
                    dialogue_lines.append(
                        f"Dialogue: 0,{ASSWriter._format_time(seg.start_seconds)},{ASSWriter._format_time(seg.end_seconds)},Default,,0,0,0,,{karaoke_text}"
                    )

                # 写入文件
                f.write("\n".join(dialogue_lines))

            print(f"卡拉OK式 ASS 字幕已保存为：{output_path}")

    finally:
        # 恢复原始解码策略，以防后续有其他操作
        if "original_decoding_cfg" in locals():
             print("正在恢复原始解码策略……")
             model.change_decoding_strategy(original_decoding_cfg)

        print("=" * 70)
        # 安全地获取 ReazonSpeech 模型加载的起止时间，如果不存在则默认为 0
        asr_model_load_start = locals().get("asr_model_load_start", 0)
        asr_model_load_end = locals().get("asr_model_load_end", 0)
        if asr_model_load_end - asr_model_load_start > 0:
            print(
                f"ReazonSpeech模型加载耗时：{format_duration(asr_model_load_end - asr_model_load_start)}"
            )

        # 安全地获取 VAD 加载的起止时间，如果不存在则默认为 0
        vad_model_load_start = locals().get("vad_model_load_start", 0)
        vad_model_load_end = locals().get("vad_model_load_end", 0)
        # 安全地获取识别的起止时间，如果不存在则默认为 0
        recognition_start_time = locals().get("recognition_start_time", 0)
        recognition_end_time = locals().get("recognition_end_time", 0)
        # 只有当VAD加载时间大于0时才打印，否则不显示
        if vad_model_load_end - vad_model_load_start > 0:
            print(
                f"Pyannote-segmentation-3.0 模型加载耗时：{format_duration(vad_model_load_end - vad_model_load_start)}"
            )
            print(
                f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time - (vad_model_load_end - vad_model_load_start))}"
            )
        else:
            print(
                f"语音识别核心流程耗时：{format_duration(recognition_end_time - recognition_start_time)}"
            )

        print("=" * 70)
        if args.debug:
            print("调试模式已启用，临时文件和目录将被保留")
            print(f"临时 WAV 文件位于: {temp_wav_path}")
            if temp_chunk_dir and temp_chunk_dir.exists():
                print(f"临时分块目录位于: {temp_chunk_dir}")
                open_folder(temp_chunk_dir) # 调用新函数打开文件夹

        else:
            # --- 清理工作：删除临时的 WAV 文件 ---
            if temp_wav_path.exists():
                temp_wav_path.unlink()
                print(f"\n临时文件 '{temp_wav_path}' 已删除")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
            if temp_chunk_dir and temp_chunk_dir.exists():
                shutil.rmtree(temp_chunk_dir)
                print(f"临时块目录 '{temp_chunk_dir.name}' 已删除")

if __name__ == "__main__":
    main()
