import argparse
import os
import tempfile
import shutil
import torch
import torchaudio
import json
from pathlib import Path
from pydub import AudioSegment
from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.decode import decode_hypothesis
from reazonspeech.nemo.asr.interface import Segment, Subword

# ONNX VAD 依赖
try:
    import onnxruntime
    import numpy as np
except ImportError:
    onnxruntime = None

# --- ONNX VAD 辅助函数 ---
def get_speech_timestamps_onnx(wav_tensor, onnx_session, threshold=0.5, sampling_rate=16000):
    """使用 ONNX 模型和后处理来获取语音时间戳"""
    # ONNX 模型需要特定的输入形状
    if wav_tensor.dim() == 2:
        wav_tensor = wav_tensor.unsqueeze(0)
    
    # 运行 ONNX 模型
    ort_inputs = {'input_values': wav_tensor.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    logits = torch.from_numpy(ort_outs[0])[0] # 获取 logits: [num_frames, num_classes]
    
    # Pyannote-segmentation-3.0 的输出中，索引2是 "speech"
    speech_probs = torch.sigmoid(logits[:, 2])
    speech_frames = speech_probs > threshold
    
    frame_duration_s = (wav_tensor.shape[2] / sampling_rate) / logits.shape[0]
    speech_timestamps = []
    is_speech = False
    start_time = 0
    
    for i, frame in enumerate(speech_frames):
        if frame and not is_speech:
            is_speech = True
            start_time = i * frame_duration_s
        elif not frame and is_speech:
            is_speech = False
            end_time = i * frame_duration_s # 结束点是当前帧的开始
            speech_timestamps.append({'start': start_time, 'end': end_time})
    if is_speech:
        end_time = len(speech_frames) * frame_duration_s
        speech_timestamps.append({'start': start_time, 'end': end_time})
            
    return speech_timestamps
    
def format_srt_time(seconds):
    """将秒数格式化为 SRT 时间戳格式 (HH:MM:SS,ms)"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_ass_time(seconds):
    """将秒数格式化为 ASS 时间戳格式 (H:MM:SS.cs)"""
    assert seconds >= 0, "non-negative timestamp expected"
    h = int(seconds / 3600)
    m = int(seconds / 60) % 60
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="使用 ReazonSpeech 模型识别语音，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题"
    )
    
    # 添加一个必须的位置参数：音频/视频文件路径
    parser.add_argument(
        "input_file", 
        help="需要识别语音的音频/视频文件路径"
    )

    # --- --no-chunk 参数 ---
    parser.add_argument(
        "--no-chunk", 
        action="store_true", 
        help="禁用智能分块功能，一次性处理整个音频文件"
    )

    # --- VAD (Silero VAD) 核心参数 ---
    parser.add_argument("--vad_threshold", type=float, default=0.2, help="【VAD】判断为语音的置信度阈值（0-1）")
    parser.add_argument("--min_speech_duration_ms", type=float, default=100, help="【过滤器】移除短于此时长（毫秒）的语音块")
    parser.add_argument("--keep_silence", type=int, default=30, help="在语音块前后扩展时长（毫秒）")

    # 输出格式参数
    parser.add_argument(
        "-text", 
        action="store_true", 
        help="仅输出完整的识别文本并保存为 .txt 文件"
    )
    parser.add_argument(
        "-segment", 
        action="store_true", 
        help="输出带时间戳的文本片段（Segment）并保存为 .segments.txt 文件"
    )
    parser.add_argument(
        "-segment2srt", 
        action="store_true", 
        help="输出带时间戳的文本片段（Segment）并转换为 .srt 字幕文件"
    )
    parser.add_argument(
        "-subword", 
        action="store_true", 
        help="输出带时间戳的所有子词（Subword）并保存为 .subwords.txt 文件"
    )
    parser.add_argument(
        "-subword2srt", 
        action="store_true", 
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.srt 字幕文件"
    )
    parser.add_argument(
        "-subword2json",
        action="store_true",
        help="输出带时间戳的所有子词（Subword）并转换为 .subwords.json 文件"
    )
    parser.add_argument(
        "-kass",
        action="store_true",
        help="生成逐字计时的卡拉OK式 .ass 字幕文件"
    )

    args = parser.parse_args()

    # --- 准备路径和临时文件 ---
    input_path = args.input_file
    output_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 创建一个临时的 WAV 文件
    # delete=False 确保在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav_file.name
    temp_wav_file.close() # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = tempfile.mkdtemp()


    # --- 执行核心的语音识别流程 ---
    try:
        # --- ffmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件……")
        audio = AudioSegment.from_file(input_path)
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        print("转换完成")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型……")
        model = load_model()
        print("模型加载完成")

        # --- 逐块识别并校正时间戳 ---
        all_segments = []
        all_subwords = []
        vad_chunk_end_times_s = [] # 用于存储VAD块的结束时间

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("检测到 --no-chunk 参数，将一次性处理整个文件……")
            hyp, _ = model.transcribe([temp_wav_path], return_hypotheses=True, verbose=True, override_config=None)
            ret = decode_hypothesis(model, hyp[0])
            all_segments = ret.segments
            all_subwords = ret.subwords

        else:
            if not onnxruntime:
                print("【错误】onnxruntime 未安装，请运行 'pip install onnxruntime'")
                return
            
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            local_onnx_model_path = script_dir / 'models' / 'model_quantized.onnx'
            
            if not local_onnx_model_path.exists():
                print(f"【错误】Pyannote-segmentation-3.0 模型未在 '{local_onnx_model_path}' 中找到")
                print("请下载 model_quantized.onnx 并放入 models 文件夹")
                return

            print("正在从本地路径加载 Pyannote-segmentation-3.0 模型（将在 CPU 上运行）……")
            onnx_session = onnxruntime.InferenceSession(str(local_onnx_model_path), providers=['CPUExecutionProvider'])
            print("Pyannote-segmentation-3.0 模型加载完成")

            print("正在使用 Pyannote-segmentation-3.0 侦测语音活动……")
            wav_tensor, sr = torchaudio.load(temp_wav_path)
            speech_timestamps_seconds = get_speech_timestamps_onnx(
                wav_tensor, onnx_session, args.vad_threshold
            )
                
            nonsilent_ranges_ms = [[ts['start'] * 1000, ts['end'] * 1000] for ts in speech_timestamps_seconds]

            if not nonsilent_ranges_ms:
                print("【警告】未侦测到语音活动"); 
                filtered_ranges = []
            else:
                original_chunk_count = len(nonsilent_ranges_ms)
                min_speech_duration_ms = args.min_speech_duration_ms
                filtered_ranges = [r for r in nonsilent_ranges_ms if (r[1] - r[0]) >= min_speech_duration_ms]
                print(f"VAD 侦测到 {original_chunk_count} 个语音块，已过滤不超过 {min_speech_duration_ms}ms 的部分，保留并处理 {len(filtered_ranges)} 个语音块")

            if len(filtered_ranges) > 0:
                wav_audio = AudioSegment.from_wav(temp_wav_path)
                for i, time_range in enumerate(filtered_ranges):
                    start_ms, end_ms = time_range
                    # 记录这个VAD块在原始音频中的精确结束时间
                    chunk_end_time_s = end_ms / 1000.0
                    vad_chunk_end_times_s.append(chunk_end_time_s)
                    start_ms = max(0, start_ms - args.keep_silence)
                    end_ms = min(len(wav_audio), end_ms + args.keep_silence)
                    chunk = wav_audio[start_ms:end_ms]
                    time_offset_s = start_ms / 1000.0

                    chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
                    print(f"正在处理语音块 {i+1}/{len(filtered_ranges)} （该块起止时间：{format_srt_time(start_ms/1000.0)} --> {format_srt_time(end_ms/1000.0)}）……")
                    chunk.export(chunk_path, format="wav")
                    
                    hyp, _ = model.transcribe([chunk_path], return_hypotheses=True, verbose=False, override_config=None)
                    
                    if hyp and hyp[0]:
                        ret = decode_hypothesis(model, hyp[0])
                        if ret.segments:
                            for seg in ret.segments: all_segments.append(Segment(start_seconds=seg.start_seconds + time_offset_s, end_seconds=seg.end_seconds + time_offset_s, text=seg.text))
                        if ret.subwords:
                            for sub in ret.subwords: all_subwords.append(Subword(seconds=sub.seconds + time_offset_s, token_id=sub.token_id, token=sub.token))

        # 如果整个过程下来没有任何识别结果，提前告知用户并退出，避免生成空文件
        if not all_segments:
            print("=" * 70)
            print("【信息】未识别到任何有效的语音内容，程序结束")
            return
        
        if all_segments and all_subwords:

            # --- 预先计算所有子词的、尊重 VAD 边界的结束时间 ---
            subword_end_seconds = []
            if all_subwords and not args.no_chunk and vad_chunk_end_times_s:
                print("=" * 70)
                print("正在根据 VAD 边界校正子词时间戳……")
                vad_cursor = 0
                for i, sub in enumerate(all_subwords):
                    # 确定当前子词的 VAD 边界
                    while vad_cursor < len(vad_chunk_end_times_s) and sub.seconds > vad_chunk_end_times_s[vad_cursor]:
                        vad_cursor += 1
                    
                    vad_boundary_end_s = float('inf')
                    if vad_cursor < len(vad_chunk_end_times_s):
                        vad_boundary_end_s = vad_chunk_end_times_s[vad_cursor]
    
                    # 计算潜在的结束时间
                    potential_end_time = 0
                    if i < len(all_subwords) - 1:
                        potential_end_time = all_subwords[i+1].seconds
                    else: # 如果是最后一个子词
                        potential_end_time = sub.seconds + 0.5 # 估算一个时长
    
                    # 取 VAD 边界 和 下一个子词开始时间 的最小值
                    corrected_end_time = min(potential_end_time, vad_boundary_end_s)
                    
                    # 安全检查，确保结束时间 > 开始时间
                    if corrected_end_time <= sub.seconds:
                        corrected_end_time = sub.seconds + 0.1
                    
                    subword_end_seconds.append(corrected_end_time)
                print("子词时间戳校正完成")
    
                # --- 用子词的时间戳来从零开始构建片段的时间戳，并创建映射关系 ---
                segment_to_subword_map = [] # 新的数据结构，存储元组 (segment, subwords_list, subwords_indices_list)
                subword_cursor = 0 # 全局子词列表的游标
                print("正在根据子词时间戳优化文本片段的时间戳……")
    
                for seg in all_segments:
                    # 为当前片段的文本，找到其对应的子词序列
                    target_text = seg.text.replace(" ", "")
                    segment_subwords = []
                    segment_indices = [] # 新增：用于存储子词的全局索引
                    temp_text = ""
                    
                    match_start_cursor = subword_cursor
                    
                    while subword_cursor < len(all_subwords):
                        sub = all_subwords[subword_cursor]
                        
                        # 记录子词对象和它的全局索引
                        segment_subwords.append(sub)
                        segment_indices.append(subword_cursor)
    
                        temp_text += sub.token.replace(' ', '')
                        subword_cursor += 1
                        
                        if temp_text == target_text:
                            break
                    else:
                        print(f"【警告】未能为片段 '{seg.text}' 找到完全匹配的子词序列，此片段将被跳过")
                        subword_cursor = match_start_cursor + 1
                        continue
                    
                    # 如果找到了匹配的子词序列，用它们精确地重建时间戳并存储映射
                    if segment_subwords:
                        new_start_time = segment_subwords[0].seconds
                        
                        last_subword_index = segment_indices[-1] # 使用它自己的最后一个子词的索引
                        
                        new_end_time = 0
                        if subword_end_seconds:
                            new_end_time = subword_end_seconds[last_subword_index]
                        else:
                            if subword_cursor < len(all_subwords):
                               new_end_time = all_subwords[subword_cursor].seconds
                            else:
                               new_end_time = segment_subwords[-1].seconds + 0.5
    
                        if new_end_time <= new_start_time:
                            new_end_time = new_start_time + 0.2
                        
                        # 创建新的、时间戳更精确的 Segment 对象
                        new_segment = Segment(start_seconds=new_start_time, end_seconds=new_end_time, text=seg.text)
                        
                        # 将新的 Segment、其对应的子词列表、以及索引列表作为一个整体存入 map
                        segment_to_subword_map.append((new_segment, segment_subwords, segment_indices))
    
                # --- 从映射中更新 all_segments ---
                if segment_to_subword_map:
                    all_segments = [item[0] for item in segment_to_subword_map]
                    print("文本片段时间戳优化完成")
                else:
                    print("【错误】未能重建任何文本片段，识别失败")
                    return

        # --- 根据参数生成输出文件 ---
        print("=" * 70)
        print("识别完成，正在生成输出文件……")

        # 检查用户是否指定了任何一种文件输出格式
        file_output_requested = any([
            args.text, args.segment, args.segment2srt, 
            args.subword, args.subword2srt, args.subword2json, args.kass
        ])

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested:
            full_text = " ".join([seg.text for seg in all_segments])
            print("\n识别结果（完整文本）：")
            print(full_text)
            print("=" * 70)
            print("提示：未指定输出参数，结果将打印至控制台")
            print("请使用 -text，-segment2srt，-kass 等参数将结果保存为文件")

        if args.text:
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            full_text = " ".join([seg.text for seg in all_segments])
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"完整的识别文本已保存为：{output_path}")

        if args.segment:
            output_path = os.path.join(output_dir, f"{base_name}.segments.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in all_segments:
                    f.write(f"[{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}] {seg.text}\n")
            print(f"带时间戳的文本片段已保存为：{output_path}")

        if args.segment2srt:
            output_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(all_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}\n")
                    f.write(f"{seg.text}\n\n")
            print(f"文本片段 SRT 字幕文件已保存为：{output_path}")

        if args.subword:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in all_subwords:
                    f.write(f"[{format_srt_time(sub.seconds)}] {sub.token.replace(' ', '')}\n")
            print(f"带时间戳的所有子词信息已保存为：{output_path}")
        
        if args.subword2srt:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                # 如果有 VAD 校正后的时间，优先使用
                if subword_end_seconds:
                    for i, sub in enumerate(all_subwords, 1):
                        end_time_s = subword_end_seconds[i-1]
                        f.write(f"{i}\n")
                        f.write(f"{format_srt_time(sub.seconds)} --> {format_srt_time(end_time_s)}\n")
                        f.write(f"{sub.token.replace(' ', '')}\n\n")
                else: # 否则，沿用旧的 --no-chunk 逻辑
                    if len(all_subwords) > 1:
                        for i, sub in enumerate(all_subwords[:-1], 1):
                            f.write(f"{i}\n")
                            f.write(f"{format_srt_time(sub.seconds)} --> {format_srt_time(all_subwords[i].seconds)}\n")
                            f.write(f"{sub.token.replace(' ', '')}\n\n")
                    if all_subwords:
                        last_sub = all_subwords[-1]
                        end_time_s = last_sub.seconds + 0.5
                        f.write(f"{len(all_subwords)}\n")
                        f.write(f"{format_srt_time(last_sub.seconds)} --> {format_srt_time(end_time_s)}\n")
                        f.write(f"{last_sub.token.replace(' ', '')}\n\n")
            print(f"所有子词信息的 SRT 文件已保存为：{output_path}")

        if args.subword2json:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.json")
            subwords_for_json = []
            for sub in all_subwords:
                subwords_for_json.append({
                    "token": sub.token.replace(' ', ' '),
                    "timestamp": sub.seconds
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subwords_for_json, f, ensure_ascii=False, indent=4)
            
            print(f"所有子词信息的 JSON 文件已保存为：{output_path}")

        if args.kass:
            output_path = os.path.join(output_dir, f"{base_name}.ass")
            
            # ASS 字幕文件头
            ass_header = """\
[Script Info]
ScriptType: v4.00+
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&HFF000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            dialogue_lines = []
            
            # 【直接遍历预先计算好的映射关系，而不是 all_segments
            for seg, segment_subwords, segment_indices in segment_to_subword_map:
                
                if not segment_subwords:
                    continue

                karaoke_text = ""
                # 遍历当前片段的子词，计算卡拉OK时长
                for i, sub in enumerate(segment_subwords):
                    duration_s = 0
                    # 优先使用预先计算好的、VAD感知的结束时间
                    if subword_end_seconds:
                        global_index = segment_indices[i] # 直接使用映射中保存的全局索引
                        duration_s = subword_end_seconds[global_index] - sub.seconds
                    else: # --no-chunk
                        if i < len(segment_subwords) - 1:
                            duration_s = segment_subwords[i+1].seconds - sub.seconds
                        else: # 最后一个词
                            duration_s = max(0.1, seg.end_seconds - sub.seconds)

                    # 转换为厘秒 (cs)
                    duration_cs = max(1, round(duration_s * 100))
                    
                    # 清理 token 中的特殊空格字符
                    clean_token = sub.token.replace(' ', ' ')
                    
                    karaoke_text += f"{{\\k{duration_cs}}}{clean_token}"
                
                # 格式化 Dialogue 行
                start_time = format_ass_time(seg.start_seconds)
                end_time = format_ass_time(seg.end_seconds)
                dialogue_lines.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_text}")

            # 写入文件
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write(ass_header)
                f.write("\n".join(dialogue_lines))
            
            print(f"卡拉OK式 ASS 字幕已保存为：{output_path}")

    finally:
        # --- 清理工作：删除临时的 WAV 文件 ---
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"\n临时文件 '{temp_wav_path}' 已删除")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"临时块目录 '{os.path.basename(temp_chunk_dir)}' 已删除")

if __name__ == "__main__":
    main()