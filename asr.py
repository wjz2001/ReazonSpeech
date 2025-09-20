import argparse
import os
import tempfile
import shutil
import torch
import torchaudio
from pydub import AudioSegment
# 导入 OmegaConf 用于创建解码配置
from omegaconf import OmegaConf

from reazonspeech.nemo.asr import load_model
from reazonspeech.nemo.asr.decode import decode_hypothesis
from reazonspeech.nemo.asr.interface import Segment, Subword
from silero_vad import load_silero_vad, get_speech_timestamps

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
        description="使用 ReazonSpeech 模型进行语音识别，并按指定格式输出结果。基于静音的智能分块方式识别长音频，以保证准确率并解决显存问题。"
    )
    
    # 添加一个必须的位置参数：音频文件路径
    parser.add_argument(
        "input_file", 
        help="需要进行语音识别的音频文件路径"
    )

    # --- --no-chunk 参数 ---
    parser.add_argument(
        "--no-chunk", 
        action="store_true", 
        help="禁用智能分块功能，一次性处理整个音频文件"
    )

    # --- 解码参数 ---
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam search 的大小。大于1会显著提升准确率但降低速度。推荐值为5或10。"
    )

    # --- VAD (Silero VAD) 核心参数 ---
    parser.add_argument("--vad_threshold", type=float, default=0.2, help="[VAD] 模型的置信度阈值 (0-1)，值越高判断越严格。")
    parser.add_argument("--vad_neg_threshold", type=float, default=0.1, help="[VAD] 语音结束判断的阈值。默认是 threshold - 0.15。")
    parser.add_argument("--vad_min_silence_ms", type=int, default=250, help="[VAD] 结束语音块前需要等待的最小静音时长（毫秒）。")
    parser.add_argument("--min_speech_duration_ms", type=int, default=0, help="[VAD 过滤器] 语音块被处理的最小持续时间（毫秒），用于过滤噪音。")
    parser.add_argument("--keep_silence", type=int, default=300, help="在语音块前后扩展的时长（毫秒），以防切断单词。")

    # 创建一个互斥的参数组，因为一次只能选择一种输出格式
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-text", 
        action="store_true", 
        help="仅输出完整的识别文本并保存到 .txt 文件 (默认选项)"
    )
    output_group.add_argument(
        "-segment", 
        action="store_true", 
        help="输出带时间戳的文本片段 (Segment)并保存到 .segments.txt 文件"
    )
    output_group.add_argument(
        "-segment2srt", 
        action="store_true", 
        help="将文本片段 (Segment) 转换为 SRT 字幕文件并保存"
    )
    output_group.add_argument(
        "-subword", 
        action="store_true", 
        help="输出所有的子词 (Subword) 及其时间戳并保存到保存到 .subwords.txt 文件"
    )
    output_group.add_argument(
        "-subword2srt", 
        action="store_true", 
        help="将子词 (Subword) 转换为 SRT 字幕文件并保存"
    )
    output_group.add_argument(
        "-kass",
        action="store_true",
        help="生成逐字计时的卡拉OK式 ASS 字幕文件 (.k.ass)"
    )

    args = parser.parse_args()

    # --- 准备路径和临时文件 ---
    input_path = args.input_file
    output_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 创建一个临时的 WAV 文件
    # delete=False 确保我们可以在 with 块外使用它，最后手动删除
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav_file.name
    temp_wav_file.close() # 关闭文件句柄，以便 ffmpeg 可以写入

    temp_chunk_dir = tempfile.mkdtemp()


    # --- 执行核心的语音识别流程 ---
    try:
        # --- FFmpeg 预处理：将输入文件转换为标准 WAV ---
        print(f"正在转换输入文件 '{input_path}' 为临时 WAV 文件...")
        audio = AudioSegment.from_file(input_path)
        # 转换为单声道，16kHz采样率，这是ASR模型的标准格式
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        print("转换完成。")

        # --- 加载模型 (只需一次) ---
        print("正在加载模型...")
        model = load_model(device='cuda')
        print("模型加载完成。")

        # 创建一个解码配置对象
        if args.beam_size > 1:
            print(f"已启用 Beam Search 解码，beam size = {args.beam_size}。")
            decoding_cfg = OmegaConf.create(
                {
                    "decoding_strategy": "beam",
                    "beam": {"beam_size": args.beam_size},
                }
            )
        else:
            # 对于贪心解码，我们可以不传递任何特殊配置
            decoding_cfg = None

        # --- 逐块识别并校正时间戳 ---
        all_segments = []
        all_subwords = []

        if args.no_chunk:
            # --- 不分块的逻辑 ---
            print("检测到 --no-chunk 参数，将一次性处理整个文件……")
            hyp, _ = model.transcribe([temp_wav_path], return_hypotheses=True, verbose=True, override_config=decoding_cfg)
            ret = decode_hypothesis(model, hyp[0])
            all_segments = ret.segments
            all_subwords = ret.subwords

        else:
            # --- 智能分块的逻辑 ---
            print("正在使用 Silero VAD 模型侦测音频中的语音活动以进行智能分块……")
            vad_model = load_silero_vad(onnx=True)
            # 加载音频为 tensor
            wav_tensor, _ = torchaudio.load(temp_wav_path)
            
            # 调用官方函数
            speech_timestamps_seconds = get_speech_timestamps(
                wav_tensor, 
                vad_model, 
                threshold=args.vad_threshold,
                neg_threshold=args.vad_neg_threshold,
                min_silence_duration_ms=args.vad_min_silence_ms,
                min_speech_duration_ms=args.min_speech_duration_ms, # 注意：这个参数函数自带，我们不再需要手动过滤
                speech_pad_ms=args.keep_silence,
                return_seconds=True
            )
            print("Silero VAD 模型加载完成。")
            
            # 将秒转换为毫秒
            nonsilent_ranges_ms = [[ts['start'] * 1000, ts['end'] * 1000] for ts in speech_timestamps_seconds]
            
            if not nonsilent_ranges_ms:
                print("警告：未侦测到语音活动。将尝试处理整个文件。"); nonsilent_ranges_ms = [[0, len(AudioSegment.from_wav(temp_wav_path))]]
            
            original_chunk_count = len(nonsilent_ranges_ms)
            filtered_ranges = [r for r in nonsilent_ranges_ms if (r[1] - r[0]) >= args.min_speech_duration_ms]
            print(f"VAD 侦测到 {original_chunk_count} 个语音块，经过滤（最短 {args.min_speech_duration_ms}ms），保留 {len(filtered_ranges)} 个块进行处理。")

            if len(filtered_ranges) > 0:
                wav_audio = AudioSegment.from_wav(temp_wav_path)
                for i, time_range in enumerate(filtered_ranges):
                    start_ms, end_ms = time_range
                    start_ms = max(0, start_ms - args.keep_silence)
                    end_ms = min(len(wav_audio), end_ms + args.keep_silence)
                    chunk = wav_audio[start_ms:end_ms]
                    time_offset_s = start_ms / 1000.0

                    chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
                    print(f"正在处理块 {i+1}/{len(filtered_ranges)} (时间: {format_srt_time(time_offset_s)})...")
                    chunk.export(chunk_path, format="wav")
                    
                    hyp, _ = model.transcribe([chunk_path], return_hypotheses=True, verbose=False, override_config=decoding_cfg)
                    
                    if hyp and hyp[0]:
                        ret = decode_hypothesis(model, hyp[0])
                        if ret.segments:
                            for seg in ret.segments: all_segments.append(Segment(start_seconds=seg.start_seconds + time_offset_s, end_seconds=seg.end_seconds + time_offset_s, text=seg.text))
                        if ret.subwords:
                            for sub in ret.subwords: all_subwords.append(Subword(seconds=sub.seconds + time_offset_s, token_id=sub.token_id, token=sub.token))
        # --- 根据参数生成输出文件 ---
        print("-" * 20)
        print("识别完成，正在生成输出文件……")

        # 检查用户是否指定了任何一种文件输出格式
        file_output_requested = any([
            args.text, args.segment, args.segment2srt, 
            args.subword, args.subword2srt, args.kass
        ])

        # 只有在用户完全没有指定任何输出参数时，才在控制台打印
        if not file_output_requested:
            full_text = " ".join([seg.text for seg in all_segments])
            print("\n识别结果 (完整文本):")
            print(full_text)
            print("-" * 20)
            print("提示：未指定输出参数，结果仅打印到控制台。")
            print("请使用 -text, -segment2srt, -kass 等参数将结果保存到文件。")

        no_output_flag = not any([args.text, args.segment, args.segment2srt, args.subword, args.subword2srt])

        if args.text:
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            full_text = " ".join([seg.text for seg in all_segments])
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"完整文本已保存到: {output_path}")

        if args.segment:
            output_path = os.path.join(output_dir, f"{base_name}.segments.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in all_segments:
                    f.write(f"[{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}] {seg.text}\n")
            print(f"文本片段已保存到: {output_path}")

        if args.segment2srt:
            output_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(all_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_time(seg.start_seconds)} --> {format_srt_time(seg.end_seconds)}\n")
                    f.write(f"{seg.text}\n\n")
            print(f"SRT 字幕文件已保存到: {output_path}")

        if args.subword:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in all_subwords:
                    f.write(f"[{format_srt_time(sub.seconds)}] {sub.token.replace(' ', '')}\n")
            print(f"子词信息已保存到: {output_path}")
        
        if args.subword2srt:
            output_path = os.path.join(output_dir, f"{base_name}.subwords.srt")
            with open(output_path, 'w', encoding='utf-8') as f:
                if len(all_subwords) > 1:
                    for i, sub in enumerate(all_subwords[:-1], 1):
                        f.write(f"{i}\n")
                        f.write(f"{format_srt_time(sub.seconds)} --> {format_srt_time(all_subwords[i].seconds)}\n")
                        f.write(f"{sub.token.replace(' ', '')}\n\n")
                if all_subwords:
                    last_sub = all_subwords[-1]
                    f.write(f"{len(all_subwords)}\n")
                    f.write(f"{format_srt_time(last_sub.seconds)} --> {format_srt_time(last_sub.seconds + 0.2)}\n")
                    f.write(f"{last_sub.token.replace(' ', '')}\n\n")
            print(f"子词 SRT 文件已保存到: {output_path}")

        if args.kass:
            output_path = os.path.join(output_dir, f"{base_name}.k.ass")
            
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
            subword_idx = 0
            
            for seg in all_segments:
                segment_subwords = []
                # 找到属于当前 segment 的所有 subwords
                while subword_idx < len(all_subwords) and all_subwords[subword_idx].seconds < seg.end_seconds:
                    if all_subwords[subword_idx].seconds >= seg.start_seconds:
                        segment_subwords.append(all_subwords[subword_idx])
                    subword_idx += 1
                
                if not segment_subwords:
                    continue

                karaoke_text = ""
                # 计算每个 subword 的时长
                for i, sub in enumerate(segment_subwords):
                    duration_s = 0
                    if i < len(segment_subwords) - 1:
                        duration_s = segment_subwords[i+1].seconds - sub.seconds
                    else:
                        # 最后一个 subword 的时长，估算为到 segment 结尾
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
            
            print(f"卡拉OK式 ASS 字幕已保存到: {output_path}")

    finally:
        # --- 清理工作：删除临时的 WAV 文件 ---
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"\n临时文件 '{temp_wav_path}' 已删除。")

        # 使用 shutil.rmtree 来删除临时块目录及其所有内容
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"临时块目录 '{os.path.basename(temp_chunk_dir)}' 已删除。")

if __name__ == "__main__":
    main()