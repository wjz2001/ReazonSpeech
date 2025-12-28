import argparse
import json
import os
import contextlib
import io
import re
import shlex
import shutil
import socket
import sys
import tempfile
import traceback
import uvicorn
from typing import Optional, Annotated
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
# 从 asr 中获取：主入口 + 构建 ArgumentParser 的函数
from asr import main as asr_main, arg_parser, get_asr_model

def get_local_ip():
    """
    获取本机在局域网内可用的 IPv4 地址。
    通过连到一个公网地址(不会真的发数据出去)来反查本机使用的 IP
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 这里的 8.8.8.8 只是用来获取路由信息，不会真的去访问
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def find_free_port(host, start_port):
    """从 start_port 开始往上找一个空闲端口"""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1

def get_args_info():
    """
    从 asr.arg_parser() 自动分析参数信息，区分：
    - config 参数：带有 --xxx 的选项（双横线）
    - output 参数：带有 -xxx 的选项（单横线）
    """
    config: dict[str, dict] = {}
    output: dict[str, dict] = {}
    config_flag_to_dest: set[str] = set()

    for action in arg_parser()._actions:
        # 位置参数（如 input_file）没有 option_strings，跳过
        if not action.option_strings:
            continue

        # 跳过 help
        if isinstance(action, argparse._HelpAction):
            continue

        long_flags = [
            opt for opt in action.option_strings
            if opt.startswith("--")
        ]
        short_flags = [
            opt for opt in action.option_strings
            if opt.startswith("-") and not opt.startswith("--")
        ]

        if long_flags:
            config[action.dest] = {
                "flags": long_flags,
            }
            config_flag_to_dest.update(long_flags)

        if short_flags:
            output[action.dest] = {
                "flags": short_flags,
            }

    return {
        "config": config,
        "output": output,
        "config_flag_to_dest": config_flag_to_dest,
    }

# 初始化参数信息
ARGS_INFO = get_args_info()

CONFIG_INFO = ARGS_INFO["config"]               # dest -> {...}
OUTPUT_INFO = ARGS_INFO["output"]               # dest -> {...}

CONFIG_DESTS = set(CONFIG_INFO.keys())          # {"no_chunk", "beam", ...}
OUTPUT_DESTS = set(OUTPUT_INFO.keys())          # {"text", "segment2srt", ...}
CONFIG_FLAGS = ARGS_INFO["config_flag_to_dest"]   # {"--no-chunk", "--beam", ...}
OPENAI_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}

# response_format: item(dest 名) -> CLI 短选项 flag
# 例如 "text" -> "-text", "segment2srt" -> "-segment2srt"
FMT_DEST_TO_FLAG: dict[str, str] = {
    dest: OUTPUT_INFO[dest]["flags"][0]
    for dest in OUTPUT_DESTS
}

def _apply_prompt_to_argv(argv: list[str], prompt: str):
    """
    将 OpenAI 样式的 prompt（CLI 字符串或 JSON）转成 asr.py 可识别的 CLI 选项，
    只允许配置类参数（--xxx），不允许输出类短选项（-text 等）。
    """
    if not prompt.strip():
        return

    # --- JSON 模式 ---
    if prompt.lstrip().startswith("{"):
        try:
            cfg = json.loads(prompt)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"prompt JSON 解析失败：{e}",
                        "type": "invalid_request_error",
                        "param": "prompt",
                        "code": "invalid_prompt_json",
                    }
                },
            )
        if not isinstance(cfg, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": 'prompt JSON 必须是一个对象（形如 {"no_chunk": true, ...}）',
                        "type": "invalid_request_error",
                        "param": "prompt",
                        "code": "invalid_prompt_type",
                    }
                },
            )
        for key, val in cfg.items():
            stripped_key = key.strip()
            dest = stripped_key.replace("-", "_")
            if dest not in CONFIG_DESTS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                f"prompt JSON 中包含未知配置参数 '{stripped_key}'，允许的配置参数有："
                                f"{', '.join(sorted(CONFIG_DESTS))}"
                            ),
                            "type": "invalid_request_error",
                            "param": stripped_key,
                            "code": "unknown_config_param",
                        }
                    },
                )
            flag = CONFIG_INFO[dest]["flags"][0]
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            elif val is None:
                continue
            else:
                argv.extend([flag, str(val)])
        return

    # --- CLI 字符串模式 ---
    try:
        tokens = shlex.split(prompt)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"prompt CLI 解析失败：{e}",
                    "type": "invalid_request_error",
                    "param": "prompt",
                    "code": "invalid_prompt_cli",
                }
            },
        )

    for tok in tokens:
        if not tok.startswith("-"):
            argv.append(tok)
            continue
        # 短选项（-xxx），prompt 不允许出现
        if tok.startswith("-") and not tok.startswith("--"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"prompt 中不允许指定 '{tok}'，输出参数请使用 response_format 指定"
                        ),
                        "type": "invalid_request_error",
                        "param": "prompt",
                        "code": "output_flag_in_prompt",
                    }
                },
            )
        # 长选项（--xxx）：必须是已知 config 参数
        if tok not in CONFIG_FLAGS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"prompt 中包含未知配置参数 '{tok}'，允许的配置参数有："
                            f"{', '.join(sorted(CONFIG_FLAGS))}"
                        ),
                        "type": "invalid_request_error",
                        "param": tok,
                        "code": "unknown_config_flag",
                    }
                },
            )
        argv.append(tok)

def build_argv_openai(
    input_path: str,
    prompt: str,
    rf: str,  # 已经是小写的 json/text/srt/verbose_json/vtt 之一
    timestamp_granularities: Optional[list[str]],
) -> list[str]:
    """
    OpenAI 风格的 response_format -> asr.py CLI 输出选项：
      json         -> -text
      text         -> -text
      srt          -> -segment2srt
      vtt          -> -segment2vtt
      verbose_json -> -text + (根据 timestamp_granularities 加 -segment/-subword)
    """
    argv: list[str] = [input_path]
    _apply_prompt_to_argv(argv, prompt)

    cli_dests: set[str] = set()

    if rf in ("json", "text"):
        cli_dests.add("text")

    elif rf == "srt":
        cli_dests.add("segment2srt")

    elif rf == "vtt":
        cli_dests.add("segment2vtt")

    elif rf == "verbose_json":
        cli_dests.add("text")
        granularities = timestamp_granularities or ["segment"]  # 官方默认 segment

        if set(granularities) - {"word", "segment"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "timestamp_granularities 只支持 'word' 和 'segment'"
                        ),
                        "type": "invalid_request_error",
                        "param": "timestamp_granularities",
                        "code": "invalid_timestamp_granularity",
                    }
                },
            )
        if "segment" in granularities:
            cli_dests.add("segment")
        if "word" in granularities:
            cli_dests.add("subword")

    # 把 dest 转成 CLI 短选项
    for dest in cli_dests:
        if dest not in OUTPUT_DESTS:
            # 这是内部配置错误，不是用户错误
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"输出类型 '{dest}' 未在输出参数中找到",
                        "type": "server_error",
                        "param": None,
                        "code": "missing_output_dest",
                    }
                },
            )
        argv.append(FMT_DEST_TO_FLAG[dest])

    return argv

def build_argv_legacy(
    input_path: str,
    prompt: str,
    response_format: str,
) -> list[str]:
    """
    旧的 ReazonSpeech 风格：直接把 response_format 当作 dest 名列表，
    例如 "text,segment2srt" -> -text -segment2srt
    """
    argv: list[str] = [input_path]
    _apply_prompt_to_argv(argv, prompt)

    cli_dests: set[str] = {
        t.strip().replace("-", "_")
        for t in re.split(r"[,\s]+", response_format or "")
        if t
    }

    for dest in cli_dests:
        if dest not in OUTPUT_DESTS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"response_format 中包含未知输出类型 '{dest}'，允许的输出类型有："
                            f"{', '.join(sorted(OUTPUT_DESTS))}"
                        ),
                        "type": "invalid_request_error",
                        "param": "response_format",
                        "code": "unknown_output_dest",
                    }
                },
            )
        argv.append(FMT_DEST_TO_FLAG[dest])

    return argv

app = FastAPI(title="ReazonSpeech API")

@app.exception_handler(HTTPException) # 捕获 raise HTTPException
async def openai_http_exception_handler(_: Request, exc: HTTPException):
    """
    将错误拦截下来，强制转换成 OpenAI API 风格错误格式：
    {
      "error": {
        "message": "...",
        "type": "...",
        "param": "...",
        "code": "..."
      }
    }
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = {
            "error": {
                "message": str(exc.detail),
                "type": "server_error" if exc.status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": None,
            }
        }
    return JSONResponse(status_code=exc.status_code, content=content)

@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    prompt: str = Form(""), # CLI 字符串或 JSON
    response_format: str = Form(""),
    timestamp_granularities: Annotated[
        Optional[list[str]],
        Form(alias="timestamp_granularities[]")
    ] = None,
):
    # 保存上传的文件到临时路径
    input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            input_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"无法保存上传的音频文件: {e}",
                    "type": "server_error",
                    "param": "file",
                    "code": "file_save_failed",
                }
            },
        )

    try:
        # -------------------------
        # 判定使用哪种 response_format 语义：OpenAI 风格 or 旧 dest 风格（禁止混用）
        # -------------------------
        tokens = [
            t.lower() 
            for t in re.split(r"[,\s]+", (response_format or "").strip()) 
            if t
        ]

        if not tokens:
            mode = "openai"
            rf = "text"
        else:
            if all(t in OPENAI_RESPONSE_FORMATS for t in tokens):
                if len(tokens) == 1:
                    mode = "openai"
                    rf = tokens[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "message": (
                                    "json / text / srt / verbose_json / vtt 作为 response_format 值时只能单选"
                                ),
                                "type": "invalid_request_error",
                                "param": "response_format",
                                "code": "multiple_openai_response_formats",
                            }
                        },
                    )
            elif any(t in OPENAI_RESPONSE_FORMATS for t in tokens):
                # 同时包含官方值和自定义 dest，显式禁止混用
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "response_format 不能同时混用 json / text / srt / verbose_json / vtt 和 ReazonSpeech 输出参数，只能二选一"
                            ),
                            "type": "invalid_request_error",
                            "param": "response_format",
                            "code": "mixed_response_format_styles",
                        }
                    },
                )
            else:
                # 全是自定义 dest 名 -> 旧模式
                mode = "legacy"
                rf = None

        # timestamp_granularities 只在 OpenAI + verbose_json 下合法
        if timestamp_granularities and not (mode == "openai" and rf == "verbose_json"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "timestamp_granularities 只能在 response_format=verbose_json 时使用",
                        "type": "invalid_request_error",
                        "param": "timestamp_granularities",
                        "code": "timestamp_granularities_requires_verbose_json",
                    }
                },
            )

        # 组装 argv 并调用 asr.main(argv=...)
        # argv 有值 => asr.main 会进入 API 模式，并返回内存中的结果 dict
        prompt_warnings = []
        # 必须同时满足
        # _apply_prompt_to_argv 不抛 HTTPException
        # 最终拼出来的 argv 能被 asr.arg_parser() 解析通过（否则说明 prompt 里有垃圾 token）
        user_prompt_ok = False
        if not prompt.strip():
            prompt_warnings.append("未输入任何提示词")
            prompt = ""
        else:
            try:
                tmp_argv = [input_path]
                _apply_prompt_to_argv(tmp_argv, prompt)
                with contextlib.redirect_stderr(io.StringIO()):
                    arg_parser().parse_args(tmp_argv)
                user_prompt_ok = True
            except (HTTPException, SystemExit):
                # 无论用户乱填啥都不报错，直接忽略用户 prompt，尝试 fallback
                prompt_warnings.append("提示词不符合格式要求")
                prompt = ""

        # prompt 为空或无效 -> 尝试读取 server.py 同级目录下的 reazonspeechprompt.txt
        if not user_prompt_ok:
            fallback_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "reazonspeechprompt.txt"
            )

            file_prompt_raw = ""
            if os.path.isfile(fallback_path):
                try:
                    with open(fallback_path, "r", encoding="utf-8-sig") as f:
                        file_prompt_raw = f.read()
                except Exception:
                    file_prompt_raw = ""
            else:
                prompt_warnings.append("找不到 reazonspeechprompt.txt")

            file_prompt = file_prompt_raw.strip()
            if not file_prompt:
                prompt_warnings.append("reazonspeechprompt.txt 中无内容")
                prompt = ""
            else:
                try:
                    tmp_argv = [input_path]
                    _apply_prompt_to_argv(tmp_argv, file_prompt)
                    with contextlib.redirect_stderr(io.StringIO()):
                        arg_parser().parse_args(tmp_argv)
                    prompt = file_prompt
                    prompt_warnings.append("使用 reazonspeechprompt.txt 中的内容作为提示词")
                except (HTTPException, SystemExit):
                    prompt_warnings.append("reazonspeechprompt.txt 中的内容不符合格式要求")
                    prompt = ""

        # 只在服务端日志打印，不返回给客户端，不影响识别流程
        if prompt_warnings:
            print(f"【提示词错误】{prompt_warnings}")

        if mode == "openai":
            argv = build_argv_openai(input_path, prompt, rf, timestamp_granularities)
        else:
            argv = build_argv_legacy(input_path, prompt, response_format)

        result = asr_main(argv)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "识别失败或未返回结果",
                        "type": "server_error",
                        "param": None,
                        "code": "empty_result",
                    }
                },
            )

        # asr.py 内部如果返回了 {"error": {...}}，直接转成 HTTP 错误
        if "error" in result:
            raise HTTPException(status_code=500 if result["error"]["type"] == "server_error" else 400, detail=result)

        if mode == "legacy":
            # 旧模式：直接把 asr_main 的结果整包 JSON 返回
            return JSONResponse(result)

        # OpenAI 模式：
        if rf == "json":
            return JSONResponse({
                "text": result["text"] or "",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 0,
                    "input_token_details": {
                        "text_tokens": 0,
                        "audio_tokens": 0
                    },
                    "output_tokens": 0,
                    "total_tokens": 0
                },
                "logprobs": None  # JSON 中会变成 null
            })

        elif rf == "text":
            # 官方 text：返回纯文本，从 -text 拿数据
            return PlainTextResponse(result["text"] or "", media_type="text/plain; charset=utf-8")

        elif rf == "srt":
            # 官方 srt：返回字幕纯文本，从 -segment2srt 拿数据
            return PlainTextResponse(result["segment2srt"] or "", media_type="application/x-subrip")

        elif rf == "vtt":
            # 官方 vtt：返回 WebVTT 纯文本，从 -segment2vtt 拿数据
            return PlainTextResponse(result["segment2vtt"] or "", media_type="text/vtt")

        elif rf == "verbose_json":
            # 根据 segment / subword 的 end 推断整段时长（秒）
            duration = result["duration"] or 0.0

            # 顶层字段
            verbose: dict = {
                "task": "transcribe",
                "language": "japanese",
                "duration": duration,
                "text": result["text"] or "",
            }

            # 根据 timestamp_granularities 决定是否返回 segments / words
            granularities = timestamp_granularities or ["segment"]

            if "segment" in granularities:
                segments = []
                for idx, seg in enumerate(result["segment"] or []):
                    segments.append(
                        {
                            "id": idx,
                            "seek": 0, # 兼容 Whisper 风格，统一设为 0
                            "start": seg["start"] or 0.0,
                            "end": seg["end"] or 0.0,
                            "text": seg["text"] or "",
                            "tokens": seg["tokens_id"],
                            # 占位
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                    )
                if segments:
                    verbose["segments"] = segments

            if "word" in granularities:
                words = []
                for w in result["subword"] or []:
                    words.append(
                        {
                            "word": w["token"] or "",
                            "start": w["start"] or 0.0,
                            "end": w["end"] or 0.0,
                        }
                    )
                if words:
                    verbose["words"] = words

            # usage：写整段音频实际时长（秒）
            verbose["usage"] = {
                "type": "duration",
                "seconds": duration,
            }

            return JSONResponse(verbose)

    except SystemExit as e:
        # argparse 报错
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "asr.py 参数解析失败，请检查 prompt 和 response_format",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": f"argparse_exit_{e.code}",
                }
            },
        )

    except HTTPException:
        # 捕获主动抛出的 HTTPException，直接原样抛出
        raise

    except Exception as e:
        print(f"Internal Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "语音识别服务内部错误",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )

    finally:
        if input_path and os.path.exists(input_path):
            # 删除临时文件
            try:
                os.remove(input_path)
            except Exception:
                pass


def main():
    """给 asr.py 调用的入口：预加载模型+选择端口 + 启动 uvicorn"""
    get_asr_model()

    host = "0.0.0.0"
    local_ip = get_local_ip()
    port = find_free_port(host, 8888)
    if port != 8888:
        print(f"端口 {8888} 已被占用，自动切换到端口 {port}")

    print("启动 ReazonSpeech API 服务：")
    print(f"  本机访问：http://127.0.0.1:{port}/v1/audio/transcriptions")
    print(f"  本机或局域网设备访问：http://{local_ip}:{port}/v1/audio/transcriptions")
    print(f"  服务监听地址：http://{host}:{port}/v1/audio/transcriptions")
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    print("请使用: reazonspeech 命令启动 API 服务，不要直接运行本文件")
    sys.exit(1)