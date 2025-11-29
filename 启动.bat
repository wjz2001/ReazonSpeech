@echo off
rem 切换代码页以正确显示中文字符
chcp 65001 > nul
setlocal enabledelayedexpansion
title ReazonSpeech 语音识别助手

REM --- 检查并启动虚拟环境 ---

echo 正在检查虚拟环境……

if not exist ".\venv\Scripts\activate.bat" (
    echo.

    echo 【错误】未在 '.\venv\Scripts\activate.bat' 中找到虚拟环境

    echo 请确保您正在项目的根目录下运行此脚本

    echo.
    pause
    exit /b 1
)

echo 正在激活虚拟环境……

call ".\venv\Scripts\activate.bat"

echo 虚拟环境激活成功！

echo.

REM --- 获取音视频文件路径 ---
:GetFile

echo ======================================================================

set "inputFile="

set /p "inputFile=请拖入您的音频/视频文件或输入它的完整路径，然后按回车键："

rem 移除可能存在于路径开头和结尾的引号
set "inputFile=!inputFile:"=!"

if not defined inputFile goto GetFile

if not exist "%inputFile%" (
    echo.

    echo 【错误】文件未找到，请检查路径后重试

    echo.
    goto GetFile
)

echo 已找到文件: %inputFile%

echo.

REM --- 询问是否使用VAD ---
:AskChunk

echo ======================================================================

set "chunkOption="
choice /c YN /m "是否使用智能语音分块？（Y=是，推荐使用；N=否，除非音频极短或为了测试，否则不推荐使用）"
if %ERRORLEVEL% == 2 (
    set "chunkOption= --no-chunk"
    goto AskBeamParams
)

REM --- (如果使用VAD) 询问是否修改VAD参数 ---
:AskVadParams
echo.
set "chunkParams="
choice /c YN /m "是否需要修改默认的VAD参数？"

if %ERRORLEVEL% == 2 (
    goto AskZCR
)
echo.

echo 请输入新数值，或直接按回车以使用默认值

set /p "vadThresh=输入 vad_threshold（语音置信度，默认：0.4，范围：0~1）："
if defined vadThresh set "chunkParams=!chunkParams! --vad_threshold !vadThresh!"

set /p "vadEndThresh=输入 vad_end_threshold（语音是否结束置信度，默认：智能，范围：0.05~1）："
if defined vadEndThresh set "chunkParams=!chunkParams! --vad_end_threshold !vadEndThresh!"

set /p "minSpeech=输入 min_speech_duration_ms（移除短于此时长（毫秒）的语音块，默认：100）："
if defined minSpeech set "chunkParams=!chunkParams! --min_speech_duration_ms !minSpeech!"

set /p "minSilence=输入 min_silence_duration_ms（最短静音间隔时长（毫秒），默认：200）："
if defined minSilence set "chunkParams=!chunkParams! --min_silence_duration_ms !minSilence!"

set /p "keep=输入 keep_silence（在语音块前后扩展的时长（毫秒），默认：300）："
if defined keep set "chunkParams=!chunkParams! --keep_silence !keep!"

echo 已设置自定义 VAD 参数

echo.

REM --- 询问 ZCR (过零率) 参数 ---
:AskZCR

echo ======================================================================

set "zcrParams="

echo.
choice /c YN /m "是否开启过零率（ZCR）检测？（保护清辅音不被切断）"

if %ERRORLEVEL% == 2 (
    goto AskRefineTail
)

set "zcrParams= --zcr"

echo.

choice /c YN /m "是否启用自适应 ZCR 阈值？（Y=自动计算，zcr_threshold 兜底；N=仅使用zcr_threshold）"
if %ERRORLEVEL% == 1 set "zcrParams=!zcrParams! --auto_zcr"

echo.

echo 请输入 ZCR 手动阈值，或直接按回车以使用默认值（0.15）

set /p "zcrTh=输入 zcr_threshold："

if defined zcrTh set "zcrParams=!zcrParams! --zcr_threshold !zcrTh!"

echo 已设置 ZCR 参数

echo.

REM --- 4.5. 询问段尾精修参数 (只有启用了VAD才会走到这里) ---
:AskRefineTail
echo ======================================================================
set "tailParams="

echo.

choice /c YN /m "是否启用“段尾精修”功能？"

if %ERRORLEVEL% == 2 (
    goto AskBeamParams
)

REM 用户选择了开启精修
set "tailParams= --refine-tail"

echo.
choice /c YN /m "是否需要修改默认的段尾精修参数？"
if %ERRORLEVEL% == 2 (
    goto AskBeamParams
)

echo.

echo 请输入新数值，或直接按回车以使用默认值

set /p "t_per=输入 tail_percentile（自适应阈值，默认：20，值越大越容易判为静音）："
if defined t_per set "tailParams=!tailParams! --tail_percentile !t_per!"

set /p "t_off=输入 tail_offset（阈值偏移量，默认：0.05，值越大越容易判为静音）："
if defined t_off set "tailParams=!tailParams! --tail_offset !t_off!"

set /p "t_look=输入 tail_lookahead_ms（滞回检查时长（毫秒），默认：80）："
if defined t_look set "tailParams=!tailParams! --tail_lookahead_ms !t_look!"

set /p "t_safe=输入 tail_safety_margin_ms（安全边距（毫秒），默认：30）："
if defined t_safe set "tailParams=!tailParams! --tail_safety_margin_ms !t_safe!"

set /p "t_keep=输入 tail_min_keep_ms（强制保留时长（毫秒），默认：30）："
if defined t_keep set "tailParams=!tailParams! --tail_min_keep_ms !t_keep!"

echo 已设置段尾精修参数

echo.

REM --- 询问高级识别参数 ---
:AskBeamParams

echo 文件: !inputFile!

echo ======================================================================

echo.

echo 请输入新数值，或直接按回车以使用默认值

set "beamParams="  REM 首先确保变量为空

set /p "beamSize=输入 beam_size（集束搜索宽度，默认：4，范围：4~64（仅整数），更大的值可能更准确但更慢）："

if NOT "!beamSize!"=="" (
    set "beamParams= --beam !beamSize!"
)

echo 已设置高级识别参数

echo.

REM --- 新增：询问是否保留句末标点 ---
:AskPunctuation

set "puncOption="
choice /c YN /m "是否剔除句末标点？（Y=是，自动剔除句末标点；N=否，保留句末标点）"
if %ERRORLEVEL% == 2 set "puncOption= --no-remove-punc"

echo.

REM --- 询问是否启用调试模式---
:AskDebug

echo.

choice /c YN /m "是否启用调试模式？（保留临时文件并自动打开目录）"

if %ERRORLEVEL% == 1 set "debugOption= --debug"

REM --- 询问输出方式 ---
:AskOutputLoop

echo ======================================================================

echo 请选择您要生成的一种或多种文件格式：

echo.

echo   1. 完整的识别文本（.txt）

echo   2. 带时间戳的文本片段（.segments.txt）

echo   3. 带时间戳的文本片段并转换为字幕（.srt）

echo   4. 输出带时间戳的文本片段并转换为 VTT 字幕文件（.vtt）

echo   5. 输出带时间戳的文本片段并转换为由制表符分隔的 .tsv 文件（.tsv）

echo   6. 带时间戳的所有子词（.subwords.txt）

echo   7. 带时间戳的所有子词并转换为字幕（.subwords.srt）

echo   8. 带时间戳的所有子词并转换为 JSON 文件（.subwords.json）

echo   9. 生成逐字计时的卡拉OK式 ASS 字幕文件（.ass）

echo.

set "userChoice="
set /p "userChoice=请输入格式编号（可多选，如：136），或直接回车跳过："

rem 情况一：用户直接回车，输入为空
if not defined userChoice (
    echo.

    echo 【提示】您没有选择任何文件输出格式，识别结果将仅显示在控制台中

    set "outputOptions="
    goto Execute
)

rem 情况二：用户输入了内容，使用字符串替换法检查输入中是否包含任何无效字符
rem  例如，增加第7个选项，就在 VALID_OPTIONS 后面加上 7
set "checker=!userChoice!"
rem 依次将所有合法的数字从检查字符串中移除，在用户输入前加上一个不会被移除的字符X以防止变量中途变为空

set "checker=X!userChoice!"
set "checker=!checker:1=!"
set "checker=!checker:2=!"
set "checker=!checker:3=!"
set "checker=!checker:4=!"
set "checker=!checker:5=!"
set "checker=!checker:6=!"
set "checker=!checker:7=!"
set "checker=!checker:8=!"
set "checker=!checker:9=!"

rem 如果移除了所有合法数字后，字符串不等于X，说明含有非法字符
if not "!checker!"=="X" (
    cls
    goto AskOutputLoop
)

echo 输入有效，正在处理……

set "outputOptions="
if NOT "!userChoice:1=!"=="!userChoice!" set "outputOptions=!outputOptions! -text"
if NOT "!userChoice:2=!"=="!userChoice!" set "outputOptions=!outputOptions! -segment"
if NOT "!userChoice:3=!"=="!userChoice!" set "outputOptions=!outputOptions! -segment2srt"
if NOT "!userChoice:4=!"=="!userChoice!" set "outputOptions=!outputOptions! -segment2vtt"
if NOT "!userChoice:5=!"=="!userChoice!" set "outputOptions=!outputOptions! -segment2tsv"
if NOT "!userChoice:6=!"=="!userChoice!" set "outputOptions=!outputOptions! -subword"
if NOT "!userChoice:7=!"=="!userChoice!" set "outputOptions=!outputOptions! -subword2srt"
if NOT "!userChoice:8=!"=="!userChoice!" set "outputOptions=!outputOptions! -subword2json"
if NOT "!userChoice:9=!"=="!userChoice!" set "outputOptions=!outputOptions! -kass"

REM --- 执行最终的 Python 命令 ---
:Execute
echo.

echo ======================================================================

echo 正在开始识别，请稍候……

echo 最终执行的命令: python asr.py "!inputFile!" %chunkOption%!chunkParams!!zcrParams!!tailParams!!beamParams!!puncOption!!debugOption!!outputOptions!

echo ======================================================================

echo.

python asr.py "!inputFile!" %chunkOption%!chunkParams!!zcrParams!!tailParams!!beamParams!!puncOption!!debugOption!!outputOptions!

echo.

echo ======================================================================

echo 识别流程已结束，您的输出文件应该已保存在与输入文件相同的目录中

echo ======================================================================

echo.

echo 按任意键继续处理下一个文件，或直接关闭此窗口以退出……

pause > nul

goto GetFile