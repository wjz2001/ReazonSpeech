@echo off
rem 切换代码页以正确显示中文字符
chcp 65001 > nul
setlocal
title ReazonSpeech 语音识别助手

REM --- 1. 检查并启动虚拟环境 ---
echo 正在检查虚拟环境...
if not exist ".\venv\Scripts\activate.bat" (
    echo.
    echo [错误] 未在 '.\venv\Scripts\activate.bat' 找到虚拟环境。
    echo 请确保您正在项目的根目录下运行此脚本。
    echo.
    pause
    exit /b 1
)

echo 正在激活虚拟环境...
call ".\venv\Scripts\activate.bat"
echo.

REM --- 2. 获取音视频文件路径 ---
:GetFile
echo ----------------------------------------------------------------------
set "inputFile="
set /p "inputFile=请输入您的音频/视频文件的完整路径，然后按回车键： "

if not exist "%inputFile%" (
    echo.
    echo [错误] 文件未找到。请检查路径后重试。
    echo.
    goto GetFile
)
echo 已找到文件: %inputFile%
echo.

REM --- 3. 询问是否分块 ---
:AskChunk
echo ----------------------------------------------------------------------
set "chunkOption="
choice /c YN /m "是否使用智能分块？ (Y=是, 推荐长文件使用; N=否, 适用于短文件)"

if %ERRORLEVEL% == 2 (
    set "chunkOption=--no-chunk"
    goto AskBeamSize
)

REM --- 4. (如果分块) 询问是否修改VAD参数 ---
:AskVadParams
echo.
set "chunkParams="
choice /c YN /m "是否需要修改默认的VAD参数？"

if %ERRORLEVEL% == 2 (
    echo 正在使用默认VAD参数。
    goto AskBeamSize
)

echo.
echo 请输入新数值，或直接按回车以使用默认值。
set /p "vadThresh=输入 vad_threshold (默认: 0.2): "
if not defined vadThresh set vadThresh=0.2

set /p "negThresh=输入 vad_neg_threshold (语音结束阈值, 默认: 0.1): "
if not defined negThresh set negThresh=0.1

set /p "minSilence=输入 vad_min_silence_ms (最小静音时长, 默认: 250): "
if not defined minSilence set minSilence=250

set /p "minSpeech=输入 min_speech_duration_ms (默认: 100): "
if not defined minSpeech set minSpeech=100

set /p "keep=输入 keep_silence (默认: 300): "
if not defined keep set keep=300

set "chunkParams=--vad_threshold %vadThresh% --min_speech_duration_ms %minSpeech% --keep_silence %keep%"
echo 已设置自定义参数。
echo.

REM --- 5. 询问是否使用Beam Search ---
:AskBeamSize
echo ----------------------------------------------------------------------
set "beamParam="
set "beamSize="
choice /c YN /m "是否启用Beam Search以提升准确率(速度会变慢)？"

if %ERRORLEVEL% == 2 (
    echo 未启用Beam Search。
    goto AskOutput
)
echo.
set /p "beamSize=请输入Beam Size的大小 (推荐值为5或10): "
if not defined beamSize set beamSize=5
set "beamParam=--beam_size %beamSize%"
echo.

REM --- 6. 询问输出方式 ---
:AskOutput
echo ----------------------------------------------------------------------
echo 请选择一个输出格式：
echo.
echo   1. 纯文本 (.txt)
echo   2. 带时间戳的片段 (.segments.txt)
echo   3. 将片段转为 SRT 字幕 (.srt)
echo   4. 带时间戳的子词 (.subwords.txt)
echo   5. 将子词转为 SRT 字幕 (.subwords.srt)
echo   6. 卡拉OK式ASS字幕 (.k.ass)
echo.

:ChoiceOutput
set "outputOption="
choice /c 123456 /m "请输入您的选择 [1-6]："

if %ERRORLEVEL% == 6 set "outputOption=-kass"
if %ERRORLEVEL% == 5 set "outputOption=-subword2srt"
if %ERRORLEVEL% == 4 set "outputOption=-subword"
if %ERRORLEVEL% == 3 set "outputOption=-segment2srt"
if %ERRORLEVEL% == 2 set "outputOption=-segment"
if %ERRORLEVEL% == 1 set "outputOption=-text"

REM --- 7. 执行最终的 Python 命令 ---
echo.
echo ======================================================================
echo 正在开始识别... 请稍候。
echo ======================================================================
echo.

python asr.py "%inputFile%" %chunkOption% %chunkParams% %beamParam% %outputOption%

echo.
echo ======================================================================
echo 识别流程已结束。
echo 您的输出文件应该已保存在与输入文件相同的目录中。
echo ======================================================================
echo.
echo 按任意键退出...
pause > nul
endlocal