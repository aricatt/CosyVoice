FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 克隆 CosyVoice 项目
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# 安装 CosyVoice 依赖
COPY requirements.txt api-requirements.txt
RUN pip install -r CosyVoice/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install -r api-requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 设置 Python 路径
ENV PYTHONPATH="${PYTHONPATH}:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

# 复制 API 代码
COPY main.py .

# 运行 API 服务
CMD ["python3", "main.py"]
