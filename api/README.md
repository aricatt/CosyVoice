# CosyVoice API

这是 CosyVoice 的 API 服务封装，提供了远程调用接口。

## 功能特点

- 支持零样本语音克隆（Zero-shot Voice Cloning）
- 支持指令控制语音生成（Instruction-based TTS）
- 支持流式生成
- 提供 RESTful API 接口
- 自动生成 API 文档
- 支持 GPU 加速

## 安装和运行

### 使用 Docker（推荐）

1. 构建 Docker 镜像：
```bash
docker build -t cosyvoice-api .
```

2. 运行容器：
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/app/pretrained_models \
  --name cosyvoice-api \
  cosyvoice-api
```

### 手动安装

1. 安装依赖：
```bash
pip install -r requirements.txt
pip install -r ../requirements.txt
```

2. 运行服务：
```bash
python main.py
```

## API 接口

访问 `http://localhost:8000/docs` 查看完整的 API 文档。

### 1. 零样本语音克隆

**接口**：`POST /tts/zero_shot`

**参数**：
- `text`: 要转换的文本
- `prompt_text`: 提示文本（可选）
- `prompt_audio`: 提示音频文件
- `stream`: 是否流式生成
- `speed`: 语速（默认 1.0）

**示例**：
```python
import requests

url = "http://localhost:8000/tts/zero_shot"
files = {
    'prompt_audio': open('prompt.wav', 'rb')
}
data = {
    'text': '你好，世界！',
    'prompt_text': '你好',
    'stream': False,
    'speed': 1.0
}

response = requests.post(url, files=files, data=data)
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### 2. 指令控制语音生成

**接口**：`POST /tts/instruct`

**参数**：
- `text`: 要转换的文本
- `instruct_text`: 指令文本
- `prompt_audio`: 提示音频文件
- `stream`: 是否流式生成
- `speed`: 语速（默认 1.0）

**示例**：
```python
import requests

url = "http://localhost:8000/tts/instruct"
files = {
    'prompt_audio': open('prompt.wav', 'rb')
}
data = {
    'text': '你好，世界！',
    'instruct_text': '请用开心的语气说',
    'stream': False,
    'speed': 1.0
}

response = requests.post(url, files=files, data=data)
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

## 注意事项

1. 确保已下载所需的模型文件到 `pretrained_models` 目录
2. 使用 GPU 时需要安装对应版本的 CUDA
3. 音频文件格式支持 wav，采样率会自动转换为 16kHz
4. 建议在生产环境中使用 Docker 部署
