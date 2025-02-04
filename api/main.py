import logging
import os
import io
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Any, Generator, Union
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, Response
from tqdm import tqdm
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn
import time
from datetime import datetime
import soundfile as sf


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
    
    def load_model(self):
        try:
            model_dir = '../pretrained_models/CosyVoice2-0.5B'
            if not os.path.exists(model_dir):
                raise RuntimeError(f"Model directory not found: {model_dir}")
            self.model = CosyVoice2(model_dir, fp16=torch.cuda.is_available())
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def process_model_output(self, speech):
        """处理模型输出的音频数据"""
        if isinstance(speech, torch.Tensor):
            speech = speech.detach().cpu().numpy()
        elif not isinstance(speech, np.ndarray):
            speech = np.array(speech)
        return speech
    
    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0):
        """Zero-shot TTS inference"""
        # 生成语音
        audio_chunks = self.model.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=stream,
            speed=speed
        )
        
        # 如果不是生成器，直接返回
        if not hasattr(audio_chunks, '__iter__'):
            return audio_chunks
        
        # 如果是生成器，逐个返回
        for chunk in audio_chunks:
            yield chunk
    
    def inference_instruct2(self, text, instruct_text, prompt_speech, stream=False, speed=1.0):
        """封装instruct推理，确保输出格式正确"""
        audio_chunks = self.model.inference_instruct2(
            text, instruct_text, prompt_speech,
            stream=stream, speed=speed
        )
        
        # 处理生成器输出
        for chunk in audio_chunks:
            if "tts_speech" in chunk:
                chunk["tts_speech"] = chunk["tts_speech"].numpy().flatten()
            yield chunk
    
    def cleanup(self):
        self.model = None
        logger.info("Model resources cleaned up!")

# 创建模型管理器实例
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 启动时加载模型
    model_manager.load_model()
    yield
    # 关闭时清理资源
    model_manager.cleanup()

app = FastAPI(
    title="CosyVoice API",
    description="API for CosyVoice text-to-speech service",
    version="1.0.0",
    lifespan=lifespan
)

class TTSRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = None
    instruct_text: Optional[str] = None
    stream: bool = False
    speed: float = 1.0

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """与 webui.py 保持一致的后处理"""
    return speech
    try:
        # 如果是 numpy 数组，转换为 tensor
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        
        # 确保是二维数组 [1, T]
        if speech.ndim == 1:
            speech = speech.unsqueeze(0)
            
        # 裁剪静音
        speech_np = speech.numpy().flatten()
        speech_trimmed, _ = librosa.effects.trim(
            speech_np, 
            top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        speech = torch.from_numpy(speech_trimmed).unsqueeze(0)
        
        # 音量归一化
        max_val = 0.8
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
            
        # 添加尾部静音
        speech = torch.concat([speech, torch.zeros(1, int(16000 * 0.2))], dim=1)
        
        return speech
        
    except Exception as e:
        logger.error(f"Error in postprocess: {str(e)}")
        raise

def postprocess_audio(audio):
    """后处理音频数据"""
    try:
        logger.info(f"Postprocess input type: {type(audio)}")
        
        # 如果是PyTorch tensor，转换为numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
            logger.info(f"Array shape after conversion: {audio.shape}")
        
        # 确保是一维数组
        if audio.ndim == 2:
            if audio.shape[0] == 1:  # (1, N) 形状
                audio = audio.squeeze(0)
                logger.info("Squeezing 2D array to 1D")
        
        # 裁剪超出范围的值
        audio = np.clip(audio, -1.0, 1.0)
        
        # 归一化音量
        logger.info("Normalizing audio volume")
        if np.abs(audio).max() > 0:
            # 使用 RMS 归一化而不是峰值归一化
            rms = np.sqrt(np.mean(audio ** 2))
            target_rms = 0.2
            audio = audio * (target_rms / rms)
            # 确保峰值不超过阈值
            peak = np.abs(audio).max()
            if peak > 0.95:
                audio = audio * (0.95 / peak)
        
        # 去除DC偏移
        audio = audio - np.mean(audio)
        
        # 应用预加重滤波器增强高频特征
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # 应用更平滑的淡入淡出
        fade_length = min(3200, len(audio) // 10)  # 最长0.2秒
        fade_in = np.sqrt(np.linspace(0, 1, fade_length))  # 使用平方根曲线使淡变更自然
        fade_out = np.sqrt(np.linspace(1, 0, fade_length))
        audio = audio.copy()
        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out
        
        logger.info(f"Final audio shape: {audio.shape}")
        logger.info("Postprocessing completed successfully")
        return audio
        
    except Exception as e:
        logger.error(f"Error in postprocess_audio: {str(e)}")
        raise

def prepare_model_input(audio):
    """准备模型输入"""
    try:
        # 确保是numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # 记录输入形状
        logger.info(f"Input audio shape: {audio.shape}")
        logger.info(f"Input audio type: {type(audio)}")
        
        # 确保是一维数组
        if audio.ndim > 1:
            audio = audio.squeeze()
            logger.info(f"After squeeze shape: {audio.shape}")
        
        # 归一化到 [-1, 1] 范围
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # 转换为PyTorch tensor并添加维度
        audio = torch.from_numpy(audio).float()
        audio = audio.unsqueeze(0)  # 添加batch维度 [1, length]
        
        logger.info(f"Final model input shape: {audio.shape}")
        return audio
        
    except Exception as e:
        logger.error(f"Error in prepare_model_input: {str(e)}")
        raise

def preprocess_audio(audio_data, target_sr=16000):
    """预处理音频数据"""
    try:
        # 读取音频数据
        with io.BytesIO(audio_data) as audio_io:
            # 保存到临时文件
            with open("temp.wav", "wb") as f:
                f.write(audio_io.read())
            
            # 使用 load_wav 函数读取
            audio = load_wav("temp.wav", target_sr=target_sr)
            logger.info(f"Loaded audio shape: {audio.shape}")
            
            # 如果是PyTorch tensor，转换为numpy数组
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            
            # 确保是一维数组
            if audio.ndim == 2:
                if audio.shape[1] == 1:  # (N, 1) 形状
                    audio = audio.squeeze(1)
                elif audio.shape[0] == 1:  # (1, N) 形状
                    audio = audio.squeeze(0)
                else:
                    logger.warning(f"Multi-channel audio detected, using first channel. Shape: {audio.shape}")
                    audio = audio[:, 0]
            
            # 去除静音部分，但保留更多的上下文
            threshold = 0.003  # 降低静音阈值，保留更多声音特征
            energy = np.abs(audio)
            mask = energy > threshold
            if np.any(mask):
                start = np.argmax(mask)
                end = len(audio) - np.argmax(mask[::-1])
                # 在有效音频前后保留更多的上下文
                start = max(0, start - int(target_sr * 0.2))  # 前面保留0.2秒
                end = min(len(audio), end + int(target_sr * 0.2))  # 后面保留0.2秒
                audio = audio[start:end]
            
            # 确保音频长度合适
            min_length = target_sr * 3  # 至少 3 秒
            max_length = target_sr * 10  # 最多 10 秒
            
            if len(audio) < min_length:
                # 如果音频太短，先尝试降低静音阈值
                threshold = 0.001
                energy = np.abs(audio)
                mask = energy > threshold
                if np.any(mask):
                    start = np.argmax(mask)
                    end = len(audio) - np.argmax(mask[::-1])
                    start = max(0, start - int(target_sr * 0.2))
                    end = min(len(audio), end + int(target_sr * 0.2))
                    audio = audio[start:end]
                
                # 如果还是太短，使用重叠拼接而不是简单重复
                if len(audio) < min_length:
                    logger.warning(f"Audio too short ({len(audio)/target_sr:.1f}s), using overlap-add to extend")
                    overlap = len(audio) // 4  # 25% 重叠
                    extended = np.zeros(min_length)
                    pos = 0
                    window = np.hanning(overlap * 2)
                    while pos < min_length:
                        if pos + len(audio) <= min_length:
                            if pos == 0:
                                extended[pos:pos + len(audio)] = audio
                            else:
                                # 应用交叉淡入淡出
                                extended[pos:pos + overlap] = extended[pos:pos + overlap] * window[:overlap] + audio[:overlap] * window[overlap:]
                                extended[pos + overlap:pos + len(audio)] = audio[overlap:]
                        else:
                            break
                        pos += len(audio) - overlap
                    audio = extended[:min_length]
                    
            elif len(audio) > max_length:
                # 取中间部分，保持句子的完整性
                center = len(audio) // 2
                half_length = max_length // 2
                start = center - half_length
                end = center + half_length
                audio = audio[start:end]
                logger.warning(f"Audio too long ({len(audio)/target_sr:.1f}s), truncated to middle {max_length/target_sr:.1f}s")
            
            # 归一化音量，但保持动态范围
            if np.abs(audio).max() > 0:
                # 使用 RMS 归一化而不是峰值归一化
                rms = np.sqrt(np.mean(audio ** 2))
                target_rms = 0.2
                audio = audio * (target_rms / rms)
                # 确保峰值不超过阈值
                peak = np.abs(audio).max()
                if peak > 0.95:
                    audio = audio * (0.95 / peak)
            
            # 去除DC偏移
            audio = audio - np.mean(audio)
            
            # 应用预加重滤波器增强高频特征
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # 应用更平滑的淡入淡出
            fade_length = min(3200, len(audio) // 10)  # 最长0.2秒
            fade_in = np.sqrt(np.linspace(0, 1, fade_length))  # 使用平方根曲线使淡变更自然
            fade_out = np.sqrt(np.linspace(1, 0, fade_length))
            audio = audio.copy()
            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out
            
            # 删除临时文件
            os.remove("temp.wav")
            
            logger.info(f"Preprocessed audio shape: {audio.shape}")
            return audio
            
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        raise

async def zero_shot_tts(text, audio_data, prompt_text=None):
    """Zero-shot TTS 生成"""
    try:
        # 加载并处理音频
        with io.BytesIO(audio_data) as audio_io:
            # 创建带时间戳的临时文件名
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_wav_path = os.path.abspath(f"temp-{current_time}.wav")
            
            # 保存到临时文件
            with open(temp_wav_path, "wb") as f:
                f.write(audio_io.read())
            
            logger.info(f"Saved prompt audio to: {temp_wav_path}")
            
            # 加载音频
            prompt_speech = load_wav(temp_wav_path, target_sr=16000)
            
            # 不删除文件，保留供后续分析
            # os.remove(temp_wav_path)
        
        # 生成语音
        audio_chunks = model_manager.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech,
            stream=False,
            speed=1.0
        )
        
        # 如果不是生成器，直接处理
        if not hasattr(audio_chunks, '__iter__'):
            if 'tts_speech' not in audio_chunks:
                raise ValueError("No tts_speech in model output")

            raw_speech = audio_chunks['tts_speech']
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.abspath(f"webui_output-{current_time}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始音频
            raw_wav_path = os.path.join(output_dir, "raw_speecha.wav")
            sf.write(raw_wav_path, raw_speech.numpy().flatten(), 24000)
            logging.info(f"A Saved raw speech to: {raw_wav_path}")

            yield audio_chunks['tts_speech']
        
        # 处理生成器输出
        for chunk in tqdm(audio_chunks):
            if 'tts_speech' not in chunk:
                logger.warning("No tts_speech in chunk")
                continue

            raw_speech = chunk['tts_speech']
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.abspath(f"webui_output-{current_time}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始音频
            raw_wav_path = os.path.join(output_dir, "raw_speechb.wav")
            sf.write(raw_wav_path, raw_speech.numpy().flatten(), 24000)
            logging.info(f"B Saved raw speech to: {raw_wav_path}")
            yield chunk['tts_speech']
        
    except Exception as e:
        logger.error(f"Error in zero_shot_tts: {str(e)}")
        raise

async def inference_zero_shot(text, audio_data, prompt_text=None):
    """零样本语音合成"""
    try:
        # 加载并处理音频
        with io.BytesIO(audio_data) as audio_io:
            # 创建带时间戳的临时文件名
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_wav_path = os.path.abspath(f"temp-{current_time}.wav")
            
            # 保存到临时文件
            with open(temp_wav_path, "wb") as f:
                f.write(audio_io.read())
            
            logger.info(f"Saved prompt audio to: {temp_wav_path}")
            
            # 加载音频
            prompt_speech = load_wav(temp_wav_path, target_sr=16000)
            
            # 不删除文件，保留供后续分析
            # os.remove(temp_wav_path)
        
        # 后处理音频
        prompt_speech = postprocess(prompt_speech)
        
        # 确保文本没有句号结尾
        if text.endswith('。'):
            text = text[:-1]
        elif text.endswith('.'):
            text = text[:-1]
        
        # 合成语音
        logger.info(f"synthesis text {text}")
        audio_chunks = model_manager.inference_zero_shot(
            text, prompt_text, prompt_speech,
            stream=False,  # 关闭流式生成，一次性生成完整音频
            speed=1.0,
            top_k=50,  # 增加采样多样性
            top_p=0.8,  # 控制采样概率分布
            temperature=0.7  # 稍微提高采样温度
        )
        logger.info("Got audio chunks from model")
        logger.info(f"audio_chunks type: {type(audio_chunks)}")
        
        # 处理音频块
        for i, chunk in enumerate(tqdm(audio_chunks)):
            # 检查音频值范围
            min_val = torch.min(chunk)
            max_val = torch.max(chunk)
            if min_val < -1.0 or max_val > 1.0:
                logger.warning(f"Audio values out of range: min={min_val}, max={max_val}")
                chunk = torch.clamp(chunk, -1.0, 1.0)
            
            # 后处理
            # chunk = postprocess(chunk)
            yield chunk
            
    except Exception as e:
        logger.error(f"Error in inference_zero_shot: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

@app.post("/tts/zero_shot")
async def zero_shot_tts_api(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    stream: str = Form(default="false"),
    speed: str = Form(default="1.0")
):
    """零样本语音合成接口"""
    try:
        if not model_manager.model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # 记录输入参数
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"Prompt text length: {len(prompt_text)}")
        logger.info(f"Text: {text}\n")
        logger.info(f"Prompt text: {prompt_text}")
        
        # 读取上传的音频文件
        audio_data = await prompt_audio.read()
        logger.info(f"Uploaded audio size: {len(audio_data)} bytes")
        
        # 生成语音
        audio_chunks = []
        async for chunk in zero_shot_tts(text, audio_data, prompt_text):
            # 如果是 tensor，转换为 numpy
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().numpy()
            audio_chunks.append(chunk.flatten())
            
        # 合并所有音频块
        audio = np.concatenate(audio_chunks)
            
        # 创建内存缓冲区
        buffer = io.BytesIO()
        
        # 写入WAV文件头
        buffer.write(b'RIFF')
        size = 36 + len(audio) * 2
        buffer.write(size.to_bytes(4, 'little'))  # 文件大小
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write((16).to_bytes(4, 'little'))  # fmt chunk size
        buffer.write((1).to_bytes(2, 'little'))  # 音频格式 (PCM)
        buffer.write((1).to_bytes(2, 'little'))  # 通道数
        buffer.write((24000).to_bytes(4, 'little'))  # 采样率
        buffer.write((48000).to_bytes(4, 'little'))  # 字节率 (采样率 * 2)
        buffer.write((2).to_bytes(2, 'little'))  # 块对齐
        buffer.write((16).to_bytes(2, 'little'))  # 位深度
        buffer.write(b'data')
        buffer.write((len(audio) * 2).to_bytes(4, 'little'))  # 数据大小
        
        # 写入音频数据
        audio_int16 = (audio * 32767).astype(np.int16)
        buffer.write(audio_int16.tobytes())
        
        # 返回音频文件
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"Error in zero_shot_tts_api: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/instruct")
async def instruct_tts(
    text: str = Form(...),
    instruct_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    stream: str = Form("false"),
    speed: str = Form("1.0")
):
    """指令控制的语音合成接口"""
    try:
        if not model_manager.model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # 记录输入参数
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"Instruct text length: {len(instruct_text)}")
        logger.info(f"Text: {text}")
        logger.info(f"Instruct text: {instruct_text}")
        
        # 读取上传的音频文件
        audio_data = await prompt_audio.read()
        logger.info(f"Uploaded audio size: {len(audio_data)} bytes")
        
        # 预处理音频
        prompt_speech = preprocess_audio(audio_data)
        logger.info(f"After preprocess shape: {prompt_speech.shape}")
        logger.info(f"Audio duration: {len(prompt_speech)/16000:.2f} seconds")
        
        prompt_speech = postprocess_audio(prompt_speech)
        logger.info(f"After postprocess shape: {prompt_speech.shape}")
        
        # 准备模型输入
        prompt_speech = prepare_model_input(prompt_speech)
        logger.info(f"Final input shape: {prompt_speech.shape}")
        
        # 转换参数
        stream_bool = stream.lower() == "true"
        speed_float = float(speed)
        
        # 生成语音
        try:
            logger.info("Starting inference_instruct2")
            audio_chunks = model_manager.inference_instruct2(
                text, instruct_text, prompt_speech,
                stream=stream_bool, speed=speed_float
            )
            logger.info("Got audio chunks from model")
            
            # 检查audio_chunks的类型
            logger.info(f"audio_chunks type: {type(audio_chunks)}")
            
            # 处理生成的音频
            all_audio_data = []
            for i, chunk in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i}")
                # 记录chunk的内容
                logger.info(f"Chunk keys: {chunk.keys()}")
                
                # 处理音频数据
                speech = chunk["tts_speech"]  # 可能是 tensor
                logger.info(f"Speech type before processing: {type(speech)}")
                if hasattr(speech, 'shape'):
                    logger.info(f"Speech shape: {speech.shape}")
                
                # 确保是numpy数组
                if isinstance(speech, torch.Tensor):
                    logger.info("Converting torch.Tensor to numpy array")
                    speech = speech.detach().cpu().numpy()
                elif not isinstance(speech, np.ndarray):
                    logger.info(f"Converting {type(speech)} to numpy array")
                    speech = np.array(speech)
                
                logger.info(f"Speech type after conversion: {type(speech)}")
                if hasattr(speech, 'shape'):
                    logger.info(f"Speech shape after conversion: {speech.shape}")
                
                # 应用后处理
                try:
                    speech = postprocess_audio(speech)
                    logger.info("Postprocessing successful")
                except Exception as e:
                    logger.error(f"Error in postprocessing: {str(e)}")
                    raise
                
                # 转换为16位整数
                speech = (speech * 32767).astype(np.int16)
                all_audio_data.append(speech)
            
            # 合并所有音频数据
            final_audio = np.concatenate(all_audio_data)
            logger.info(f"Final audio duration: {len(final_audio)/16000:.2f} seconds")
            
            # 创建WAV文件头
            wav_header = io.BytesIO()
            wav_header.write(b'RIFF')
            wav_header.write((36 + len(final_audio) * 2).to_bytes(4, 'little'))  # 文件大小
            wav_header.write(b'WAVE')
            wav_header.write(b'fmt ')
            wav_header.write((16).to_bytes(4, 'little'))  # fmt chunk size
            wav_header.write((1).to_bytes(2, 'little'))  # 音频格式 (PCM)
            wav_header.write((1).to_bytes(2, 'little'))  # 通道数
            wav_header.write((24000).to_bytes(4, 'little'))  # 采样率
            wav_header.write((48000).to_bytes(4, 'little'))  # 字节率 (采样率 * 2)
            wav_header.write((2).to_bytes(2, 'little'))  # 块对齐
            wav_header.write((16).to_bytes(2, 'little'))  # 位深度
            wav_header.write(b'data')
            wav_header.write((len(final_audio) * 2).to_bytes(4, 'little'))  # 数据大小
            
            async def audio_stream():
                # 发送WAV文件头
                yield wav_header.getvalue()
                # 发送音频数据
                yield final_audio.tobytes()
            
            # 返回音频流
            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav"
            )
        
        except Exception as e:
            logger.error(f"Error in inference_instruct2: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error in instruct_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8507,
        reload=False
    )
