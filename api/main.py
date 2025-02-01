import logging
import os
import io
import numpy as np
import torch
import torchaudio
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
from tqdm import tqdm

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
    
    def inference_zero_shot(self, text, prompt_text, prompt_speech, stream=False, speed=1.0):
        """封装zero-shot推理，确保输出格式正确"""
        audio_chunks = self.model.inference_zero_shot(
            text, prompt_text, prompt_speech,
            stream=stream, speed=speed
        )
        
        # 处理生成器输出
        for chunk in audio_chunks:
            if "tts_speech" in chunk:
                chunk["tts_speech"] = self.process_model_output(chunk["tts_speech"])
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
                chunk["tts_speech"] = self.process_model_output(chunk["tts_speech"])
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

def postprocess(speech, max_val=0.95):
    """后处理音频数据"""
    try:
        # 记录输入信息
        logger.info(f"Postprocess input type: {type(speech)}")
        
        # 确保是numpy数组
        if isinstance(speech, torch.Tensor):
            speech = speech.detach().cpu().numpy()
        elif not isinstance(speech, np.ndarray):
            speech = np.array(speech)
        
        logger.info(f"Array shape after conversion: {speech.shape}")
        
        # 如果是2D数组，转换为1D
        if speech.ndim == 2:
            logger.info("Squeezing 2D array to 1D")
            speech = speech.squeeze()
        
        # 去除DC偏移
        speech = speech - np.mean(speech)
        
        # 音量归一化
        if np.abs(speech).max() > 0:
            logger.info("Normalizing audio volume")
            speech = speech / np.abs(speech).max() * max_val
        
        # 应用淡入淡出以避免爆音
        fade_length = min(1600, len(speech) // 20)  # 淡入淡出长度，最长0.1秒
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        speech[:fade_length] *= fade_in
        speech[-fade_length:] *= fade_out
        
        # 应用低通滤波器去除高频噪声
        from scipy import signal
        b, a = signal.butter(4, 0.95, 'low')
        speech = signal.filtfilt(b, a, speech)
        
        logger.info(f"Final audio shape: {speech.shape}")
        logger.info("Postprocessing completed successfully")
        return speech
        
    except Exception as e:
        logger.error(f"Error in postprocess: {str(e)}")
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
            
            # 去除静音部分
            threshold = 0.005  # 降低静音阈值
            energy = np.abs(audio)
            mask = energy > threshold
            if np.any(mask):
                start = np.argmax(mask)
                end = len(audio) - np.argmax(mask[::-1])
                # 在有效音频前后保留一小段静音
                start = max(0, start - int(target_sr * 0.1))  # 前面保留0.1秒
                end = min(len(audio), end + int(target_sr * 0.1))  # 后面保留0.1秒
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
                    start = max(0, start - int(target_sr * 0.1))
                    end = min(len(audio), end + int(target_sr * 0.1))
                    audio = audio[start:end]
                
                # 如果还是太短，才进行重复
                if len(audio) < min_length:
                    repeats = int(np.ceil(min_length / len(audio)))
                    audio = np.tile(audio, repeats)[:min_length]
                    logger.warning(f"Audio too short ({len(audio)/target_sr:.1f}s), repeated to {min_length/target_sr:.1f}s")
            elif len(audio) > max_length:
                # 取中间部分，保持句子的完整性
                center = len(audio) // 2
                half_length = max_length // 2
                start = center - half_length
                end = center + half_length
                audio = audio[start:end]
                logger.warning(f"Audio too long ({len(audio)/target_sr:.1f}s), truncated to middle {max_length/target_sr:.1f}s")
            
            # 归一化音量
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            # 去除DC偏移
            audio = audio - np.mean(audio)
            
            # 应用预加重滤波器增强高频
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # 应用淡入淡出
            fade_length = min(1600, len(audio) // 20)  # 最长0.1秒
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            audio = audio.copy()  # 创建副本以避免修改原始数据
            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out
            
            # 删除临时文件
            os.remove("temp.wav")
            
            logger.info(f"Preprocessed audio shape: {audio.shape}")
            return audio
            
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

async def zero_shot_tts(text, audio_data, prompt_text=None):
    """Zero-shot TTS 生成"""
    try:
        logger.info("Starting inference_zero_shot")
        
        # 预处理音频数据
        audio = preprocess_audio(audio_data)
        audio = prepare_model_input(audio)
        
        # 如果没有提供 prompt_text，使用生成文本
        if not prompt_text:
            prompt_text = text
        
        # 检查文本长度
        if len(text) < len(prompt_text):
            logger.warning(f"synthesis text {text} too short than prompt text {prompt_text}, this may lead to bad performance")
        
        # 记录生成参数
        logger.info(f"synthesis text {text}")
        
        # 生成语音
        audio_chunks = model_manager.inference_zero_shot(
            text=text,
            prompt_text=prompt_text,
            prompt_speech=audio,
            stream=False,  # 关闭流式生成，一次性生成完整音频
            speed=1.0
        )
        
        logger.info("Got audio chunks from model")
        logger.info(f"audio_chunks type: {type(audio_chunks)}")
        
        # 如果不是生成器，直接处理
        if not hasattr(audio_chunks, '__iter__'):
            speech = audio_chunks['tts_speech']
            processed_speech = postprocess(speech)
            final_audio = (processed_speech * 32767).astype(np.int16)
            duration = len(final_audio) / 16000
            logger.info(f"Final audio duration: {duration:.2f} seconds")
            return final_audio
        
        # 处理生成的音频块
        all_chunks = []
        total_duration = 0
        start_time = time.time()
        
        for i, chunk in enumerate(tqdm(audio_chunks)):
            if 'tts_speech' not in chunk:
                logger.warning(f"Missing tts_speech in chunk {i}")
                continue
            
            speech = chunk['tts_speech']
            processed_speech = postprocess(speech)
            all_chunks.append(processed_speech)
            
            chunk_duration = len(processed_speech) / 16000
            total_duration += chunk_duration
            rtf = (time.time() - start_time) / total_duration
            logger.info(f"yield speech len {chunk_duration:.2f}, rtf {rtf}")
        
        # 合并所有音频块
        if not all_chunks:
            raise ValueError("No valid audio chunks generated")
        
        final_audio = np.concatenate(all_chunks)
        final_audio = postprocess(final_audio)  # 对整体再次进行后处理
        final_audio = (final_audio * 32767).astype(np.int16)
        
        duration = len(final_audio) / 16000
        logger.info(f"Final audio duration: {duration:.2f} seconds")
        return final_audio
        
    except Exception as e:
        logger.error(f"Error in zero_shot_tts: {str(e)}")
        raise

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
        audio = await zero_shot_tts(text, audio_data, prompt_text)
        
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
        buffer.write((16000).to_bytes(4, 'little'))  # 采样率
        buffer.write((32000).to_bytes(4, 'little'))  # 字节率
        buffer.write((2).to_bytes(2, 'little'))  # 块对齐
        buffer.write((16).to_bytes(2, 'little'))  # 位深度
        buffer.write(b'data')
        buffer.write((len(audio) * 2).to_bytes(4, 'little'))  # 数据大小
        
        # 写入音频数据
        buffer.write(audio.tobytes())
        
        # 将缓冲区指针移到开始
        buffer.seek(0)
        
        # 读取所有数据
        content = buffer.read()
        
        # 返回完整的WAV文件
        return Response(
            content=content,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(content)),
                "Content-Disposition": 'attachment; filename="generated.wav"'
            }
        )
        
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
        
        prompt_speech = postprocess(prompt_speech)
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
                    speech = postprocess(speech)
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
            wav_header.write((16000).to_bytes(4, 'little'))  # 采样率
            wav_header.write((32000).to_bytes(4, 'little'))  # 字节率
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
