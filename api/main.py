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

def postprocess(speech, max_val=0.8):
    """后处理音频数据"""
    logger.info(f"Postprocess input type: {type(speech)}")
    
    # 如果是生成器，转换为列表
    if hasattr(speech, '__iter__') and not isinstance(speech, (np.ndarray, torch.Tensor)):
        logger.info("Converting iterator to list")
        speech = list(speech)
    
    # 确保输入是 numpy 数组
    if isinstance(speech, torch.Tensor):
        logger.info("Converting torch.Tensor to numpy array")
        speech = speech.detach().cpu().numpy()
    elif not isinstance(speech, np.ndarray):
        logger.info(f"Converting {type(speech)} to numpy array")
        try:
            speech = np.array(speech, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to convert to numpy array: {str(e)}")
            raise ValueError(f"Could not convert {type(speech)} to numpy array: {str(e)}")
    
    logger.info(f"Array shape after conversion: {speech.shape}")
    
    # 确保是一维数组
    if speech.ndim == 2:
        logger.info("Squeezing 2D array to 1D")
        if speech.shape[1] == 1:  # 如果是 (N, 1) 形状
            speech = speech.squeeze(1)
        elif speech.shape[0] == 1:  # 如果是 (1, N) 形状
            speech = speech.squeeze(0)
        else:
            logger.warning(f"Unexpected 2D shape: {speech.shape}")
            # 如果是其他形状，我们取第一个通道
            speech = speech[:, 0]
    elif speech.ndim > 2:
        raise ValueError(f"Unexpected array dimensions: {speech.ndim}")
    
    # 音量归一化
    if np.abs(speech).max() > max_val:
        logger.info("Normalizing audio volume")
        speech = speech / np.abs(speech).max() * max_val
    
    # 添加静音片段
    silence_length = min(int(16000 * 0.2), len(speech) // 10)  # 静音长度不超过音频长度的1/10
    silence = np.zeros(silence_length, dtype=speech.dtype)
    speech = np.concatenate([silence, speech, silence])
    
    logger.info(f"Final audio shape: {speech.shape}")
    logger.info("Postprocessing completed successfully")
    return speech

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
            
            # 确保音频长度合适
            min_length = target_sr  # 至少 1 秒
            max_length = target_sr * 30  # 最多 30 秒
            
            if len(audio) < min_length:
                # 在前后添加静音以达到最小长度
                padding_length = (min_length - len(audio)) // 2
                silence = np.zeros(padding_length, dtype=audio.dtype)
                audio = np.concatenate([silence, audio, silence])
            elif len(audio) > max_length:
                logger.warning(f"Audio too long ({len(audio)/target_sr:.1f}s), truncating to 30s")
                audio = audio[:max_length]
            
            # 删除临时文件
            os.remove("temp.wav")
            
            logger.info(f"Preprocessed audio shape: {audio.shape}")
            return audio
            
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
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
        
        # 转换为PyTorch tensor并添加维度
        audio = torch.from_numpy(audio).float()
        audio = audio.unsqueeze(0)  # 添加batch维度 [1, length]
        
        logger.info(f"Final model input shape: {audio.shape}")
        return audio
        
    except Exception as e:
        logger.error(f"Error in prepare_model_input: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

@app.post("/tts/zero_shot")
async def zero_shot_tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    stream: str = Form("false"),
    speed: str = Form("1.0")
):
    """零样本语音合成接口"""
    try:
        if not model_manager.model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # 读取上传的音频文件
        audio_data = await prompt_audio.read()
        
        # 预处理音频
        prompt_speech = preprocess_audio(audio_data)
        prompt_speech = postprocess(prompt_speech)
        
        # 准备模型输入
        prompt_speech = prepare_model_input(prompt_speech)
        
        # 转换参数
        stream_bool = stream.lower() == "true"
        speed_float = float(speed)
        
        # 生成语音
        try:
            logger.info("Starting inference_zero_shot")
            audio_chunks = model_manager.inference_zero_shot(
                text, prompt_text, prompt_speech,
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
            logger.error(f"Error in inference_zero_shot: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error in zero_shot_tts: {str(e)}")
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
        
        # 读取上传的音频文件
        audio_data = await prompt_audio.read()
        
        # 预处理音频
        prompt_speech = preprocess_audio(audio_data)
        prompt_speech = postprocess(prompt_speech)
        
        # 准备模型输入
        prompt_speech = prepare_model_input(prompt_speech)
        
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
