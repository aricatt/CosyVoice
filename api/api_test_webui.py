import gradio as gr
import requests
import json
import os
import numpy as np
import tempfile
import torchaudio
import logging
from typing import Generator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置
API_BASE_URL = "http://localhost:8507"

def generate_seed():
    return np.random.randint(0, 1000000)

def change_instruction(mode_checkbox_group):
    if 'instruct' in mode_checkbox_group:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def generate_audio(
    tts_text: str,
    mode_checkbox_group: list,
    prompt_text: str,
    prompt_wav_upload: str,
    prompt_wav_record: str,
    instruct_text: str,
    seed: int,
    stream: bool,
    speed: float,
) -> str:
    """调用 API 生成音频"""
    try:
        # 确定使用哪个音频文件作为提示
        prompt_audio = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        if not prompt_audio:
            raise ValueError("Please provide a prompt audio file (upload or record)")

        # 准备请求数据
        files = {'prompt_audio': ('prompt.wav', open(prompt_audio, 'rb'), 'audio/wav')}
        data = {
            'text': tts_text,
            'stream': str(stream).lower(),
            'speed': str(speed)
        }

        # 根据模式选择 API 端点
        if 'instruct' in mode_checkbox_group:
            endpoint = f"{API_BASE_URL}/tts/instruct"
            data['instruct_text'] = instruct_text
        else:
            endpoint = f"{API_BASE_URL}/tts/zero_shot"
            data['prompt_text'] = prompt_text

        # 发送请求
        response = requests.post(endpoint, files=files, data=data)
        
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json().get('detail', '')
                error_msg += f": {error_detail}"
            except:
                pass
            raise ValueError(error_msg)

        # 保存返回的音频到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        return temp_file_path

    except Exception as e:
        logger.error(f"Error in generate_audio: {str(e)}")
        raise gr.Error(str(e))

def main():
    with gr.Blocks() as demo:
        gr.Markdown('## CosyVoice API Test')
        with gr.Row():
            with gr.Column():
                tts_text = gr.TextArea(label='Text', value='')
                mode_checkbox_group = gr.CheckboxGroup(['instruct'],
                                                     label='Mode',
                                                     value=[])
                instruction_text = gr.TextArea(label='Instruction',
                                            value='',
                                            visible=False)
                prompt_text = gr.TextArea(label='Prompt Text', value='')
                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        label='Prompt Speech Upload',
                        type='filepath')
                    prompt_wav_record = gr.Audio(
                        label='Prompt Speech Record',
                        type='filepath',
                        sources=["microphone"])
                with gr.Row():
                    seed = gr.Number(label='Seed',
                                   value=generate_seed(),
                                   precision=0)
                    stream = gr.Checkbox(label='Stream', value=False)
                    speed = gr.Slider(minimum=0.5,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label='Speed')
            with gr.Column():
                audio_output = gr.Audio(label='Output Speech',
                                      type='filepath')
                with gr.Row():
                    gen_btn = gr.Button('Generate', variant='primary')

        gen_btn.click(fn=generate_audio,
                     inputs=[
                         tts_text, mode_checkbox_group, prompt_text,
                         prompt_wav_upload, prompt_wav_record, instruction_text,
                         seed, stream, speed
                     ],
                     outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction,
                                 inputs=[mode_checkbox_group],
                                 outputs=[instruction_text])

    # 修改启动配置
    demo.queue()  # 使用默认配置
    demo.launch(
        server_name='0.0.0.0', 
        server_port=8508,
        share=False,  # 不创建公共链接
        show_error=True,  # 显示详细错误信息
        quiet=True  # 减少不必要的输出
    )

if __name__ == '__main__':
    main()
