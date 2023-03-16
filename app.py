import os
from dotenv import load_dotenv
import gradio as gr
import numpy as np
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from mimetypes import MimeTypes

load_dotenv()

hg_token = os.getenv("HG_ACCESS_TOKEN") 

if hg_token != None:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hg_token)
else:
    print('''No hugging face access token set. 
You need to set it via an .env or environment variable HG_ACCESS_TOKEN''')
    exit(1)


def diarization(audio_file: tuple[int, np.array]) -> np.array:
    """
    Receives a tuple with the sample rate and audio data and returns the
    a numpy array containing the audio segments, track names and speakers for 
    each segment.
    """
    waveform = torch.tensor(audio_file[1].astype(np.float32, order='C')).reshape(1,-1)
    audio_data = {
        "waveform": waveform,
        "sample_rate": audio_file[0]
    }

    diarization = pipeline(audio_data)
    
    return np.array(list(diarization.itertracks(yield_label=True)))



demo = gr.Interface(
    fn=diarization,
    inputs=gr.Audio(type="numpy"),
    outputs="text",
)

demo.launch()