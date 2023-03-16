import os
from dotenv import load_dotenv
import gradio as gr
import numpy as np
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from mimetypes import MimeTypes
import whisper

load_dotenv()

hg_token = os.getenv("HG_ACCESS_TOKEN") 

if hg_token != None:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hg_token)
    whisper_ml = whisper.load_model("base")
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

def combine_segments(segments: np.array) -> np.array:
    new_arr = []
    prev_label = None
    for row in segments:
        if prev_label is None or row[2] != prev_label:
            new_arr.append(row)
            prev_label = row[2]
        else:
            new_arr[-1][0] = new_arr[-1][0] | row[0]
            new_arr[-1][1] = new_arr[-1][1]
            new_arr[-1][2] = prev_label
    return np.array(new_arr)

def split_audio(audio_file: tuple[int, np.array], segments):
    pass


def prep_audio(audio_segment):
    """
    This function preps a pydub AudioSegment for a ml model.

    Both pyannote audio and whisper require mono audio with a 16khz rate as float32.
    """
    audio_data = audio_segment.set_channels(1).set_frame_rate(16000)
    return np.array(audio_data.get_array_of_samples()).flatten().astype(np.float32) / 32768.0

def transcribe(audio_file: str) -> str:
    audio = AudioSegment.from_file(audio_file)
    
    audio_data = prep_audio(audio)
    return whisper_ml.transcribe(audio_data)['text']


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
)

demo.launch()