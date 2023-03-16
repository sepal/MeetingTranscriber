import os
from dotenv import load_dotenv
import gradio as gr
import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from mimetypes import MimeTypes
import whisper
import tempfile

load_dotenv()

hg_token = os.getenv("HG_ACCESS_TOKEN") 
open_api_key = os.getenv("OPENAI_API_KEY")

if hg_token == None:
    print('''No hugging face access token set. 
You need to set it via an .env or environment variable HG_ACCESS_TOKEN''')
    exit(1)


def diarization(audio) -> np.array:
    """
    Receives a pydub AudioSegment and returns an numpy array with all segments.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hg_token)
    audio.export("/tmp/dz.wav", format="wav")
    diarization = pipeline("/tmp/dz.wav")
    return pd.DataFrame(list(diarization.itertracks(yield_label=True)),columns=["Segment","Trackname", "Speaker"])


def combine_segments(df):
    grouped_df = df.groupby((df['Speaker'] != df['Speaker'].shift()).cumsum())
    return grouped_df.agg({'Segment': lambda x: x.min() | x.max(), 
                            'Trackname': 'first',
                            'Speaker': 'first'})


def prep_audio(audio_segment):
    """
    This function preps a pydub AudioSegment for a ml model.

    Both pyannote audio and whisper require mono audio with a 16khz rate as float32.
    """
    audio_data = audio_segment.set_channels(1).set_frame_rate(16000)
    return np.array(audio_data.get_array_of_samples()).flatten().astype(np.float32) / 32768.0

def transcribe_row(row, audio):
    segment = audio[row.start_ms:row.end_ms]
    if open_api_key == None:
        whisper_ml = whisper.load_model("large")
        data = prep_audio(segment)
        return whisper_ml.transcribe(data)['text']
    else:
        print("Using openai API")
        # the open ai whisper AI only accepts audio files with a length of at 
        # least 0.1 seconds.
        if row['end_ms'] - row['start_ms'] < 100:
            return ""
        import openai
        import tempfile
        temp_file = f"/tmp/{row['Trackname']}.mp3"
        segment.export(temp_file, format="mp3")
        print(temp_file)
        audio_file = open(temp_file, "rb")
        return openai.Audio.translate("whisper-1", audio_file)['text']



def combine_transcription(segments):
    text = ""
    for _,row in segments.iterrows():
        text += f"[{row.Speaker}]: {row.text}\n"
    
    return text

def transcribe(audio_file: str) -> str:
    audio = AudioSegment.from_file(audio_file)
    print("diarization")
    df = diarization(audio)

    print("combining segments")
    df = combine_segments(df)

    df['start'] = df.Segment.apply(lambda x: x.start)
    df['end'] = df.Segment.apply(lambda x: x.end)

    df['start_ms'] = df.Segment.apply(lambda x: int(x.start*1000))
    df['end_ms'] = df.Segment.apply(lambda x: int(x.end*1000))

    print("transcribing segments")
    df['text'] = df.apply(lambda x: transcribe_row(x, audio), axis=1)

    return combine_transcription(df)


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
)

demo.launch()