import os
import openai
import streamlit as st

from audio_recorder_streamlit import audio_recorder
from streamlit_chat import message
from elevenlabs import generate
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

load_environment = load_dotenv()

TEMP_AUDIO_PATH = 'temp_audio.wav'
AUDIO_FORMAT = "audio/wav"

openai.api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
deeplake_data_set_path = os.environ.get('DEEPLAKE_DATASET_PATH')

def load_database_and_embeddings():
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=deeplake_data_set_path,
        read_only=True,
        embedding=embeddings
    )
    return db

def transcribe_audio(audio_file_path, openai_key):
    openai.api_key = openai_key
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None
    
    