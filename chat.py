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
deeplake_dataset_path = os.environ.get('DEEPLAKE_DATASET_PATH')

def load_database_and_embeddings(deeplake_dataset_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=deeplake_dataset_path,
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
    
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.text", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio")

def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, openai.api_key)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

        return transcription
    
def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key='input')
    
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10

    model = ChatOpenAI(model='gpt-3.5-turbo')

    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)

    return qa({'query': user_input})

def display_conversation(history):
    for i in range(len(history['generated'])):
        message(history['past'][i], is_user=True, key=str(i) + "_user")
        message(history['generated'][i], key=str(i))

        voice = 'Bella'
        text = history['generated'][i]
        audio = generate(text=text, voice=voice, api_key=eleven_api_key)
        st.audio(audio, format='audio/mp3')

def main():
    st.write("# Voxtalum ")

    db = load_database_and_embeddings(deeplake_dataset_path)

    transcription = record_and_transcribe_audio()

    user_input = get_user_input(transcription)

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['Voxtalum is ready']

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hello World']

    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output['result'])
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()