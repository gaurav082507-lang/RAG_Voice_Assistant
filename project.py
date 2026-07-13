from dotenv import load_dotenv
load_dotenv()

from whisper_model import model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydub import AudioSegment
from langchain_community.document_loaders import PyPDFLoader
import subprocess
import os

def build_retriever(file_path:str):
    data=PyPDFLoader(file_path=file_path)
    docs=data.load()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    embedding_model=MistralAIEmbeddings(model='mistral-embed-2312')
    chunk_docs=splitter.split_documents(docs)
    vectorstore=FAISS.from_documents(
        documents=chunk_docs,
        embedding=embedding_model
    )
    retriever=vectorstore.as_retriever(
        search_='mmr',
        search_kwargs={
            'k':5,
            'fetch_k':20,
            'lambda_mult':0.5
        }
    )
    return retriever

def audio_chunks(audio_path:str, output_dir: str = "chunks"):
    os.makedirs(output_dir, exist_ok=True)

    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(os.path.join(output_dir, "full_audio.wav"), format='wav')

    chunk_size = 10 * 1000
    chunks = [sound[i:i + chunk_size] for i in range(0, len(sound), chunk_size)]

    chunk_audio_path = []
    for index, chunk_audio in enumerate(chunks):
        path = os.path.join(output_dir, f"chunk_{index+1}.wav")
        chunk_audio.export(path, format='wav')
        chunk_audio_path.append(path)

    return chunk_audio_path

def transcribe_all(chunk_audio_path):
    transcription=""
    for chunk_audio in chunk_audio_path:
        result=model.transcribe(chunk_audio)
        main_text=result['text']
        transcription=transcription+" "+main_text
    
    return transcription

def text_to_speech(response:str):
    subprocess.run([
        "piper",
        "--model", "en_US-lessac-medium.onnx",
        "--output_file", "response.wav"
    ], input=response.encode(), check=True)








        



