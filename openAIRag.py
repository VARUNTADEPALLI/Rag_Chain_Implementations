import streamlit as st
import os
import fitz
from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from openai import OpenAI

st.set_page_config(layout="wide")



float_init()

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message['role'] == 'user':
            history.add_user_message(message['content'])
        else:
            history.add_ai_message(message['content'])
    return history

OPENAI_API_KEY = '<OPENAI_API_KEY>'
# llm = ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY)

pdf = st.file_uploader("upload a file")

if 'vectors' not in st.session_state:
    st.session_state.vectors = None

docs = []

if pdf is not None and st.session_state.vectors is None:
    file = pdf.read()
    with st.spinner("Processing document..."):
        doc = fitz.open(stream=file,filetype='pdf')

        for i in range(len(doc)):
            page =doc.load_page(i)
            page_content = page.get_text("text")
            document=Document(
                page_content=page_content,
                meta_data={
                    'source':pdf.name,
                    'page':i
                }

            )
            docs.append(document)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 150)
        docs = text_splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2')
        vectors = FAISS.from_documents(docs,embedding=embedding)
    
        st.session_state.vectors = vectors

chain = None
if st.session_state.vectors is not None:
    chain = RetrievalQA.from_chain_type(
            retriever=st.session_state.vectors.as_retriever(),
            llm=ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY),
    )

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False 

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Hi!How may i help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
container = st.container()

col1,col2 = st.columns([2,10])
with container:
    with col1:
        audio = audio_recorder()
    with col2:
        query = st.chat_input("hi there",disabled= st.session_state.is_recording)
container.float("bottom:0rem;")

if audio:
        with st.spinner("Transcribbing..."):
            st.session_state.is_recording = True 
        
            path = 'audio.mp3'
            with open(path,'wb') as f:
                f.write(audio)
            
            with open(path,'rb') as audio_file:
                question = OpenAI(api_key=OPENAI_API_KEY).audio.transcriptions.create(
                    model='whisper-1',
                    response_format='text',
                    file=audio_file
                )

        if question:
            st.session_state.messages.append({"role":"user","content":question})
            os.remove(path)
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    history = create_history(st.session_state.messages)
                    if chain is not None:
                        response = chain.invoke({'query':question,'messages': history.messages})
                    if response is not None:
                        st.markdown(response['result'])
                        st.session_state.messages.append({"role":"assistant","content":response['result']})
        st.session_state.is_recording = False 


    
if query is not None:
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("human"):
            st.markdown(query)
        with st.spinner("Generating response..."):  
            history = create_history(st.session_state.messages)
            if chain is not None:
                response = chain.invoke({'query':query,'messages': history.messages})
            if response is not None:
                with st.chat_message("assistant"):
                    st.markdown(response['result'])
                st.session_state.messages.append({"role":"assistant","content":response['result']})




