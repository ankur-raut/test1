import streamlit as st 
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os
from langchain.llms import Cohere
import speech_recognition as sr
from langchain.document_loaders import UnstructuredURLLoader
import PyPDF2
import shutil
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
def audio_to_text(audio_file):
    # Create a recognizer object
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file) as source:
        # Read the audio data from the file
        audio_data = recognizer.record(source)

        # Perform speech recognition
        text = recognizer.recognize_google(audio_data)

    # Return the recognized text
    return text
# #Function to convert PDF Files to TXT Files
def pdf_to_txt(pdf_path, output_folder):
    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Create the output file path
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
        output_path = os.path.join(output_folder, output_filename)

        # Extract text from each page and write to the output file
        with open(output_path, "w", encoding="utf-8") as txt_file:
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                txt_file.write(text)
def text_loader(file_contents,uploaded_file):
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    save_path = os.path.join(folder_path, uploaded_file.name)
    if file_extension == ".pdf":
        with open(save_path, "wb") as f:
            f.write(file_contents)
        loader = PyPDFLoader(save_path) #Step 1.1
    if file_extension == ".txt":
        file_contents=file_contents.decode('utf-8')
        print(save_path)
        with open(save_path, "w") as f:
            f.write(file_contents)
        loader = TextLoader(save_path) #Step 1.1
    elif file_extension == ".wav":
        with open(save_path, "wb") as f:
            f.write(file_contents)
        result = audio_to_text(save_path)
    # Specify the path for the output text file
        output_file_path = f"{os.path.splitext(uploaded_file.name)[0]}.txt"
    # Write the result to a text file
        with open(output_file_path, 'w') as file:
            file.write(result)
            loader = TextLoader(output_file_path)

    return loader

def query(loader, question):
    documents = loader.load()

    #1.2
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0) #Splitting the text and creating chunks
    docs = text_splitter.split_documents(documents)
    
    #1.3
    embeddings = CohereEmbeddings(cohere_api_key=api_key) #Creating Cohere Embeddings
    db = Chroma.from_documents(docs, embeddings) #Storing the embeddings in the vector database
    #2.2
    docs = db.similarity_search(question) #Searching for the query in the Vector Database and using cosine similarity for the same.
    return docs[0].page_content

def query_consise(loader, question):
    documents = loader.load()
    persist_directory = 'db'

    #1.2
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0) #Splitting the text and creating chunks
    docs = text_splitter.split_documents(documents)
    
    #1.3
    embeddings = CohereEmbeddings(cohere_api_key=api_key) #Creating Cohere Embeddings
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory) #Storing the embeddings in the vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use only the document for your answer and you may summarize the answer to make it look better. 
    

    {context}

    Question: {question}
    """
    

    qa_chain = RetrievalQA.from_chain_type(llm=Cohere(cohere_api_key=api_key),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs={
                                  "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
                                       )}) #Read documentation- Link.
    llm_response = qa_chain(question)
    return llm_response["result"]


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except WindowsError:
        pass

st.title(" Question Answering Over Documents") 
st.sidebar.markdown(
    """
    ### Steps:
    1. Chose LLM
    2. Enter Your Secret Key for Embeddings
    3. Perform Q&A
    """
)
folder_path="Files"
sub_folder_path_text="Files/SubFolder/Textfiles"
sub_folder_path_original="Files/SubFolder/OrignalFiles" #This will be under Files folder.
 
delete_folder(folder_path)
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
if not os.path.exists(sub_folder_path_original):
    os.makedirs(sub_folder_path_original)
if not os.path.exists(sub_folder_path_text):
    os.makedirs(sub_folder_path_text)
with st.form('Cohere/OpenAI'): 

        model = st.radio('Choose OpenAI/Cohere',('OpenAI','Cohere')) 
        api_key = st.text_input('Enter API key',             
                                type="password",
                                help="https://platform.openai.com/account/api-keys",)  

        submitted = st.form_submit_button("Submit")
if api_key:
        llm = Cohere(cohere_api_key=api_key)
        embeddings = CohereEmbeddings(cohere_api_key=api_key)

else:
     st.write("Please enter valid API key")
genre = st.radio( 
        "Select ", 
        ('Single Files', 'Multiple Files','Url')) 
st.write('You selected:', genre)
if genre == "Single Files":
    uploaded_file = st.file_uploader('Upload a file (Only txt, pdf, wav formats allowed)', type=['txt', 'pdf','wav']) 
    if uploaded_file is not None: 
        st.write("File uploaded successfully!")
        file_contents = uploaded_file.read()
        
        loader=text_loader(file_contents,uploaded_file)    

    else:
         st.write("Upload File")
if genre == "Multiple Files":
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt","wav"],accept_multiple_files=True)
    for i in uploaded_files:
        file_extension = os.path.splitext(i.name)[1].lower()
        file_path=sub_folder_path_original+"/"+i.name
        if file_extension== ".pdf":
            file_contents=i.read()
            with open(file_path, "wb") as f:
                f.write(file_contents)
            pdf_to_txt(file_path,sub_folder_path_text)

        if file_extension==".wav":
            file_contents=i.read()
            with open(file_path, "wb") as f:
                f.write(file_contents)
            result = audio_to_text(file_path)
        # Specify the path for the output text file
            output_file_path = sub_folder_path_text+"/"+f"{os.path.splitext(i.name)[0]}.txt"
        # Write the result to a text file
            with open(output_file_path, 'w') as file:
                file.write(result)
        if file_extension == ".txt":
            output_file_path=sub_folder_path_text+"/"+i.name
            file_contents=i.read().decode('utf-8')
            with open(output_file_path, "w") as f:
                f.write(file_contents)
    loader = DirectoryLoader(sub_folder_path_text, glob="./*.txt", loader_cls=TextLoader)
    
if genre == "Url":
    url = st.text_input("Enter a URL:")
    loader=UnstructuredURLLoader(urls=[url])
    st.write(loader)
with st.form('Question'):
    question=st.text_input("Ask a question from the document")

    button_clicked=st.form_submit_button("Submit")
if (question and button_clicked):
    try: 
        answer_consise=query_consise(loader, question)
        st.write("Consise Answer:") #FFor this we use the LLM model
        st.write(answer_consise)
        # ans=query(loader,question) #Single
        # st.write("Answer from the documents:") #We use cosine similarity to query in the database
        # st.write(ans)

    except:
        st.write("Enter Valid API key")
        
    
# absl-py==1.0.0
# aiohttp==3.8.4
# aiosignal==1.3.1
# altair==4.2.0
# altgraph==0.17.3
# annotated-types==0.5.0
# anyio==3.6.1
# argon2-cffi==21.3.0
# argon2-cffi-bindings==21.2.0
# asgiref==3.6.0
# astor==0.8.1
# astroid==2.5.6
# astunparse==1.6.3
# async-generator==1.10
# async-timeout==4.0.2
# attrs==21.2.0
# auto-py-to-exe==2.23.1
# autopep8==1.6.0
# Babel==2.10.1
# backcall==0.2.0
# backoff==2.2.1
# backports.entry-points-selectable==1.1.1
# base58==2.1.1
# beautifulsoup4==4.9.1
# bleach==4.1.0
# blinker==1.4
# blis==0.7.8
# bottle==0.12.23
# bottle-websocket==0.2.9
# cachetools==5.0.0
# caer==2.0.8
# catalogue==1.0.0
# certifi==2022.9.14
# cffi==1.15.0
# chardet==4.0.0
# charset-normalizer==2.1.1
# chromadb==0.3.27
# cinemagoer==2022.2.11
# click==7.1.2
# clickhouse-connect==0.6.6
# cog==0.7.2
# cohere==4.11.2
# colorama==0.4.4
# coloredlogs==15.0.1
# comtypes==1.1.10
# cryptography==38.0.1
# cycler==0.11.0
# cymem==2.0.6
# dataclasses-json==0.5.9
# debugpy==1.5.1
# decorator==5.1.0
# defusedxml==0.7.1
# distlib==0.3.3
# Django==4.1.7
# Django-Gtts==0.4
# docx2txt==0.8
# duckdb==0.8.1
# Eel==0.14.0
# en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
# entrypoints==0.3
# fastapi==0.85.1
# fastjsonschema==2.15.3
# filelock==3.4.0
# Flask==1.1.1
# flatbuffers==2.0
# fonttools==4.33.3
# fpdf==1.7.2
# frozenlist==1.3.3
# future==0.18.2
# gast==0.5.3
# gevent==21.12.0
# gevent-websocket==0.10.1
# gitdb==4.0.9
# GitPython==3.1.24
# google-auth==2.6.6
# google-auth-oauthlib==0.4.6
# google-pasta==0.2.0
# google-search-results==2.4.2
# GPUtil==1.4.0
# greenlet==1.1.3
# grpcio==1.44.0
# gTTS==2.2.4
# gunicorn==19.9.0
# h11==0.13.0
# h5py==3.6.0
# hnswlib==0.7.0
# html5lib==1.1
# httptools==0.5.0
# humanfriendly==10.0
# idna==2.10
# imageio==2.19.3
# IMDbPY==2022.7.9
# imgaug==0.4.0
# importlib-metadata==6.7.0
# ipykernel==6.6.0
# ipython==7.30.1
# ipython-genutils==0.2.0
# ipywidgets==7.6.5
# isort==5.8.0
# itsdangerous==2.0.1
# jedi==0.18.1
# Jinja2==3.1.2
# joblib==1.0.1
# json5==0.9.8
# jsonschema==3.2.0
# jupyter-client==7.1.0
# jupyter-core==4.9.1
# jupyter-server==1.17.0
# jupyterlab==3.4.2
# jupyterlab-pygments==0.1.2
# jupyterlab-server==2.14.0
# jupyterlab-widgets==1.0.2
# kiwisolver==1.4.2
# langchain==0.0.230
# langchainplus-sdk==0.0.20
# lazy-object-proxy==1.6.0
# libclang==14.0.1
# lxml==4.9.3
# lz4==4.3.2
# Markdown==3.3.6
# MarkupSafe==2.1.1
# marshmallow==3.19.0
# marshmallow-enum==1.5.1
# matplotlib==3.5.2
# matplotlib-inline==0.1.3
# mccabe==0.6.1
# mistune==0.8.4
# monotonic==1.6
# mpmath==1.2.1
# multidict==6.0.4
# multitasking==0.0.11
# murmurhash==1.0.8
# mypy==0.971
# mypy-extensions==0.4.3
# mysql-connector-python==8.0.25
# nbclassic==0.3.7
# nbclient==0.5.9
# nbconvert==6.5.0
# nbformat==5.4.0
# nest-asyncio==1.5.4
# networkx==2.8.2
# notebook==6.4.6
# notebook-shim==0.1.0
# numexpr==2.8.4
# numpy
# oauthlib==3.2.0
# onnxruntime==1.15.1
# openai==0.27.8
# openapi-schema-pydantic==1.2.4
# opt-einsum==3.3.0
# outcome==1.2.0
# overrides==7.3.1
# packaging==21.3
# pandas==1.3.3
# pandocfilters==1.5.0
# parso==0.8.3
# pascal-voc-writer==0.1.4
# pdfminer.six==20220524
# pefile==2022.5.30
# pickleshare==0.7.5
# Pillow==10.0.0
# plac==1.1.3
# platformdirs==2.4.0
# playsound==1.3.0
# plotly==5.15.0
# posthog==3.0.1
# preshed==3.0.7
# prometheus-client==0.12.0
# prompt-toolkit==3.0.24
# protobuf==3.19.1
# psutil==5.9.2
# pulsar-client==3.2.0
# pyadl==0.1
# pyarrow==6.0.1
# pyasn1==0.4.8
# pyasn1-modules==0.2.8
# PyAudio==0.2.12
# pycodestyle==2.8.0
# pycparser==2.21
# pycryptodome==3.15.0
# pydantic
# pydantic_core
# pydeck==0.7.1
# Pygments==2.11.1
# pyinstaller==5.4.1
# pyinstaller-hooks-contrib==2022.10
# pylint==2.8.3
# Pympler==1.0.1
# pyparsing==3.0.6
# pypdf==3.12.1
# PyPDF2==3.0.1
# PyQt5==5.15.7
# PyQt5-Qt5==5.15.2
# PyQt5-sip==12.11.0
# pyqtgraph==0.12.4
# pyreadline3==3.4.1
# pyresparser==1.0.6
# pyrsistent==0.18.0
# PySocks==1.7.1
# python-dateutil==2.8.2
# python-dotenv==1.0.0
# pyttsx3==2.90
# pytz==2021.1
# pytz-deprecation-shim==0.1.0.post0
# PyYAML==6.0
# pyzmq==22.3.0
# recorder==0.0.2
# regex==2023.6.3
# replicate==0.8.3
# reportlab==4.0.4
# requests==2.28.1
# requests-futures==1.0.1
# requests-oauthlib==1.3.1
# rsa==4.8
# scipy==1.7.1
# selenium==4.4.3
# Send2Trash==1.8.0
# Shapely==1.8.4
# six==1.16.0
# smmap==5.0.0
# sniffio==1.2.0
# sortedcontainers==2.4.0
# soupsieve==2.2.1
# spacy==2.3.5
# SpeechRecognition==3.8.1
# spotipy==2.19.0
# SQLAlchemy==1.4.41
# sqlparse==0.4.3
# srsly==1.0.5
# starlette==0.20.4
# streamlit==1.3.1
# streamlit-drawable-canvas==0.9.3
# streamlit-image-annotation==0.3.1
# streamlit-lottie==0.0.3
# structlog==22.3.0
# sympy==1.11.1
# sysmon==1.0.1
# tabulate==0.9.0
# temp==2020.7.2
# tenacity==8.2.2
# termcolor==1.1.0
# terminado==0.12.1
# testpath==0.5.0
# textblob==0.17.1
# tf-estimator-nightly==2.8.0.dev2021122109
# thinc==7.4.5
# threadpoolctl==2.2.0
# tifffile==2022.5.4
# tiktoken==0.4.0
# tinycss2==1.1.1
# tk==0.1.0
# tmdbv3api==1.6.1
# toml==0.10.2
# tomli==2.0.1
# toolz==0.11.2
# tornado==6.1
# tqdm==4.65.0
# traitlets==5.1.1
# trio==0.21.0
# trio-websocket==0.9.2
# typing-inspect==0.9.0
# typing_extensions==4.7.1
# tzdata==2021.5
# tzlocal==4.1
# urllib3==1.26.5
# uvicorn==0.22.0
# validators==0.18.2
# vidaug==1.5
# virtualenv==20.10.0
# virtualenvwrapper-win==1.2.6
# wasabi==0.10.1
# watchdog==2.1.6
# watchfiles==0.19.0
# wcwidth==0.2.5
# webencodings==0.5.1
# websocket-client==1.3.2
# websockets==11.0.3
# Werkzeug==0.15.5
# whichcraft==0.6.1
# whitenoise==5.3.0
# widgetsnbextension==3.5.2
# wikipedia==1.4.0
# WMI==1.5.1
# wrapt==1.12.1
# wsproto==1.2.0
# xmltodict==0.13.0
# yahooquery==2.3.1
# yarl==1.9.2
# yfinance==0.1.74
# zipp==3.8.0
# zope.event==4.5.0
# zope.interface==5.4.0
# zstandard==0.21.0
