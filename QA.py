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
        
    

     