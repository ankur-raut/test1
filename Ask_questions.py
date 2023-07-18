from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
import streamlit as st
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi"

with st.form('Question'):
    question=st.text_input("Ask a question from the Documents already ingested")

    button_clicked=st.form_submit_button("Submit")
if (question and button_clicked):
    persist_directory = 'db'
    embedding = CohereEmbeddings(cohere_api_key=cohere_api_key)

    vectordb2 = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding,
                   )

    retriever = vectordb2.as_retriever()

# create the chain to answer questions
    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use only the document for your answer and you may summarize the answer to make it look better. 
    

    {context}

    Question: {question}
    """
    

    qa_chain = RetrievalQA.from_chain_type(llm=Cohere(cohere_api_key=cohere_api_key),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs={
                                  "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
                                       )})

    llm_response = qa_chain(question)
    answer=llm_response["result"]
    st.write("Consise Answer")
    st.write(answer)
    # st.write("Detailed Answer from Text")
    # docs = vectordb2.similarity_search(question)
    # st.write(docs[0].page_content)
