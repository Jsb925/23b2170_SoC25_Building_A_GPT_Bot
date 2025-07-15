import dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from  google import generativeai as genai
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()
key=os.getenv('GEMINI_API_KEY')
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

ug_rule_text=get_pdf_text(['UG_RULE_BOOK.pdf'])
textSplitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks=textSplitter.split_text(ug_rule_text)

genai.configure(api_key=key)
os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if not os.path.exists("faiss_index"):
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("faiss_index")
else:
    vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    
    
prompt_template="You are a helpful assistant.Give out as much information as possible, and in detail.Answer using information from the given context only. If the required information is not present in the context, \
    say 'I do not know'.Use simple english. Do not use any asterisks. Do not include any characters apart from letters, numbers \
        and punctuation.  Context: {context} Question: {question}"
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
prompt=PromptTemplate.from_template(prompt_template)
model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)
chain= prompt | model

def respond(question):
    docs=vectorstore.similarity_search(question, k=3)
    context=""
    for doc in docs:
        context+=doc.page_content
    response=chain.invoke({"context":context,"question":question},return_only_outputs=True)
    return response.content

# print(respond("Explain academic categories in detail."))

################
import streamlit as st
import time
st.title("Interactive IITB UG Rulebook")
st.write("This is a simple RAG application using Langchain and Google Generative AI.")

# Add background image using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('logo.webp');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

question = st.text_input("Ask a question about the UG Rulebook:")
if question:
    with st.spinner("Thinking..."):
        answer = respond(question)
        st.write(answer)

