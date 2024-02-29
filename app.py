import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set page title and favicon
st.set_page_config(page_title="HR Service Chatbot", page_icon=":robot_face:")

# Function to customize Streamlit theme
def customize_theme():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f9f9f9;
        }
        .sidebar .sidebar-content {
            background: #4c566a;
            color: #d8dee9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

customize_theme()

# Title and introduction
st.title("HR Service Chatbot")
st.write(
    "Welcome to the HR Service Chatbot! This chatbot can assist you with HR-related questions."
)

# Load PDF documents and create QA system
loader = PyPDFDirectoryLoader("data")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000,
    chunk_overlap=20
)

text_chunks = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()

vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

llm = OpenAI(temperature=0.9, max_tokens=100)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

# User input section
st.write("\n")
user_input = st.text_input("You:", "")
if st.button("Send", key="send_button"):
    if user_input.strip() == "":
        st.error("Please enter a query.")
    else:
        # Show spinner while processing
        with st.spinner("Thinking..."):
            response = qa.invoke(user_input)
        st.write("\n")
        st.info("AI Assistant:")
        st.write(response["result"])
