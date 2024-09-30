import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
from prompt import get_conversational_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is missing. Please check your environment variables.")
else:
    genai.configure(api_key=api_key)

# Caching functions for better performance
@st.cache_data
def get_pdf_text(pdf_docs):
    """Extract text from PDF and text files."""
    text = ""
    for _, doc in enumerate(pdf_docs):
        if doc.name.endswith('.pdf'):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.name.endswith('.txt'):
            text += doc.read().decode("utf-8")
    return text

@st.cache_data
def get_text_chunks(text, chunk_size, chunk_overlap):
    """Chunk large text documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks, collection_name, file_name):
    """Create a vector store for embeddings and save it in the specified collection."""
    # Initialize the embeddings using Google's API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Generate metadata for each chunk with ID format 'id-001-001' (id-file-chunk)
    metadata = [{"id": f"id-{str(file_name).zfill(3)}-{str(i + 1).zfill(3)}"} for i in range(len(text_chunks))]

    # Initialize Chroma with a collection name and store metadata
    vector_store = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings, 
        collection_name=collection_name,  
        persist_directory="chroma_db_directory",  
        metadatas=metadata  
    )
    return vector_store

def user_input(user_question, collection_name):
    """Handle user queries using the vector store and LLM."""
    try:
        # Load the vector store from the designated collection
        vector_store = get_vector_store(st.session_state.text_chunks, collection_name=collection_name, file_name="user")
        retrieved_docs = vector_store.similarity_search(user_question)
        
        # Load the chain and get the conversational response
        chain = get_conversational_chain()
        response = chain({"input_documents": retrieved_docs, "question": user_question}, return_only_outputs=True)
        
        # Display the conversational response
        st.write("Reply:", response["output_text"])
        
        # Use expander to hide the documents until the user clicks to open
        with st.expander("Show Retrieved Documents"):
            for i, doc in enumerate(retrieved_docs):
                st.warning("Note: The retrieved documents below are shown purely for demo purposes. In a real application, these would not be visible to the user.")
                st.markdown(f"**The most relevant document from Chroma DB:{doc.id}**")
                st.write(doc.page_content)
                if i == 0:
                    break

    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    """Main function for Streamlit app."""
    
    # Sidebar menu for page navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ("ETL Pipeline", "Chatbot")
    )

    st.header("Gen AI ChatBot")

    # Page 1: Upload, Extract, Process Text, and Generate Embeddings - For the developers / SMEs
    if page == "ETL Pipeline":
        st.subheader("Upload PDF or Text Files")
        pdf_docs = st.file_uploader("Upload your PDF or Text Files", accept_multiple_files=True)
        
        if st.button("Extract Data"):
            with st.spinner("Extracting text from documents..."):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.raw_text = raw_text 
                st.session_state.pdf_docs = pdf_docs 
                st.success("Text extraction is done!")
        
        if 'raw_text' in st.session_state:
            st.subheader("Chunk Text for Processing")
            chunk_size = st.number_input("Chunk Size", min_value=500, max_value=10000, value=1000, step=500)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000, value=100, step=100)

            if st.button("Chunking & Indexing"):
                with st.spinner("Chunking text..."):
                    text_chunks = get_text_chunks(st.session_state.raw_text, chunk_size, chunk_overlap)
                    st.session_state.text_chunks = text_chunks  
                    st.success("Chunking is done!")

        if 'text_chunks' in st.session_state:
            st.subheader("Generate Embeddings and Create Vector Store")
            collection_name = st.text_input("Enter Collection Name for Vector Store", value="dataengine-collection")
            if st.button("Generate Embeddings & Store Vector Store"):
                with st.spinner("Generating embeddings and storing vector store..."):
                    for file_index, file in enumerate(st.session_state.pdf_docs):
                        get_vector_store(st.session_state.text_chunks, collection_name, file_index + 1)
                    st.success(f"Embeddings generated and vector store '{collection_name}' created! You can now chat with the bot.")
    
    # Page 2: Chatbot interaction - Designed for Users
    elif page == "Chatbot":
        if 'text_chunks' in st.session_state:
            collection_name = st.text_input("Ensure the Chatbot is connected to the Vector DB", value="dataengine-collection")
            user_question = st.text_input("Ask me a question!!")
            if user_question:
                user_input(user_question, collection_name)
        else:
            st.warning("Please upload and process text in the 'ETL Pipeline' page first.")

if __name__ == "__main__":
    main()
