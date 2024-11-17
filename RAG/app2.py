import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from PIL import Image
import base64
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Access Hugging Face API Key from the environment
hf_api_key = os.getenv('HF_API_KEY')

# Check if the API key is available
if not hf_api_key:
    st.error("❌ Hugging Face API key is missing. Please configure it in the .env file.")
    st.stop()

# Set up custom CSS for the title, text styles, and centering image
st.markdown(
    """
    <style>
    /* Apply background color to the entire Streamlit app */
    .stApp {
        background-color:  #D0EFF4;  /* Set the background color for the whole app */
    }

    /* Apply background color to the sidebar */
    .stSidebar {
        background-color: #7fbbc5;  /* Set the sidebar background color */
    }

    /* Apply background color to the main content area */
    .main {
        background-color: #BDF2D8;  /* Set the background color for the main content area */
        padding: 20px;
    }

    body {
        font-family: Arial, sans-serif;
    }

    .title {
        color: #000000;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    .description {
        font-size: 18px;
        color: #000000;
        text-align: center;
        margin-bottom: 20px;
    }

    .error {
        color: red;
        font-size: 16px;
        font-weight: bold;
    }

    .success {
        color: green;
        font-size: 16px;
    }

    .upload-box {
        border: 2px dashed #D0EFF4;
        padding: 20px;
        margin-top: 20px;
        background-color: #D0EFF4;
        border-radius: 10px;
    }

    .query-input {
        margin-top: 20px;
    }

    .response {
        color: #333;
        font-size: 16px;
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
    }

 /* Custom styles for the file uploader box */
    .stFileUploader {
        background-color: #D0EFF4 !important;  /* Set the background color of the file uploader box */
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #4CAF50;  /* Optional: Add a dashed border for styling */
    }

    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%; /* Ensure image doesn't overflow */
    }
    </style>
    """, unsafe_allow_html=True)

# Load the image for the main content
image_path = "static/image.jpg"  # Update this path to match your static folder location
bg_image = Image.open(image_path)

# Convert the image to Base64
buffered = BytesIO()
bg_image.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Main content: Title and Image (Centered)
st.markdown('<p class="title">RAG by Using LangChain & HuggingFace</p>', unsafe_allow_html=True)

# Use HTML to center the image
st.markdown(
    f'<div style="display: block; margin-left: auto; margin-right: auto; text-align: center;">'
    f'<img src="data:image/jpeg;base64,{img_base64}" class="centered-image" alt="Image" style="max-width: 100%"/>'
    f'</div>', unsafe_allow_html=True)

# Sidebar: PDF Upload and Document Processing
st.sidebar.markdown('<p class="description">Upload a PDF file and ask a question about the document to get insights.</p>', unsafe_allow_html=True)

# File uploader to allow a single PDF upload in the sidebar
uploaded_file = st.sidebar.file_uploader("", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.read())

    # Load the PDF file
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()

    # Check if the document is loaded properly
    if len(docs) == 0:
        st.sidebar.markdown('<p class="error">No content was extracted from the PDF: {}</p>'.format(uploaded_file.name), unsafe_allow_html=True)
        st.stop()
    else:
        st.sidebar.markdown('<p class="success">Loaded {} pages from the uploaded PDF: {}</p>'.format(len(docs), uploaded_file.name), unsafe_allow_html=True)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Check if chunk splitting worked properly
    if len(splits) == 0:
        st.sidebar.markdown('<p class="error">No chunks were created from the documents. Please check the document content and chunk size.</p>', unsafe_allow_html=True)
        st.stop()

    # Generate embeddings for the chunks
    embeddings_model = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings_model)

    # Display a text input for user queries in the main content area
    query = st.text_input("Ask a question about the document:", key="query", placeholder="Type your question here...", label_visibility="collapsed")

    if query:
        # Perform a similarity search to find relevant documents
        results = vectorstore.similarity_search(query, top_k=4)

        if not results:
            st.warning("❓ No relevant documents found for the query.")
        else:
            # Set up Hugging Face LLM (language model)
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                huggingfacehub_api_token=hf_api_key,
                max_new_tokens=512
            )
            chat = ChatHuggingFace(llm=llm, verbose=True)

            # Define the RAG chain to retrieve and generate answers
            retriever = vectorstore.as_retriever()
            prompt = hub.pull("rlm/rag-prompt")

            rag_chain = (
                {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                 "question": RunnablePassthrough()}
                | prompt
                | chat
                | StrOutputParser()
            )

            # Get the answer from the chain
            try:
                response = rag_chain.invoke(query)
                st.markdown(f'<div class="response"><strong>Answer:</strong><br>{response}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error during RAG chain invocation: {e}")
