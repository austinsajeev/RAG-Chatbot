import os
import pandas as pd
import pickle
import pdfplumber
import concurrent.futures
import threading
import time

# ✅ Updated imports to avoid deprecation warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Global cache for loaded documents
_document_cache = {}
_document_cache_lock = threading.Lock()

# Pre-initialize the LLM to avoid cold start latency
_llm = None
_llm_lock = threading.Lock()

def test_ollama():
    """Test if Ollama is working correctly"""
    try:
        llm = OllamaLLM(model="gemma:2b")
        test_result = llm.invoke("Test. Please respond with only one word: Working")
        print(f"✓ Ollama test response: {test_result}")
        return True
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        return False

def get_llm():
    """Lazily initialize and return LLM singleton"""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:  # Double-check lock pattern
                # Set timeout in initialization instead of as a property
                _llm = OllamaLLM(
                    model="gemma:2b", 
                    temperature=0.1,
                    # Remove setting timeout as a property later
                )
    return _llm

def load_pdf(file_path):
    """Optimized PDF loader using pdfplumber with caching."""
    if file_path in _document_cache:
        return _document_cache[file_path]
    
    text = ""
    with pdfplumber.open(file_path) as pdf:
        # Process pages in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            page_texts = list(executor.map(lambda p: p.extract_text() or "", pdf.pages))
            text = "\n".join(page_texts)
    
    with _document_cache_lock:
        _document_cache[file_path] = text
    return text

def load_txt(file_path):
    """Load text files with caching."""
    if file_path in _document_cache:
        return _document_cache[file_path]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read() + "\n"
    
    with _document_cache_lock:
        _document_cache[file_path] = text
    return text

def load_csv(file_path):
    """Load CSV files and convert to string with caching."""
    if file_path in _document_cache:
        return _document_cache[file_path]
    
    df = pd.read_csv(file_path)
    text = df.to_string() + "\n"
    
    with _document_cache_lock:
        _document_cache[file_path] = text
    return text

def load_documents(data_dir="data"):
    """Load all documents from the specified directory using parallel processing."""
    all_text = ""

    files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, f) for f in files if f.endswith(('.pdf', '.txt', '.csv'))]
    
    # Use ThreadPoolExecutor for parallel file loading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(load_file, all_files))
        all_text = "".join(results)

    return all_text

def load_file(file_path):
    """Determine file type and load accordingly with caching."""
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.txt'):
        return load_txt(file_path)
    elif file_path.endswith('.csv'):
        return load_csv(file_path)
    return ""

def create_or_load_vectorstore(text, path="vectorstore.pkl"):
    """Create or load a vectorstore for text embeddings with optimized parameters."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # Optimize chunk size for better retrieval performance
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        # Use a faster embedding model with caching
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./embeddings_cache"
        )
        
        # Build FAISS index with improved search speed
        db = FAISS.from_texts(chunks, embeddings)
        
        with open(path, "wb") as f:
            pickle.dump(db, f)
        return db

def get_qa_chain(vectorstore):
    """Retrieve optimized QA chain with timeout."""
    llm = get_llm()  # Use the singleton LLM instance
    
    # Define a prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant.\n\n"
            "Answer the question based on the following context:\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    # Initialize the retrieval-based QA chain with the prompt template
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Retrieve the top 2 results
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}  # Pass the prompt template here
    )

    return qa

