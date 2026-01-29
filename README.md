# Custom Chatbot with RAG

This is a local chatbot application that uses Retrieval-Augmented Generation (RAG) to answer questions based on your own documents. It uses **Ollama** for the LLM and **LangChain** for document processing.

## Features
- **Local Privacy**: Runs entirely on your machine.
- **RAG Architecture**: Retrieves context from your specific documents.
- **Multi-Format Support**: ingest PDF, TXT, and CSV files.
- **Fast Inference**: Uses the `gemma:2b` model by default.

## Prerequisites
1.  **Python 3.8+** installed.
2.  **Ollama** installed and running.
    - Download from [ollama.com](https://ollama.com).
    - Run `ollama pull gemma:2b` to get the required model.

## Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/austinsajeev/Custom-Chatbot.git
    cd Custom-Chatbot
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Add your Data
Place your documents (PDF, TXT, or CSV) into the **`data/`** folder.  
*Example: Save your `handbook.pdf` into `data/`.*

### 2. Run the Application
Start the Flask server:
```bash
python app.py
```
*Note: The first run might take a moment to index your documents.*

### 3. Chat
Open your browser and navigate to:  
`http://127.0.0.1:5000`

You can now ask questions specifically about the documents you uploaded!
