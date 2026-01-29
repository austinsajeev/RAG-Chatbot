import warnings
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from backend import load_documents, create_or_load_vectorstore, get_qa_chain, test_ollama

# Suppress CropBox missing warnings
warnings.filterwarnings("ignore", message="CropBox missing")

# Initialize components in a background thread to avoid blocking startup
def initialize_components():
    global vectorstore, qa_chain
    # Test if Ollama is working properly
    if not test_ollama():
        print("ERROR: Ollama is not working correctly!")
    
    print("Loading documents...")
    docs = load_documents("data")
    print("Creating vector store...")
    vectorstore = create_or_load_vectorstore(docs)
    print("Setting up QA chain...")
    qa_chain = get_qa_chain(vectorstore)
    print("✅ Backend components initialized and ready")

# Set up global variables
vectorstore = None
qa_chain = None

# Start initialization in background
init_thread = threading.Thread(target=initialize_components)
init_thread.daemon = True
init_thread.start()

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "pong"}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "ready" if qa_chain else "initializing"})

@app.route('/chat', methods=['POST'])
def chat():
    # Wait for initialization if needed
    if not qa_chain:
        return jsonify({"answer": "System is still initializing. Please try again in a few seconds."}), 503
    
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided."}), 400
    
    try:
        # Set maximum execution time
        result = {"answer": None, "error": None}
        
        def run_query():
            try:
                result["answer"] = qa_chain.run(query)
            except Exception as e:
                result["error"] = str(e)
        
        # Start query in a thread
        query_thread = threading.Thread(target=run_query)
        query_thread.daemon = True
        query_thread.start()
        
        # Wait for completion with timeout
        query_thread.join(timeout=45)  # 45 second timeout
        
        if query_thread.is_alive():
            # If still running after timeout
            return jsonify({"answer": "Query is taking too long. Please try a simpler question or check back later."}), 200
        
        if result["error"]:
            return jsonify({"error": result["error"]}), 500
            
        return jsonify({"answer": result["answer"] or "No answer found"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use Flask's development server with threading
    app.run(host="0.0.0.0", port=5000, threaded=True)
