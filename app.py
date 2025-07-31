import os
import pandas as pd
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import io
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify, render_template_string
import json

# Flask app setup
app = Flask(__name__)

# Constants
EMBEDDING_MODEL = "embedding-001"
GENERATIVE_MODEL = "gemini-2.5-pro"
VECTOR_DIM = 768
DATA_DIR = "data"
API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key")  # Replace with your key or use env variable

# State management
state = {
    "index": None,
    "chat_history": []
}

# HTML template with Tailwind CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Estate RAG Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
            font-family: 'Orbitron', 'Arial', sans-serif;
            color: #e0e0e0;
        }
        .glow {
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00ddeb, 0 0 20px #00ddeb; }
            to { text-shadow: 0 0 20px #ff00ff, 0 0 30px #ff00ff; }
        }
        .chat-message {
            transition: all 0.3s ease;
        }
        .chat-message:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 25px rgba(0, 221, 235, 0.5);
        }
        .property-card {
            transition: all 0.3s ease;
        }
        .property-card:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 221, 235, 0.4);
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1b263b; }
        ::-webkit-scrollbar-thumb { background: #00ddeb; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #ff00ff; }
    </style>
</head>
<body class="min-h-screen flex">
    <!-- Sidebar -->
    <div class="w-1/4 bg-gray-900/90 border-r border-cyan-400 shadow-lg p-4 space-y-4">
        <h2 class="text-xl text-cyan-400 font-bold">📁 Data Sources</h2>
        <div class="space-y-2">
            <label class="block text-sm text-gray-300">Upload Property Data (Excel/CSV)</label>
            <input type="file" id="csv_file" accept=".csv,.xls,.xlsx" class="w-full bg-white/10 border border-cyan-400 rounded-lg p-2 text-gray-300">
        </div>
        <div class="space-y-2">
            <label class="block text-sm text-gray-300">Upload Guidelines (PDF)</label>
            <input type="file" id="pdf_file" accept=".pdf" class="w-full bg-white/10 border border-cyan-400 rounded-lg p-2 text-gray-300">
        </div>
        <button id="load_button" class="w-full bg-gradient-to-r from-cyan-400 to-pink-500 text-white py-2 rounded-lg hover:from-pink-500 hover:to-cyan-400 hover:shadow-lg transform hover:-translate-y-1 transition-all">🚀 Load and Process Data</button>
        <div id="load_output" class="text-sm text-gray-300"></div>
        <div id="progress_output" class="text-sm text-gray-300"></div>
        <h2 class="text-xl text-cyan-400 font-bold mt-4">💬 Chat History</h2>
        <div id="chat_output" class="space-y-2 max-h-96 overflow-y-auto"></div>
    </div>

    <!-- Main Content -->
    <div class="w-3/4 p-6 space-y-6">
        <h1 class="text-4xl text-center text-cyan-400 font-bold glow">Real Estate RAG Assistant</h1>
        <div id="client_status" class="text-red-400"></div>

        <!-- Chat Interface -->
        <div class="space-y-4">
            <h2 class="text-2xl text-cyan-400 font-bold">💬 Chat with Your Assistant</h2>
            <div id="latest_query" class="text-gray-300 bg-white/5 border border-cyan-400 rounded-lg p-4 mb-2"></div>
            <div id="query_output" class="text-gray-300 bg-white/5 border border-cyan-400 rounded-lg p-4"></div>
            <div class="flex space-x-2">
                <input id="user_query" type="text" placeholder="e.g., Show me 3BHK properties under ₹1 crore in Chennai" class="flex-grow bg-white/10 border border-cyan-400 rounded-lg p-2 text-gray-300 focus:ring-2 focus:ring-pink-500">
                <button id="submit_button" class="bg-gradient-to-r from-cyan-400 to-pink-500 text-white px-4 py-2 rounded-lg hover:from-pink-500 hover:to-cyan-400 hover:shadow-lg transform hover:-translate-y-1 transition-all">Ask</button>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="space-y-4">
            <h2 class="text-2xl text-cyan-400 font-bold">🚀 Quick Actions</h2>
            <div class="flex space-x-4">
                <button id="budget_button" class="bg-gradient-to-r from-cyan-400 to-pink-500 text-white px-4 py-2 rounded-lg hover:from-pink-500 hover:to-cyan-400 hover:shadow-lg transform hover:-translate-y-1 transition-all">💰 Show Budget Properties</button>
                <button id="locations_button" class="bg-gradient-to-r from-cyan-400 to-pink-500 text-white px-4 py-2 rounded-lg hover:from-pink-500 hover:to-cyan-400 hover:shadow-lg transform hover:-translate-y-1 transition-all">🏙️ Popular Locations</button>
                <button id="insights_button" class="bg-gradient-to-r from-cyan-400 to-pink-500 text-white px-4 py-2 rounded-lg hover:from-pink-500 hover:to-cyan-400 hover:shadow-lg transform hover:-translate-y-1 transition-all">📈 Market Insights</button>
            </div>
            <button id="clear_button" class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-500 hover:shadow-lg transform hover:-translate-y-1 transition-all">🗑️ Clear Chat History</button>
        </div>
    </div>

    <script>
        async function sendFiles() {
            const csvFile = document.getElementById('csv_file').files[0];
            const pdfFile = document.getElementById('pdf_file').files[0];
            const loadOutput = document.getElementById('load_output');
            const progressOutput = document.getElementById('progress_output');
            const formData = new FormData();
            if (csvFile) formData.append('csv_file', csvFile);
            if (pdfFile) formData.append('pdf_file', pdfFile);

            loadOutput.textContent = 'Processing...';
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            loadOutput.textContent = result.message;
            progressOutput.innerHTML = result.progress.join('<br>');
        }

        async function sendQuery(query) {
            const latestQuery = document.getElementById('latest_query');
            const queryOutput = document.getElementById('query_output');
            const chatOutput = document.getElementById('chat_output');
            latestQuery.textContent = `You: ${query}`;
            queryOutput.textContent = 'Processing...';
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const result = await response.json();
            queryOutput.textContent = result.response;
            chatOutput.innerHTML = result.chat_history.map(item => `
                <div class="chat-message bg-white/5 border border-cyan-400 rounded-lg p-2">
                    <p><b>You:</b> ${item[0]}</p>
                    <p><b>Assistant:</b> ${item[1]}</p>
                </div>
            `).join('');
            chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to bottom for new messages
        }

        async function sendQuickAction(action) {
            const latestQuery = document.getElementById('latest_query');
            const queryOutput = document.getElementById('query_output');
            const chatOutput = document.getElementById('chat_output');
            let query = '';
            if (action === 'budget') query = "Show me affordable properties under ₹1 crore in Chennai";
            else if (action === 'locations') query = "What are the most popular locations for real estate investment in Chennai?";
            else if (action === 'insights') query = "Give me insights about current real estate market trends in Chennai";
            latestQuery.textContent = `You: ${query}`;
            queryOutput.textContent = 'Processing...';
            const response = await fetch('/quick-action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });
            const result = await response.json();
            queryOutput.textContent = result.response;
            chatOutput.innerHTML = result.chat_history.map(item => `
                <div class="chat-message bg-white/5 border border-cyan-400 rounded-lg p-2">
                    <p><b>You:</b> ${item[0]}</p>
                    <p><b>Assistant:</b> ${item[1]}</p>
                </div>
            `).join('');
            chatOutput.scrollTop = chatOutput.scrollHeight; // Scroll to bottom for new messages
        }

        async function clearHistory() {
            const queryOutput = document.getElementById('query_output');
            const chatOutput = document.getElementById('chat_output');
            const response = await fetch('/clear-history', {
                method: 'POST'
            });
            const result = await response.json();
            queryOutput.textContent = result.message;
            chatOutput.innerHTML = '';
        }

        document.getElementById('load_button').addEventListener('click', sendFiles);
        document.getElementById('submit_button').addEventListener('click', () => {
            const query = document.getElementById('user_query').value;
            if (query) sendQuery(query);
        });
        document.getElementById('budget_button').addEventListener('click', () => sendQuickAction('budget'));
        document.getElementById('locations_button').addEventListener('click', () => sendQuickAction('locations'));
        document.getElementById('insights_button').addEventListener('click', () => sendQuickAction('insights'));
        document.getElementById('clear_button').addEventListener('click', clearHistory);

        // Check client status
        fetch('/client-status').then(response => response.json()).then(data => {
            if (data.status !== 'success') {
                document.getElementById('client_status').textContent = data.message;
            }
        });
    </script>
</body>
</html>
"""

def get_gemini_client():
    try:
        return True, "success"
    except Exception as e:
        return False, f"Failed to initialize Gemini client: {str(e)}"

async def get_embedding_async(texts: List[str]) -> List[np.ndarray]:
    try:
        if not texts or not all(isinstance(t, str) for t in texts):
            return []
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
        embeddings_list = await embeddings.aembed_documents(texts)
        return [np.array(e, dtype=np.float32) for e in embeddings_list]
    except Exception as e:
        return []

def get_embedding(text: str) -> np.ndarray:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embedding = loop.run_until_complete(get_embedding_async([text]))[0] if loop.run_until_complete(get_embedding_async([text])) else None
        loop.close()
        return embedding
    except Exception as e:
        return None

def load_csv_data(csv_file) -> List[Tuple[str, str, Dict[str, Any]]]:
    try:
        if isinstance(csv_file, bytes):
            file_bytes = csv_file
            if csv_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            if csv_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(csv_file)
            else:
                df = pd.read_csv(csv_file)
        
        documents = []
        for idx, row in df.iterrows():
            content = f"""
Property ID: {row.get('Property ID', 'N/A')}
Project Name: {row.get('Project Name', 'N/A')}
Location: {row.get('Location', 'N/A')}
Address: {row.get('Address', 'N/A')}
Status: {row.get('Status', 'N/A')}
Type: {row.get('Type', 'N/A')}
BHK: {row.get('BHK', 'N/A')}
Size: {row.get('Size (sq.ft.)', 'N/A')} sq.ft.
Price: ₹{row.get('Start Price', 'N/A')}
Price per sq.ft.: ₹{row.get('Price/sq.ft', 'N/A')}
Amenities: {row.get('Amenities', 'N/A')}
Nearby: {row.get('Nearby', 'N/A')}
Furnishing: {row.get('Furnishing', 'N/A')}
Contact Person: {row.get('Contact Person', 'N/A')}
Contact Number: {row.get('Contact', 'N/A')}
Offers: {row.get('Offers', 'N/A')}
"""
            metadata = {
                "source": "property_data",
                "property_id": row.get('Property ID', 'N/A'),
                "location": row.get('Location', 'N/A'),
                "price": row.get('Start Price', 'N/A'),
                "bhk": row.get('BHK', 'N/A')
            }
            documents.append((f"property-{idx}", content.strip(), metadata))
        return documents
    except Exception as e:
        return []

def load_pdf_data(pdf_file) -> List[Tuple[str, str, Dict[str, Any]]]:
    try:
        if isinstance(pdf_file, bytes):
            doc = fitz.open(stream=pdf_file, filetype="pdf")
        else:
            doc = fitz.open(pdf_file)
        
        full_text = ""
        for page in doc:
            text = page.get_text()
            if text:
                full_text += text + "\n"
        doc.close()
        
        chunks = []
        chunk_size = 500
        overlap = chunk_size // 4
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append((f"guide-{i}", chunk, {"source": "guidelines"}))
        return documents
    except Exception as e:
        return []

class SmartFaissIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.metadata = []
        self.documents = []

    def add_documents(self, documents: List[Tuple[str, str, Dict[str, Any]]]) -> List[str]:
        if not documents:
            return ["No documents provided to index."]
        
        contents = [doc[1] for doc in documents]
        embeddings = asyncio.run(get_embedding_async(contents))
        if not embeddings or len(embeddings) != len(documents):
            return ["Failed to generate embeddings for all documents."]
        
        progress_messages = []
        for i, (doc_id, _, meta) in enumerate(documents):
            progress_messages.append(f"Processing document {i+1}/{len(documents)}...")
            self.documents.append((doc_id, contents[i], meta))
        
        self.index.add(np.array(embeddings))
        self.metadata.extend(self.documents)
        progress_messages.append("✅ All documents processed successfully!")
        time.sleep(1)
        return progress_messages

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if self.index.ntotal == 0:
            return []
        
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                _, content, _ = self.documents[idx]
                results.append(content)
        return results

def generate_intelligent_response(query: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "I couldn't find relevant information to answer your question. Please try rephrasing or upload new data."
    
    context = "\n\n---\n\n".join(context_chunks)
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=""" 
You are an expert real estate assistant with deep knowledge of property markets in Chennai.
Based on the following property data and community guidelines, provide a helpful, detailed response to the user's question.

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer that:
- Directly addresses the user's question
- Uses specific details from the provided data (e.g., Property ID, prices, locations, amenities)
- Offers practical insights and recommendations
- Maintains a friendly, professional tone
- Organizes information clearly with bullet points when appropriate
- If the question is about specific properties, include details like Property ID, price, location, and amenities
- If the question involves community guidelines, reference specific rules or policies
- If the question is general, provide market insights based on the available data
"""
    )
    
    try:
        if not get_gemini_client()[0]:
            return "Gemini client not initialized."
        
        llm = ChatGoogleGenerativeAI(model=GENERATIVE_MODEL, google_api_key=API_KEY, temperature=0.7)
        prompt = prompt_template.format(context=context, query=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"I apologize, but I encountered an error while generating the response: {str(e)}"

def load_and_process_data(csv_file, pdf_file) -> Tuple[str, List[str]]:
    if not csv_file and not pdf_file:
        return "Please upload at least one file (Excel/CSV or PDF).", []
    
    all_documents = []
    output = []
    if csv_file:
        csv_docs = load_csv_data(csv_file)
        all_documents.extend(csv_docs)
        output.append(f"✅ Loaded {len(csv_docs)} property records")
    if pdf_file:
        pdf_docs = load_pdf_data(pdf_file)
        all_documents.extend(pdf_docs)
        output.append(f"✅ Loaded {len(pdf_docs)} guideline sections")
    
    if all_documents:
        index = SmartFaissIndex()
        output.append("🔍 Building search index...")
        progress_messages = index.add_documents(all_documents)
        output.extend(progress_messages)
        state["index"] = index
        output.append("🎉 Your real estate assistant is ready! Start asking questions below.")
    else:
        output.append("No documents were loaded. Please check your uploaded files.")
    
    return "\n".join(output), progress_messages

# Flask Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/client-status', methods=['GET'])
def client_status():
    success, message = get_gemini_client()
    return jsonify({"status": "success" if success else "error", "message": message})

@app.route('/upload', methods=['POST'])
def upload_files():
    csv_file = request.files.get('csv_file')
    pdf_file = request.files.get('pdf_file')
    message, progress = load_and_process_data(
        csv_file.read() if csv_file else None,
        pdf_file.read() if pdf_file else None
    )
    return jsonify({"message": message, "progress": progress})

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"response": "Please enter a query.", "chat_history": state["chat_history"]})
    
    if state["index"] is None:
        return jsonify({"response": "Please load data first by uploading files.", "chat_history": state["chat_history"]})
    
    relevant_docs = state["index"].search(query, top_k=5)
    response = generate_intelligent_response(query, relevant_docs)
    state["chat_history"].append((query, response))
    return jsonify({"response": response, "chat_history": state["chat_history"]})

@app.route('/quick-action', methods=['POST'])
def quick_action():
    data = request.get_json()
    action = data.get('action')
    if action == 'budget':
        query = "Show me affordable properties under ₹1 crore in Chennai"
    elif action == 'locations':
        query = "What are the most popular locations for real estate investment in Chennai?"
    elif action == 'insights':
        query = "Give me insights about current real estate market trends in Chennai"
    else:
        return jsonify({"response": "Invalid action.", "chat_history": state["chat_history"]})
    
    if state["index"] is None:
        return jsonify({"response": "Please load data first by uploading files.", "chat_history": state["chat_history"]})
    
    relevant_docs = state["index"].search(query, top_k=5)
    response = generate_intelligent_response(query, relevant_docs)
    state["chat_history"].append((query, response))
    return jsonify({"response": response, "chat_history": state["chat_history"]})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    state["chat_history"] = []
    return jsonify({"message": "Chat history cleared!", "chat_history": state["chat_history"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
