import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.serving import WSGIRequestHandler
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompts import bhogiai_system_prompt, general_system_prompt
import fitz
import pytesseract
from PIL import Image
import json
from datetime import datetime
import traceback
from functools import lru_cache
import time

load_dotenv()

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='uploads',
    SEND_FILE_MAX_AGE_DEFAULT=0,
)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Custom request handler to fix Flask dev server connection reset issue
class ForkedRequestHandler(WSGIRequestHandler):
    """Fix for Flask dev server file upload connection reset"""
    def handle(self):
        try:
            WSGIRequestHandler.handle(self)
        except ConnectionResetError:
            print("⚠️ Connection reset by client - this is normal for Flask dev server file uploads")
        except Exception as e:
            print(f"⚠️ Request handler error: {e}")
        finally:
            try:
                self.close_connection = True
            except:
                pass

@app.before_request
def consume_request_data():
    """Ensure request data is consumed to prevent connection resets"""
    if request.method == 'POST' and request.content_type and 'multipart/form-data' in request.content_type:
        try:
            _ = request.files
            _ = request.form
        except:
            pass

# Initialize OpenAI client for OpenRouter + Fast Models
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", 
        api_key=os.getenv("OPENROUTER_API_KEY"),  
        default_headers={
            "HTTP-Referer": "http://localhost:5000", 
            "X-Title": "NEXA AI Assistant"  
        }
    )
    print("✅ OpenRouter client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing OpenRouter client: {e}")
    client = None

# Global variables with caching
vectorstore = None
history_cache = []
pdf_loaded = False
current_pdf_name = ""

# Response cache for common queries
response_cache = {}
CACHE_EXPIRY = 300  # 5 minutes

# [Keep your existing PDF functions - they're fine]
def load_pdf_docs(pdf_path: str):
    docs = []
    try:
        print(f"📄 Opening PDF: {pdf_path}")
        pdf = fitz.open(pdf_path)
        print(f"📄 PDF opened successfully, {len(pdf)} pages")
        
        for i, page in enumerate(pdf):
            try:
                text = page.get_text()
                if not text.strip():
                    print(f"📄 Page {i+1}: No text found, trying OCR...")
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    print(f"📄 Page {i+1}: OCR extracted {len(text)} characters")
                
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"page": i + 1}))
                    print(f"📄 Page {i+1}: Added to documents")
            except Exception as e:
                print(f"❌ Error processing page {i+1}: {e}")
                continue
                
        pdf.close()
        print(f"✅ PDF processing complete: {len(docs)} pages processed")
        return docs if docs else None
        
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return None

def build_vectorstore_from_text(raw_texts):
    try:
        print("🔧 Building vectorstore...")
        # Smaller chunks for faster search
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=25)  # Reduced size
        docs = splitter.split_text(raw_texts) if isinstance(raw_texts, str) else splitter.split_documents(raw_texts)
        print(f"🔧 Text split into {len(docs)} chunks")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("🔧 Embeddings model loaded")
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        print("✅ Vectorstore created successfully")
        return vectorstore
    except Exception as e:
        print(f"❌ Error building vectorstore: {e}")
        return None

@lru_cache(maxsize=100)  # Cache search results
def get_relevant_context(query: str, k: int = 2) -> str:  # Reduced k from 3 to 2
    if vectorstore and pdf_loaded:
        try:
            results = vectorstore.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in results])
        except Exception as e:
            print(f"❌ Error searching vectorstore: {e}")
            return ""
    return ""

def get_response(message: str) -> dict:
    global history_cache, response_cache
    
    start_time = time.time()
    
    if not message.strip():
        return {"error": "Please type a question."}

    if not client:
        return {"error": "AI client not initialized. Check your API key."}

    # Check cache first
    cache_key = f"{message}_{pdf_loaded}_{current_pdf_name}"
    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        if time.time() - cached_response['timestamp'] < CACHE_EXPIRY:
            print(f"⚡ Cache hit - Response time: {(time.time() - start_time)*1000:.0f}ms")
            return cached_response['data']

    context = get_relevant_context(message)
    
    # Simplified prompts for speed
    if pdf_loaded and vectorstore:
        if context.strip():
            system_prompt = f"Based on this PDF content: {context[:500]}...\n\nQuestion: {message}\nAnswer concisely:"
            response_prefix = "📄 **PDF:** "
        else:
            system_prompt = f"No relevant PDF content found.\nQuestion: {message}\nAnswer briefly:"
            response_prefix = "📄 **No match:** "
    else:
        system_prompt = f"Question: {message}\nAnswer concisely:"
        response_prefix = "🧠 **General:** "

    try:
        # Keep only last 4 exchanges for speed
        recent_history = history_cache[-8:] if len(history_cache) > 8 else history_cache
        
        messages = [{"role": "system", "content": system_prompt}] + recent_history + [{"role": "user", "content": message}]
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",  # Fast model
            messages=messages,
            max_tokens=100,
            temperature=0.1,
        )
        
        # ✅ FIXED: Handle different response structures
        try:
            # Try standard OpenAI format first
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message'):
                    answer = response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    answer = response.choices[0].text
                else:
                    answer = str(response.choices[0])
            else:
                # Handle if response is a list or different structure
                if isinstance(response, list):
                    answer = str(response[0]) if response else "No response received"
                else:
                    answer = str(response)
        except Exception as parse_error:
            print(f"⚠️ Response parsing error: {parse_error}")
            print(f"⚠️ Response structure: {type(response)}")
            print(f"⚠️ Response content: {response}")
            answer = "Sorry, I received an unexpected response format. Please try again."
        
        # Update history (keep smaller)
        history_cache.append({"role": "user", "content": message})
        history_cache.append({"role": "assistant", "content": answer})
        
        # Keep only last 8 messages total
        if len(history_cache) > 8:
            history_cache = history_cache[-8:]
        
        result = {
            "answer": answer,
            "prefix": response_prefix,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # Cache the response
        response_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        print(f"⚡ API Response time: {(time.time() - start_time)*1000:.0f}ms")
        return result
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        print(f"❌ Full error traceback: {traceback.format_exc()}")
        return {"error": f"Error: {str(e)}"}


# [Keep all your existing Flask routes - they're fine]
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore, pdf_loaded, current_pdf_name, response_cache
    
    try:
        print("📤 Upload request received")
        
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Please upload a valid PDF file"}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"📤 Saving file: {filename}")
        file.save(filepath)
        
        # Process PDF
        new_docs = load_pdf_docs(filepath)
        
        if not new_docs:
            os.remove(filepath)
            return jsonify({
                "success": False, 
                "error": "No text found in PDF. Please ensure the PDF contains readable text."
            }), 400
        
        # Build vectorstore
        if vectorstore:
            vectorstore.add_documents(new_docs)
        else:
            vectorstore = build_vectorstore_from_text(new_docs)
            if not vectorstore:
                os.remove(filepath)
                return jsonify({
                    "success": False, 
                    "error": "Failed to process PDF. Please try again."
                }), 500
        
        pdf_loaded = True
        current_pdf_name = filename
        
        # Clear cache when new PDF uploaded
        response_cache.clear()
        get_relevant_context.cache_clear()
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            "success": True,
            "message": f"PDF uploaded and processed successfully! ({len(new_docs)} pages loaded)",
            "filename": filename,
            "pages": len(new_docs)
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        response = get_response(data['question'])
        return jsonify(response)
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    try:
        status = {
            "mode": "pdf" if pdf_loaded else "general",
            "message": f"PDF loaded - Ready for queries" if pdf_loaded else "General knowledge mode",
            "pdf_name": current_pdf_name if pdf_loaded else ""
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_data():
    global history_cache, vectorstore, pdf_loaded, current_pdf_name, response_cache
    try:
        history_cache = []
        vectorstore = None
        pdf_loaded = False
        current_pdf_name = ""
        response_cache.clear()
        get_relevant_context.cache_clear()
        
        return jsonify({
            "success": True,
            "message": "All data cleared successfully!"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting NEXA AI Assistant...")
    print("⚡ Optimized for SPEED - Fast model + caching enabled")
    print("💡 For production, consider using Gunicorn or Waitress")
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000, 
        threaded=True,
        request_handler=ForkedRequestHandler,
        use_reloader=False
    )
