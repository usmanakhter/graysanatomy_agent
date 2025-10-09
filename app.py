import os
import requests
import streamlit as st
import pickle
from rank_bm25 import BM25Okapi
import numpy as np

# -------------------------------
# Configuration - MUCH FASTER
# -------------------------------
USE_OPENAI = False  # Set to True if you want to use OpenAI (costs ~$0.01/query)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# -------------------------------
# Fast BM25 Search (No embeddings needed!)
# -------------------------------
@st.cache_resource
def download_and_chunk_text():
    """Download and split text into chunks - FAST"""
    filepath = "grays_anatomy.txt"
    
    if not os.path.exists(filepath):
        st.info("📥 Downloading Gray's Anatomy... (30 seconds)")
        url = "https://archive.org/stream/anatomyofhumanbo1918gray/anatomyofhumanbo1918gray_djvu.txt"
        response = requests.get(url, timeout=60)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple chunking
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if len(chunk) > 100:  # Skip tiny chunks
            chunks.append(chunk)
    
    return chunks

@st.cache_resource
def create_bm25_index():
    """Create BM25 index - VERY FAST (1-2 minutes max)"""
    st.info("🔨 Building search index... (1-2 minutes, one-time only)")
    
    chunks = download_and_chunk_text()
    
    # Tokenize for BM25
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Cache it
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump((bm25, chunks), f)
    
    st.success(f"✅ Indexed {len(chunks)} chunks!")
    return bm25, chunks

@st.cache_resource
def load_search_index():
    """Load or create search index"""
    if os.path.exists("bm25_index.pkl"):
        with open("bm25_index.pkl", "rb") as f:
            bm25, chunks = pickle.load(f)
        return bm25, chunks
    else:
        return create_bm25_index()

def search_documents(query, k=5):
    """Fast keyword search"""
    bm25, chunks = load_search_index()
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get top k results
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-k:][::-1]
    
    results = [chunks[i] for i in top_indices]
    return results

# -------------------------------
# LLM Options (Choose one)
# -------------------------------
def get_llm_response(context, question):
    """Get response from LLM"""
    
    if USE_OPENAI:
        # Option 1: OpenAI (best quality, costs ~$0.01/query)
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        prompt = f"""Based on the following excerpts from Gray's Anatomy, answer the question accurately.

Context:
{context}

Question: {question}

Answer concisely and educationally:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    else:
        # Option 2: Free HuggingFace (but faster API)
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        
        if not api_token:
            st.error("❌ Set HUGGINGFACEHUB_API_TOKEN in Streamlit secrets")
            st.stop()
        
        # Use HuggingFace Inference API (faster than loading models)
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        prompt = f"""<s>[INST] Based on the following excerpts from Gray's Anatomy, answer the question accurately and concisely.

Context from Gray's Anatomy:
{context[:3000]}

Question: {question}

Provide a clear, educational answer: [/INST]"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Error generating response")
            return str(result)
        else:
            return f"Error: {response.status_code} - {response.text}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Gray's Anatomy AI", layout="wide", page_icon="🧠")

st.title("🧠 Gray's Anatomy AI Assistant (Fast Version)")
st.markdown("""
Ask questions about human anatomy based on **Gray's Anatomy (1918)**.
Uses **BM25 keyword search** (no heavy embeddings) + LLM for answers.
""")

# Sidebar
with st.sidebar:
    st.header("⚡ Fast Mode Features")
    st.markdown("""
    **Why this is faster:**
    - ✅ BM25 keyword search (no embeddings)
    - ✅ Lightweight indexing (~2 min first time)
    - ✅ Direct API calls (no model loading)
    - ✅ Simple chunking
    
    **Trade-offs:**
    - Keyword-based (not semantic)
    - Good for specific terms
    - May miss conceptual matches
    """)
    
    st.markdown("---")
    
    # Check status
    if os.path.exists("bm25_index.pkl"):
        st.success("✅ Search index ready")
    else:
        st.warning("⏳ First run: ~2 minutes")
    
    if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.success("✅ API token configured")
    elif os.environ.get("OPENAI_API_KEY"):
        st.success("✅ OpenAI key configured")
    else:
        st.error("❌ No API key found")
    
    st.markdown("---")
    st.header("💡 Example Questions")
    
    examples = [
        "What are the bones of the skull?",
        "Describe the heart chambers",
        "What is the brachial plexus?",
        "Explain the vertebral column",
        "What muscles control breathing?",
        "Describe the liver structure",
        "What are the cranial nerves?",
        "Explain the knee joint"
    ]

# Initialize
try:
    bm25, chunks = load_search_index()
    st.success(f"✅ Ready! Searching {len(chunks):,} text chunks")
except Exception as e:
    st.error(f"❌ Error loading search index: {e}")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input("Ask a question about anatomy:", key="user_input")
with col2:
    st.write("")
    ask_button = st.button("🔍 Ask", type="primary")

# Clear button
if st.button("🗑️ Clear"):
    st.session_state.chat_history = []
    st.rerun()

# Example buttons in sidebar
for ex in examples:
    if st.sidebar.button(ex, key=f"ex_{ex[:20]}"):
        user_question = ex
        ask_button = True

# Process question
if ask_button and user_question and user_question.strip():
    with st.spinner("🔍 Searching... 🤔 Thinking..."):
        try:
            # Search for relevant chunks
            relevant_chunks = search_documents(user_question, k=5)
            context = "\n\n".join(relevant_chunks)
            
            # Get LLM response
            answer = get_llm_response(context, user_question)
            
            # Save to history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display chat history
st.markdown("---")
if st.session_state.chat_history:
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"### ❓ {chat['question']}")
        st.markdown(chat['answer'])
        st.markdown("---")
else:
    st.info("👆 Ask a question to get started!")

# Footer
st.caption("""
⚠️ **Note:** Uses Gray's Anatomy (1918). Fast keyword search may miss some semantic matches.
For better semantic understanding, upgrade to the full vector store version.
""")