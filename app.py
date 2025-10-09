"""
Streamlit UI for Gray's Anatomy AI Agent
"""
import streamlit as st
import os

# Import our modules
from search import search, load_search_index
from llm import get_answer
from config import USE_OPENAI, LLM_MODEL, TOP_K_RESULTS

# Page config
st.set_page_config(
    page_title="Gray's Anatomy AI",
    layout="wide",
    page_icon="🧠"
)

# Header
st.title("🧠 Gray's Anatomy AI Assistant")
st.markdown("""
Ask questions about human anatomy based on **Gray's Anatomy (1918 edition)**.
Uses BM25 keyword search + LLM for intelligent answers.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    
    # Model info
    model_name = "OpenAI GPT-3.5" if USE_OPENAI else LLM_MODEL.split('/')[-1]
    st.markdown(f"""
    **Current Setup:**
    - 🤖 LLM: {model_name}
    - 🔍 Search: BM25 (keyword)
    - 📚 Source: Gray's Anatomy (1918)
    - 🎯 Results: Top {TOP_K_RESULTS} chunks
    """)
    
    st.markdown("---")
    
    # Status checks
    st.subheader("📊 Status")
    
    # Check if index exists
    if os.path.exists("bm25_index.pkl"):
        st.success("✅ Search index ready")
    else:
        st.warning("⏳ Building index on first query...")
    
    # Check API key
    if USE_OPENAI:
        if os.environ.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI key configured")
        else:
            st.error("❌ OPENAI_API_KEY missing")
    else:
        if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
            st.success("✅ HuggingFace token configured")
        else:
            st.error("❌ HUGGINGFACEHUB_API_TOKEN missing")
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "What are the bones of the skull?",
        "Describe the heart chambers",
        "What is the brachial plexus?",
        "Explain the vertebral column",
        "What muscles control breathing?",
        "Describe the liver structure",
        "What are the cranial nerves?",
        "Explain the knee joint",
        "What is the spinal cord?",
        "Describe blood circulation"
    ]
    
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"example_{i}"):
            st.session_state.example_clicked = ex

# Initialize search index (cached)
@st.cache_resource
def init_system():
    """Initialize the search system"""
    try:
        bm25, chunks = load_search_index()
        return True, f"Ready! Searching {len(chunks):,} text chunks"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Initialize
with st.spinner("🔧 Initializing system..."):
    success, message = init_system()
    
if success:
    st.success(f"✅ {message}")
else:
    st.error(f"❌ {message}")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input area
col1, col2 = st.columns([5, 1])

with col1:
    # Check if example was clicked
    default_value = ""
    if "example_clicked" in st.session_state:
        default_value = st.session_state.example_clicked
        del st.session_state.example_clicked
    
    user_question = st.text_input(
        "Your question:",
        value=default_value,
        placeholder="e.g., What are the bones of the skull?",
        key="question_input"
    )

with col2:
    st.write("")  # Spacing
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# Clear button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Process question
if (ask_button or user_question) and user_question.strip():
    
    # Check if already answered (avoid duplicate on rerun)
    already_answered = False
    if st.session_state.chat_history:
        if st.session_state.chat_history[-1]["question"] == user_question:
            already_answered = True
    
    if not already_answered:
        with st.spinner("🔍 Searching... 🤔 Thinking..."):
            try:
                # Step 1: Search for relevant chunks
                relevant_chunks = search(user_question, k=TOP_K_RESULTS)
                context = "\n\n".join(relevant_chunks)
                
                # Step 2: Get LLM answer
                answer = get_answer(context, user_question)
                
                # Step 3: Save to history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
                # Show helpful debug info
                if "HUGGINGFACEHUB_API_TOKEN" in str(e):
                    st.info("💡 Add your HuggingFace token in Streamlit Cloud: Settings → Secrets")
                    st.code('HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"')
                elif "OPENAI_API_KEY" in str(e):
                    st.info("💡 Add your OpenAI key in Streamlit Cloud: Settings → Secrets")
                    st.code('OPENAI_API_KEY = "sk-your_key_here"')

# Display chat history
st.markdown("---")

if st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    # Display in reverse order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"### ❓ {chat['question']}")
            st.markdown(chat['answer'])
            
            # Add some visual separation
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
else:
    st.info("👆 Ask a question above or click an example from the sidebar to get started!")

# Footer
st.markdown("---")
st.caption("""
⚠️ **Disclaimer:** This AI uses Gray's Anatomy (1918 edition). While anatomical fundamentals 
remain accurate, medical terminology and some concepts may reflect the knowledge of that era.
Always consult current medical resources for clinical decisions.
""")

# Debug info (optional)
with st.expander("🔧 Debug Info"):
    st.json({
        "USE_OPENAI": USE_OPENAI,
        "LLM_MODEL": LLM_MODEL,
        "TOP_K_RESULTS": TOP_K_RESULTS,
        "Index exists": os.path.exists("bm25_index.pkl"),
        "Text file exists": os.path.exists("grays_anatomy.txt"),
        "Chat history length": len(st.session_state.chat_history)
    })