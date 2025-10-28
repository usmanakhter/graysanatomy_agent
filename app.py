"""
Gray's Anatomy AI Assistant - Slice 3
Semantic + Hybrid Search with GPT/Anthropic
Temperature = 0 for deterministic, context-only answers
"""
import streamlit as st
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from orchestrator import Orchestrator
from config import (
    DEFAULT_SETTINGS, 
    LLM_OPTIONS, 
    EMBEDDING_OPTIONS, 
    SEARCH_STRATEGIES
)

# Page config
st.set_page_config(
    page_title="Gray's Anatomy AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .config-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧠 Gray's Anatomy AI Assistant")
st.markdown("""
Ask questions about human anatomy based on **Gray's Anatomy (1918 edition)**.  
**Answers are derived ONLY from the provided text** (Temperature = 0, no external knowledge).
""")

# Main content area
col1, col2 = st.columns([4, 1])

with col1:
    # Check if example was selected
    default_question = ""
    auto_ask = False
    if "selected_example" in st.session_state:
        default_question = st.session_state.selected_example
        auto_ask = True
        del st.session_state.selected_example
    
    def handle_enter():
        if st.session_state.question_input.strip():
            st.session_state.auto_ask = True
    
    question = st.text_input(
        "Your question:",
        value=default_question,
        placeholder="e.g., What are the main bones of the skull?",
        key="question_input",
        on_change=handle_enter
    )

with col2:
    st.write("")  # Spacing
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# Clear chat button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history or welcome message
if st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # Question
            st.markdown(f"### ❓ {chat['question']}")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Time", f"{chat['elapsed_time']:.2f}s")
            with col2:
                st.metric("📚 Sources", chat['metadata']['num_sources'])
            with col3:
                st.metric("🔍 Search", chat['settings']['search'].title())
            with col4:
                st.metric("🤖 Model", chat['settings']['llm'].upper())
            
            # Answer
            st.markdown("**Answer:**")
            st.markdown(chat['answer'])
            
            # Sources (expandable)
            with st.expander(f"📖 View {len(chat['sources'])} Retrieved Sources"):
                for j, source in enumerate(chat['sources'], 1):
                    st.markdown(f"**Source {j}** ({source['method']} - Score: {source['score']:.3f})")
                    st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                    if j < len(chat['sources']):
                        st.markdown("---")
            
            # Visual separator
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
else:
    st.info("👆 **Get started!** Ask a question above or click an example from the sidebar.")
    
    st.markdown("### 🎯 What's New in Slice 3?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 Semantic Search**")
        st.caption("Understanding meaning, not just keywords")
    
    with col2:
        st.markdown("**⚡ Hybrid Search**")
        st.caption("Best of both: keywords + semantics")
    
    with col3:
        st.markdown("**🎯 Context-Only**")
        st.caption("Temperature=0, no external knowledge")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Selection
    st.markdown("### 🤖 Language Model")
    llm_choice = st.selectbox(
        "Select LLM:",
        options=list(LLM_OPTIONS.keys()),
        format_func=lambda x: f"{LLM_OPTIONS[x]['model']} - {LLM_OPTIONS[x]['description']}",
        index=list(LLM_OPTIONS.keys()).index(DEFAULT_SETTINGS["llm"]),
        key="llm_select"
    )
    
    st.caption(f"Provider: {LLM_OPTIONS[llm_choice]['provider'].upper()}")
    st.caption(f"Context: {LLM_OPTIONS[llm_choice]['context_length']:,} tokens")
    
    st.markdown("---")
    
    # Search Strategy Selection
    st.markdown("### 🔍 Search Strategy")
    search_choice = st.selectbox(
        "Select search method:",
        options=list(SEARCH_STRATEGIES.keys()),
        format_func=lambda x: f"{SEARCH_STRATEGIES[x]['name']} - {SEARCH_STRATEGIES[x]['description']}",
        index=list(SEARCH_STRATEGIES.keys()).index(DEFAULT_SETTINGS["search_strategy"]),
        key="search_select"
    )
    
    st.caption(f"Speed: {SEARCH_STRATEGIES[search_choice]['speed']}")
    
    # Embedding Selection (only for semantic/hybrid)
    embedding_choice = DEFAULT_SETTINGS["embedding"]
    if search_choice in ["semantic", "hybrid"]:
        st.markdown("#### Embedding Model")
        embedding_choice = st.selectbox(
            "Select embeddings:",
            options=list(EMBEDDING_OPTIONS.keys()),
            format_func=lambda x: f"{EMBEDDING_OPTIONS[x]['description']} ({EMBEDDING_OPTIONS[x]['speed']})",
            index=list(EMBEDDING_OPTIONS.keys()).index(DEFAULT_SETTINGS["embedding"]),
            key="embedding_select"
        )
        
        st.caption(f"Dimension: {EMBEDDING_OPTIONS[embedding_choice]['dimension']}")
        if EMBEDDING_OPTIONS[embedding_choice].get("size_mb"):
            st.caption(f"Size: {EMBEDDING_OPTIONS[embedding_choice]['size_mb']}MB")
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        top_k = st.slider(
            "Number of sources:",
            min_value=1,
            max_value=10,
            value=DEFAULT_SETTINGS["top_k"],
            help="How many text chunks to retrieve"
        )
        
        st.info("🌡️ Temperature: **0.0** (deterministic, context-only answers)")
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    
    # Check files
    if os.path.exists("grays_anatomy.txt"):
        st.success("✅ Text data ready")
    else:
        st.info("⏳ Will download on first use")
    
    # Check indexes
    if os.path.exists("indexes/bm25_index.pkl"):
        st.success("✅ Lexical index ready")
    else:
        st.info("⏳ Building on first lexical search")
    
    if search_choice in ["semantic", "hybrid"]:
        vector_path = f"indexes/vector_store/{embedding_choice}_index.faiss"
        if os.path.exists(vector_path):
            st.success(f"✅ Vector index ready ({embedding_choice})")
        else:
            st.warning(f"⏳ Will build vector index (~5-10 min first time)")
    
    # API Key Management
    st.markdown("### 🔑 API Keys")
    
    # Initialize session state for API keys if not exists
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}
    
    # Required API key for current model
    provider = LLM_OPTIONS[llm_choice]["provider"]
    key_name = LLM_OPTIONS[llm_choice]["requires_key"]
    key_url = LLM_OPTIONS[llm_choice]["key_url"]
    
    # Check both environment and session state
    api_key = os.environ.get(key_name) or st.session_state.api_keys.get(key_name)
    
    if not api_key:
        st.markdown(f"#### 🔑 {provider.upper()} API Key Required")
        new_key = st.text_input(
            f"Enter your {key_name}:",
            type="password",
            key=f"input_{key_name}",
            help=f"Get your API key at: {key_url}"
        )
        if new_key:
            # Save to session state (temporary)
            st.session_state.api_keys[key_name] = new_key
            st.success(f"✅ {provider.upper()} API key saved to session")
            api_key = new_key
        else:
            # No API key provided
            st.error(f"❌ {key_name} required")
            st.stop()
    else:
        st.success(f"✅ {provider.upper()} API configured")
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "WWhat are the sympathetic efferent fibers?",
        "What protects the skull?",
        "Why does my leg hurt in the back?"
    ]
    
    for i, example in enumerate(examples):
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.selected_example = example

# Initialize orchestrator with current settings
def get_orchestrator(_llm, _search, _embedding, _k, _api_keys):
    """Initialize orchestrator with current settings"""
    settings = {
        "llm": _llm,
        "search_strategy": _search,
        "embedding": _embedding,
        "top_k": _k,
        "knowledge_graph": "none",
    }
    
    # Add API keys to settings
    settings.update(_api_keys)
    
    return Orchestrator(settings)

# Process question
if (ask_button or auto_ask or st.session_state.get('auto_ask', False)) and question.strip():
    # Reset auto_ask flag
    if 'auto_ask' in st.session_state:
        st.session_state.auto_ask = False
    # Show appropriate loading message
    if search_choice == "semantic" or search_choice == "hybrid":
        loading_msg = "🔍 Searching (semantic analysis)... 🤔 Generating answer..."
    else:
        loading_msg = "🔍 Searching Gray's Anatomy... 🤔 Generating answer..."
    
    with st.spinner(loading_msg):
        try:
            start_time = time.time()
            
            # Set API keys in environment from session state
            if hasattr(st.session_state, 'api_keys'):
                for key, value in st.session_state.api_keys.items():
                    os.environ[key] = value
            
            orchestrator = get_orchestrator(
                _llm=llm_choice,
                _search=search_choice,
                _embedding=embedding_choice,
                _k=top_k,
                _api_keys={}
            )
            
            # Query the system
            result = orchestrator.query(question)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "metadata": result["metadata"],
                "elapsed_time": elapsed,
                "settings": {
                    "llm": llm_choice,
                    "search": search_choice,
                    "embedding": embedding_choice if search_choice in ["semantic", "hybrid"] else None
                }
            })
            
            # Refresh the UI to show the new response
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Helpful debugging info
            if "OPENAI_API_KEY" in str(e):
                st.info("💡 **How to fix:**")
                st.code('# Add to .env file:\nOPENAI_API_KEY=sk-your_key_here')
                st.markdown("Get key at: https://platform.openai.com/api-keys")
            elif "ANTHROPIC_API_KEY" in str(e):
                st.info("💡 **How to fix:**")
                st.code('# Add to .env file:\nANTHROPIC_API_KEY=sk-ant-your_key_here')
                st.markdown("Get key at: https://console.anthropic.com/settings/keys")





# Footer
st.markdown("---")
st.caption("""
**Slice 3 Status:** ✅ All search strategies operational  
**Temperature:** 0.0 (deterministic, context-only answers)  
**Next:** Slice 4 (Knowledge Graph + GraphRAG)

⚠️ **Note:** This uses Gray's Anatomy (1918). Answers are based EXCLUSIVELY on the provided text.
Always consult current medical resources for clinical decisions.
""")