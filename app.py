"""
Gray's Anatomy AI Assistant - Slice 1
Working end-to-end system: Question → BM25 Search → Mistral-7B → Answer
"""


import streamlit as st
import time
import os
from orchestrator import Orchestrator
from config import DEFAULT_SETTINGS, LLM_OPTIONS

from dotenv import load_dotenv
load_dotenv()


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
    .metric-card {
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
This system uses advanced RAG (Retrieval-Augmented Generation) architecture.
""")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    st.markdown("**🤖 LLM Model**")
    llm_choice = st.selectbox(
        "Select language model:",
        options=list(LLM_OPTIONS.keys()),
        format_func=lambda x: f"{x.upper()} - {LLM_OPTIONS[x]['description']}",
        help="GPT and Claude models available",
        key="llm_select"
    )
    
    # API Key for selected model
    st.markdown("**🔑 API Key**")
    required_key = LLM_OPTIONS[llm_choice]["requires_key"]
    provider = "OpenAI" if "OPENAI" in required_key else "Anthropic"
    
    # Initialize session state for key validation
    if "key_valid" not in st.session_state:
        st.session_state.key_valid = False
        
    # Show key input for selected model
    key_input = st.text_input(
        f"{provider} API Key:",
        type="password",
        key=f"{required_key.lower()}_input",
        help=f"Enter your {provider} API key for {llm_choice.upper()}"
    )
    
    # Validate and store key
    if key_input:
        if key_input != os.environ.get(required_key, ""):
            os.environ[required_key] = key_input
            # Clear orchestrator cache to reinitialize with new key
            if "orchestrator" in st.session_state:
                del st.session_state["orchestrator"]
            st.session_state.key_valid = True
            st.success(f"✅ {provider} API key updated")
            st.rerun()  # Refresh to update status
    
    # Slice 1: Only Lexical search available
    st.markdown("**🔍 Search Strategy**")
    search_choice = st.selectbox(
        "Select search method:",
        options=["lexical"],
        format_func=lambda x: "Lexical (BM25 - Keyword Search)",
        help="Semantic and Hybrid search coming in Slice 3!",
        key="search_select"
    )
    
    # Slice 1: No graph available yet
    st.markdown("**🕸️ Knowledge Graph**")
    graph_choice = st.selectbox(
        "Enable knowledge graph:",
        options=["none"],
        format_func=lambda x: "None (coming in Slice 4!)",
        help="Knowledge graph features coming in Slice 4!",
        key="graph_select"
    )
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        top_k = st.slider(
            "Number of sources to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="How many text chunks to use as context"
        )
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    
    # Check components
    if os.path.exists("grays_anatomy.txt"):
        st.success("✅ Text data ready")
    else:
        st.info("⏳ Will download on first use")
    
    if os.path.exists("indexes/bm25_index.pkl"):
        st.success("✅ Search index ready")
    else:
        st.info("⏳ Will build on first query (~2 min)")
    
    # Check API key for selected model
    required_key = LLM_OPTIONS[llm_choice]["requires_key"]
    if os.environ.get(required_key):
        model_name = llm_choice.upper()
        st.success(f"✅ {model_name} ready")
    else:
        provider = "OpenAI" if "OPENAI" in required_key else "Anthropic"
        get_key_url = ("https://platform.openai.com/api-keys" if provider == "OpenAI" 
                      else "https://console.anthropic.com/settings/keys")
        st.error(f"❌ {provider} API key required")
        st.caption(f"[Get your API key here]({get_key_url})")
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Example Questions")
    examples = [
        "What are the bones of the skull?",
        "Describe the structure of the heart",
        "What muscles control breathing?",
        "Explain the vertebral column",
        "What is the brachial plexus?",
        "Describe the liver anatomy",
        "What are the cranial nerves?",
        "Explain the knee joint structure"
    ]
    
    for i, example in enumerate(examples):
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.selected_example = example

# Initialize orchestrator
@st.cache_resource
def get_orchestrator(llm, search, graph, k):
    """Initialize orchestrator with current settings"""
    settings = {
        "llm": llm,
        "search_strategy": search,
        "knowledge_graph": graph,
        "top_k": k,
        "embedding": "minilm"  # Not used in Slice 1
    }
    return Orchestrator(settings)

# Main content area
col1, col2 = st.columns([4, 1])

with col1:
    # Check if example was selected
    default_question = ""
    if "selected_example" in st.session_state:
        default_question = st.session_state.selected_example
        del st.session_state.selected_example
    
    question = st.text_input(
        "Your question:",
        value=default_question,
        placeholder="e.g., What are the main bones of the skull?",
        key="question_input"
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

# Process question
if (ask_button or question) and question.strip():
    
    # Avoid duplicate processing
    if st.session_state.chat_history and st.session_state.chat_history[-1].get("question") == question:
        pass  # Already processed
    else:
        with st.spinner("🔍 Searching Gray's Anatomy... 🤔 Generating answer..."):
            try:
                start_time = time.time()
                
                # Get orchestrator
                orchestrator = get_orchestrator(
                    llm=llm_choice,
                    search=search_choice,
                    graph=graph_choice,
                    k=top_k
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
                    "elapsed_time": elapsed
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
                # Helpful debugging info
                error_lower = str(e).lower()
                if "invalid" in error_lower and "api key" in error_lower:
                    # Reset key validation state
                    st.session_state.key_valid = False
                    required_key = LLM_OPTIONS[llm_choice]["requires_key"]
                    provider = "OpenAI" if "OPENAI" in required_key else "Anthropic"
                    get_key_url = ("https://platform.openai.com/api-keys" if provider == "OpenAI" 
                                 else "https://console.anthropic.com/settings/keys")
                    st.error(f"❌ Invalid {provider} API key")
                    st.info(f"💡 Get a valid API key at: {get_key_url}")
                    if required_key in os.environ:
                        del os.environ[required_key]  # Clear invalid key
                elif "rate limit" in error_lower:
                    st.warning("⏸️ Rate limit reached. Please wait a few minutes and try again.")
                    st.info("💡 Consider switching to a different model temporarily")

# Display chat history
st.markdown("---")

if st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # Question
            st.markdown(f"### ❓ {chat['question']}")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Response Time", f"{chat['elapsed_time']:.2f}s")
            with col2:
                st.metric("📚 Sources Used", chat['metadata']['num_sources'])
            with col3:
                st.metric("🤖 Model", chat['metadata']['llm_model'].upper())
            
            # Answer
            st.markdown("**Answer:**")
            st.markdown(chat['answer'])
            
            # Sources (expandable)
            with st.expander("📖 View Retrieved Sources"):
                for j, source in enumerate(chat['sources'], 1):
                    st.markdown(f"**Source {j}** (Score: {source['score']:.3f})")
                    st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                    st.markdown("---")
            
            # Visual separator
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")

else:
    st.info("👆 **Get started!** Ask a question above or click an example from the sidebar.")
    
    st.markdown("### 🎯 What can you ask about?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🦴 Bones & Skeleton**")
        st.caption("Skull, spine, limbs, joints")
    
    with col2:
        st.markdown("**💪 Muscles & Movement**")
        st.caption("Muscle groups, attachments, actions")
    
    with col3:
        st.markdown("**🫀 Organs & Systems**")
        st.caption("Heart, lungs, brain, digestive system")

# Footer
st.markdown("---")
st.caption("""
**Slice 1 Status:** ✅ Core system operational  
**Next:** Slice 2 (More LLM choices) → Slice 3 (Semantic search) → Slice 4 (Knowledge graph)

⚠️ **Note:** This uses Gray's Anatomy (1918). While anatomical fundamentals remain accurate, 
terminology reflects that era. Always consult current medical resources for clinical decisions.
""")