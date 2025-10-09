import os
import requests
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# -------------------------------
# Configuration
# -------------------------------
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Much better than all-MiniLM-L6-v2
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Powerful model
CHUNK_SIZE = 1500  # Larger for more context
CHUNK_OVERLAP = 300  # More overlap
RETRIEVAL_K = 5  # Number of chunks to retrieve

# -------------------------------
# Step 0: Download Gray's Anatomy if not present
# -------------------------------
@st.cache_resource
def download_text():
    filepath = "grays_anatomy.txt"
    if not os.path.exists(filepath):
        with st.spinner("Downloading Gray's Anatomy text..."):
            url = "https://archive.org/stream/anatomyofhumanbo1918gray/anatomyofhumanbo1918gray_djvu.txt"
            response = requests.get(url)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
    return filepath

# -------------------------------
# Step 1: Load and split text with better chunking
# -------------------------------
@st.cache_resource
def load_and_split_documents():
    filepath = download_text()
    
    with st.spinner("Loading document..."):
        loader = TextLoader(filepath, encoding="utf-8")
        documents = loader.load()
    
    with st.spinner("Splitting text into optimized chunks..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],  # Better split points
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
    
    return chunks

# -------------------------------
# Step 2: Load or build FAISS vectorstore with better embeddings
# -------------------------------
@st.cache_resource
def load_vectorstore():
    chunks = load_and_split_documents()
    
    with st.spinner(f"Loading embeddings model ({EMBEDDING_MODEL})..."):
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Improves retrieval
        )
    
    vectorstore_path = "grays_anatomy_vectorstore_v2"  # New version with better embeddings
    
    if os.path.exists(vectorstore_path):
        with st.spinner("Loading vector store..."):
            vectorstore = FAISS.load_local(
                vectorstore_path,
                hf_embeddings,
                allow_dangerous_deserialization=True
            )
    else:
        with st.spinner("Creating vector store (this will take a few minutes)..."):
            vectorstore = FAISS.from_documents(chunks, hf_embeddings)
            vectorstore.save_local(vectorstore_path)
    
    return vectorstore

# -------------------------------
# Step 3: Initialize QA chain with LLM and custom prompt
# -------------------------------
@st.cache_resource
def initialize_qa_chain():
    vectorstore = load_vectorstore()
    
    # Check for API token
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        st.error("⚠️ HUGGINGFACEHUB_API_TOKEN not found in environment variables!")
        st.info("Get a free token at: https://huggingface.co/settings/tokens")
        st.stop()
    
    with st.spinner(f"Initializing LLM ({LLM_MODEL})..."):
        llm = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            temperature=0.1,  # Low temperature for factual answers
            max_new_tokens=500,
            top_k=50,
            huggingfacehub_api_token=api_token
        )
    
    # Custom prompt for better medical answers
    template = """You are an expert anatomist with deep knowledge of Gray's Anatomy. Use the following excerpts from Gray's Anatomy (1918 edition) to answer the question accurately and educationally.

Context from Gray's Anatomy:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate, and educational answer based on the context above
- Cite specific anatomical structures, regions, or systems mentioned in the text
- Use proper medical terminology while remaining accessible
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Structure your answer logically (e.g., definition, location, structure, function, clinical relevance)

Answer:"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# -------------------------------
# Step 4: Streamlit UI
# -------------------------------
st.set_page_config(page_title="Gray's Anatomy AI", layout="wide", page_icon="🧠")

# Header
st.title("🧠 Gray's Anatomy AI Assistant")
st.markdown("""
Ask questions about human anatomy based on the complete text of **Gray's Anatomy (1918 edition)**.
This AI uses advanced retrieval and language models to provide accurate, educational answers.
""")

# Initialize the QA chain
try:
    qa_chain = initialize_qa_chain()
    st.success("✅ System ready! Ask your anatomy questions below.")
except Exception as e:
    st.error(f"❌ Error initializing system: {str(e)}")
    st.stop()

# Sidebar with info and examples
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Powered by:**
    - 🤖 Mixtral-8x7B (LLM)
    - 🔍 BGE-Large embeddings
    - 📚 Complete Gray's Anatomy text
    - ⚡ FAISS vector search
    """)
    
    st.markdown("---")
    
    st.header("💡 Example Questions")
    examples = [
        "What are the main bones of the skull?",
        "Describe the structure of the heart",
        "What muscles are involved in breathing?",
        "Explain the layers of the skin",
        "What is the function of the cerebellum?",
        "Describe the structure of a long bone",
        "What are the parts of the digestive system?",
        "Explain the vertebral column",
        "What are the chambers of the heart?",
        "Describe the brachial plexus"
    ]
    
    for ex in examples:
        if st.sidebar.button(ex, key=f"example_{ex[:20]}"):
            st.session_state.clicked_example = ex

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main input area
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input(
        "Ask a question about anatomy:",
        value=st.session_state.get("clicked_example", ""),
        key="user_input"
    )
    if "clicked_example" in st.session_state:
        del st.session_state.clicked_example

with col2:
    st.write("")  # Spacing
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

# Clear button
if st.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Process question
if (ask_button or user_question) and user_question.strip():
    with st.spinner("🤔 Thinking..."):
        try:
            result = qa_chain.invoke({"query": user_question})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer,
                "num_sources": len(source_docs)
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display chat history (most recent first)
st.markdown("---")
if st.session_state.chat_history:
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"### ❓ {chat['question']}")
            st.markdown(chat['answer'])
            st.caption(f"📚 Answer based on {chat['num_sources']} relevant sections from Gray's Anatomy")
            st.markdown("---")
else:
    st.info("👆 Ask a question above or click an example from the sidebar to get started!")

# Footer
st.markdown("---")
st.caption("""
⚠️ **Note:** This AI uses the 1918 edition of Gray's Anatomy. While anatomical fundamentals remain accurate,
medical terminology and some concepts may reflect the knowledge of that era. Always consult current medical 
resources for clinical decisions.
""")