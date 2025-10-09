import os
import requests
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------------------
# Step 0: Download Gray's Anatomy if not present
# -------------------------------
filepath = "grays_anatomy.txt"
if not os.path.exists(filepath):
    url = "https://archive.org/stream/anatomyofhumanbo1918gray/anatomyofhumanbo1918gray_djvu.txt"
    response = requests.get(url)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("✓ Gray's Anatomy downloaded!")
else:
    print("✓ Gray's Anatomy already exists!")

# -------------------------------
# Step 1: Load and split text
# -------------------------------
loader = TextLoader(filepath, encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"✓ Text split into {len(chunks)} chunks")

# -------------------------------
# Step 2: Load or build FAISS vectorstore
# -------------------------------
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("grays_anatomy_vectorstore"):
    vectorstore = FAISS.load_local(
        "grays_anatomy_vectorstore",
        hf_embeddings,
        allow_dangerous_deserialization=True
    )
    print("✓ Loaded existing FAISS vector store")
else:
    vectorstore = FAISS.from_documents(chunks, hf_embeddings)
    vectorstore.save_local("grays_anatomy_vectorstore")
    print("✓ Created and saved FAISS vector store")

# -------------------------------
# Step 3: Streamlit UI
# -------------------------------
st.set_page_config(page_title="Gray's Anatomy AI", layout="wide")
st.title("🧠 Gray's Anatomy FAQ Agent")
st.markdown("Ask questions about human anatomy based on Gray's Anatomy (1918).")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input question
user_question = st.text_input("Ask a question about anatomy:")

# Submit button
if st.button("Ask") and user_question.strip():
    docs = vectorstore.similarity_search(user_question, k=4)
    answer = "\n\n".join([d.page_content for d in docs])[:2000]  # truncate for display
    st.session_state.chat_history.append((user_question, answer))

# Clear conversation
if st.button("Clear Conversation"):
    st.session_state.chat_history = []

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown("---")

# Optional: sidebar examples
st.sidebar.header("💡 Example Questions")
examples = [
    "What are the main bones of the skull?",
    "Describe the structure of the heart",
    "What muscles are involved in breathing?",
    "Explain the layers of the skin",
    "What is the function of the cerebellum?",
    "Describe the structure of a long bone",
    "What are the parts of the digestive system?",
    "Explain the vertebral column"
]
for ex in examples:
    if st.sidebar.button(ex):
        st.session_state.chat_history.append((ex, "\n\n".join([d.page_content for d in vectorstore.similarity_search(ex, k=4)])[:2000]))
