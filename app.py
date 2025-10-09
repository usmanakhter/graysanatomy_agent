import os
import requests
import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------------------
# Step 0: Download Gray's Anatomy
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
# Step 1: Load & split text
# -------------------------------
loader = TextLoader(filepath, encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"✓ Text split into {len(chunks)} chunks")

# -------------------------------
# Step 2: Build or load FAISS vector store
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
# Step 3: Define embeddings-only QA
# -------------------------------
def answer_question_no_llm(question, k=4):
    docs = vectorstore.similarity_search(question, k=k)
    answer_text = "\n\n".join([d.page_content for d in docs])
    return answer_text[:2000]  # truncate for display

def chat_fn(question, chat_history):
    answer = answer_question_no_llm(question)
    chat_history.append((question, answer))
    return "", chat_history

# -------------------------------
# Step 4: Build Gradio UI
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Gray's Anatomy AI") as demo:
    gr.Markdown(
        "# 🧠 Gray's Anatomy FAQ Agent\nAsk questions about human anatomy based on Gray's Anatomy (1918)."
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            question_input = gr.Textbox(
                placeholder="Ask about anatomy...",
                label="Your Question",
                lines=2
            )
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear Conversation")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Questions")
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
                btn = gr.Button(ex, size="sm")
                btn.click(lambda x=ex: x, outputs=question_input)

    submit_btn.click(chat_fn, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    question_input.submit(chat_fn, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    clear_btn.click(lambda: ("", []), None, chatbot)

# -------------------------------
# Step 5: Launch
# -------------------------------
if __name__ == "__main__":
    demo.launch()
