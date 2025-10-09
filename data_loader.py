"""
Data loading and preprocessing
"""
import os
import requests
from config import GRAYS_ANATOMY_URL, TEXT_FILE, CHUNK_SIZE, CHUNK_OVERLAP


def download_text():
    """Download Gray's Anatomy text if not present"""
    if os.path.exists(TEXT_FILE):
        print(f"✓ Found {TEXT_FILE}")
        return TEXT_FILE
    
    print(f"Downloading from {GRAYS_ANATOMY_URL}...")
    response = requests.get(GRAYS_ANATOMY_URL, timeout=60)
    
    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print(f"✓ Downloaded {len(response.text):,} characters")
    return TEXT_FILE


def load_text():
    """Load text from file"""
    download_text()
    
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text


def chunk_text(text):
    """Split text into overlapping chunks"""
    chunks = []
    
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        if len(chunk) > 100:  # Skip tiny chunks
            chunks.append(chunk)
    
    print(f"✓ Created {len(chunks):,} chunks")
    return chunks


def load_and_chunk():
    """Main function to load and chunk text"""
    text = load_text()
    chunks = chunk_text(text)
    return chunks