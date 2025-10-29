#!/bin/bash

# Setup script for Slice 4 - Knowledge Graph
# Run this to install all dependencies for the knowledge graph feature

echo "============================================================"
echo "Gray's Anatomy Agent - Slice 4 Setup"
echo "Installing Knowledge Graph Dependencies"
echo "============================================================"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip not found. Please install pip first."
    exit 1
fi

# Use pip3 if pip is not available
PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

echo "Using: $PIP_CMD"
echo ""

# Install all requirements
echo "Step 1/3: Installing Python packages..."
echo "------------------------------------------------------------"
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install Python packages"
    exit 1
fi

echo ""
echo "✓ Python packages installed successfully"
echo ""

# Download spaCy model
echo "Step 2/3: Downloading spaCy language model..."
echo "------------------------------------------------------------"

PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

$PYTHON_CMD -m spacy download en_core_web_sm

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to download spaCy model"
    echo "Try manually: python -m spacy download en_core_web_sm"
    exit 1
fi

echo ""
echo "✓ spaCy model downloaded successfully"
echo ""

# Run tests
echo "Step 3/3: Running knowledge graph tests..."
echo "------------------------------------------------------------"
$PYTHON_CMD test_graph.py

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Some tests failed. Check output above."
    echo ""
else
    echo ""
    echo "✓ All tests passed!"
    echo ""
fi

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Set up your API keys in .env file:"
echo "       OPENAI_API_KEY=sk-..."
echo "       ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  2. Run the application:"
echo "       streamlit run app.py"
echo ""
echo "  3. In the UI, select a knowledge graph mode:"
echo "       - None (default, standard RAG)"
echo "       - Entity Graph (entity-based traversal)"
echo "       - Community Graph (hierarchical communities)"
echo ""
echo "First query with graph enabled will build the graph (~2-5 min)"
echo "Subsequent queries use cached graph (instant)"
echo ""
echo "See KNOWLEDGE_GRAPH_SETUP.md for full documentation"
echo ""
