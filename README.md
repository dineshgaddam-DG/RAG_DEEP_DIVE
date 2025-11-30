# RAG Deep Dive

A hands-on implementation of **Retrieval-Augmented Generation (RAG)** using ChromaDB, SentenceTransformers, and OpenAI's GPT models.

## Overview

This project demonstrates how to build a RAG system that:
- Chunks documents into smaller pieces for better retrieval
- Generates embeddings using SentenceTransformers
- Stores embeddings in ChromaDB vector database
- Retrieves relevant context based on user queries
- Generates accurate answers using OpenAI's GPT-4o-mini

## Features

- **Document Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` for intelligent text splitting
- **Vector Embeddings**: Leverages `all-MiniLM-L6-v2` model for generating embeddings
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: OpenAI GPT-4o-mini for generating contextual answers

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG-DeepDive
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv312
   source venv312/bin/activate  # On Windows: venv312\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the main script:
```bash
python handson1.py
```

The script will:
1. Load and chunk sample documents
2. Generate embeddings and store them in ChromaDB
3. Answer the query: "What is RAG?"

### Customizing Queries

Modify the query at the bottom of `handson1.py`:
```python
print(answer_with_rag("Your question here"))
```

## Project Structure

```
RAG-DeepDive/
├── handson1.py          # Main RAG implementation
├── overview.ipynb       # Jupyter notebook with examples
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore rules
├── requirements.txt     # Python dependencies
└── chroma/             # ChromaDB data directory (not in git)
```

## How It Works

1. **Document Processing**: Raw documents are converted to LangChain `Document` objects
2. **Chunking**: Documents are split into smaller chunks with overlap for better context
3. **Embedding**: Each chunk is converted to a vector embedding
4. **Storage**: Embeddings are stored in ChromaDB with unique IDs
5. **Retrieval**: User queries are embedded and similar chunks are retrieved
6. **Generation**: Retrieved context is passed to GPT-4o-mini to generate answers

## Dependencies

- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `langchain-core` - Document handling
- `langchain-text-splitters` - Text chunking
- `langchain-openai` - OpenAI integration
- `python-dotenv` - Environment variable management

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
