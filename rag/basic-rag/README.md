# Basic RAG Implementation

A simple implementation of Retrieval-Augmented Generation using ChromaDB and OpenAI.

## Overview

This example demonstrates the fundamental concepts of RAG:
- Document chunking with LangChain
- Embedding generation with SentenceTransformers
- Vector storage with ChromaDB
- Context retrieval and answer generation with OpenAI GPT-4o-mini

## Files

- `handson1.py` - Main implementation script

## Usage

```bash
# From the project root
cd rag/basic-rag
python handson1.py
```

## How It Works

1. **Documents**: Sample documents about RAG concepts
2. **Chunking**: Splits documents into smaller pieces (25 chars with 5 char overlap)
3. **Embeddings**: Uses `all-MiniLM-L6-v2` model
4. **Storage**: Stores in ChromaDB with cosine similarity
5. **Query**: Retrieves top 3 relevant chunks
6. **Generation**: GPT-4o-mini generates answer from context

## Example Output

```
Query: "What is RAG?"
Answer: RAG, or Retrieval-Augmented Generation, is a method that enhances 
the factual accuracy of language models by incorporating retrieved information 
from external sources during the generation process.
```
