import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

# Initialize Chroma client with local persistence
chroma_client = chromadb.PersistentClient(
    path="./chroma/handson1"  # directory where Chroma data is saved
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # using cosine similarity
)

# Load your embedding model (SentenceTransformer)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Raw documents as list of strings
raw_documents = [
    "Retrieval-Augmented Generation improves the factual accuracy of LLMs.",
    "Vector databases store embeddings and enable fast similarity search.",
    "Chunking is the process of splitting long documents into smaller pieces."
]

# Convert raw strings to LangChain Document objects
documents = [Document(page_content=doc) for doc in raw_documents]

# Initialize splitter with desired chunk size and overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=5,
    separators=["\n", "\n\n", ". ", "! ", "? "]
)

# Split documents into smaller chunks
split_docs = splitter.split_documents(documents)

# Print first two split chunks
#print(split_docs[:1])

# Optionally: embed each split chunk using SentenceTransformer
embeddings = [embed_model.encode(doc.page_content) for doc in split_docs]

# Insert documents and embeddings into Chroma collection if desired
for idx, (doc, embedding) in enumerate(zip(split_docs, embeddings)):
    collection.add(
        ids=[f"doc_{idx}"],  # ChromaDB requires unique IDs
        documents=[doc.page_content],
        embeddings=[embedding.tolist()]  # Convert numpy array to list
    )


openai_client=ChatOpenAI(
model_name="gpt-4o-mini",max_tokens=1000,temperature=1)

def build_context(results):
    return "\n\n".join(results['documents'][0])

def answer_with_rag(query):
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)

    context = build_context(results)

    prompt = f"""
You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

    response = openai_client.invoke(
        input=prompt
    )

    return response.content

print(answer_with_rag("What is RAG?"))

