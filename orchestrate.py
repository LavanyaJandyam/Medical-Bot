# rag_gemini.py

import requests
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#from support import getdata  # Assuming getdata is in support.py
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Load or create vector DB
import chromadb

def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client.get_collection("medical_chatbot")
    except chromadb.errors.NotFoundError:
        collection = client.create_collection("medical_chatbot")
    
    return collection


# Get top K relevant documents from Chroma

def get_top_k_context(query: str, top_k: int = 3) -> List[str]:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedding_model.encode(query).tolist()
    collection= get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_texts = results["documents"][0]  # Top-k documents
    return retrieved_texts

# Query Gemini API with context + question

def query_gemini(question: str, context: List[str]) -> str:
    context_text = "\n\n".join(context)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

# Entry function for RAG + Gemini

def ask_with_rag(question: str, top_k: int = 3):
    context = get_top_k_context(question, top_k)
    answer = query_gemini(question, context)
    return answer

a = ask_with_rag("what is heart attack")
print(a)