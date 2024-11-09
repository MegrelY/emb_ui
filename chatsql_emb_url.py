import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from pgvector.sqlalchemy import Vector
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP)
    url = Column(String)
    title = Column(String)
    headings = Column(JSON)
    paragraphs = Column(JSON)
    lists = Column(JSON)
    links = Column(JSON)
    embedding = Column(Vector(1536))

# Function to generate embeddings
def generate_query_embedding(query, model="text-embedding-3-small"):
    """Generate an embedding for a search query."""
    try:
        response = client.embeddings.create(
            input=[query],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_documents(session: Session, query_embedding, top_k=3):
    """Find documents in the database that are most similar to the query embedding."""
    documents = session.query(
        Document.id, Document.title, Document.headings, Document.paragraphs, Document.url, Document.embedding, Document.lists 
    ).all()

    # Calculate cosine similarity between query and document embeddings
    similarities = []
    for doc in documents:
        if doc.embedding is not None:
            similarity = cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc.id, doc.title, doc.headings, doc.paragraphs, doc.url, similarity))

    # Sort by similarity and retrieve the top_k results
    similarities = sorted(similarities, key=lambda x: x[-1], reverse=True)[:top_k]
    return similarities

def get_response_with_context(user_input, session):
    """Generate response from OpenAI with relevant document context."""
    query_embedding = generate_query_embedding(user_input)
    if query_embedding is None:
        return "Failed to generate embedding for the query.", []

    # Find similar documents based on embeddings
    similar_docs = find_similar_documents(session, query_embedding, top_k=3)
    
    # Create context from the most similar documents
    context = "\n\n".join(
        [
            f"Title: {doc[1]}\nHeadings: {doc[2]}\nParagraphs: {doc[3]}\nURL: {doc[4]}"
            for doc in similar_docs
        ]
    )

    messages = [
        {"role": "system", "content": f"Relevant context: {context}"},
        {"role": "user", "content": user_input}
    ]

    # Get the assistant's response using OpenAI API
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        assistant_response = completion.choices[0].message.content
        reference_urls = [doc[4] for doc in similar_docs]
        return assistant_response, reference_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request.", []
