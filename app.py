from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from sqlalchemy.orm import Session
from chatsql_emb_url import SessionLocal, find_similar_documents, generate_query_embedding

# Initialize Flask app
app = Flask(__name__)

# Load OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def index():
    """Render the main UI for the chatbot."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle the chat request from the UI and return the chatbot response and reference URL."""
    user_input = request.json.get('message')
    
    # Generate query embedding
    session = SessionLocal()
    query_embedding = generate_query_embedding(user_input)
    
    # Find similar documents to get reference URLs
    similar_docs = find_similar_documents(session, query_embedding, top_k=3)
    session.close()

    # Format the context for the chatbot with retrieved documents
    context = "\n\n".join([
        f"Title: {doc[1]}\nHeadings: {doc[2]}\nParagraphs: {doc[3]}\nURL: {doc[4]}"
        for doc in similar_docs
    ])
    
    # Send context to OpenAI API to get response
    messages = [
        {"role": "system", "content": f"Relevant context: {context}"},
        {"role": "user", "content": user_input}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        assistant_response = completion.choices[0].message.content

        # Prepare URL list to send back
        reference_urls = [doc[4] for doc in similar_docs]

        return jsonify({"response": assistant_response, "urls": reference_urls})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
