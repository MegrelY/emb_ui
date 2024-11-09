from flask import Flask, render_template, request, jsonify, session
from sqlalchemy.orm import Session
from chatsql_emb_url import SessionLocal, get_response_with_context
from flask_session import Session as FlaskSession

# Initialize Flask app and session
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
FlaskSession(app)

@app.route('/')
def index():
    """Render the main UI for the chatbot."""
    session['conversation_history'] = []  # Clear the conversation history on each new session
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle the chat request from the UI and return the updated chatbot conversation history and reference URLs."""
    user_input = request.json.get('message')
    
    # Initialize conversation history if it doesn't exist in session
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Start a database session
    db_session = SessionLocal()
    
    # Get response with context
    assistant_response, reference_urls = get_response_with_context(user_input, db_session)
    db_session.close()

    # Append user input and response to the session conversation history
    session['conversation_history'].append({
        "user": user_input,
        "assistant": assistant_response,
        "urls": reference_urls
    })

    # Return the entire conversation history
    return jsonify({"conversation_history": session['conversation_history']})

if __name__ == "__main__":
    app.run(debug=True)
