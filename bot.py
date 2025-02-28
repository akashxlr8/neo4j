import streamlit as st
from utils import write_message
from agent import generate_response
from logger import log

logger = log('bot')

# Page Config
st.set_page_config("Neo4j Chatbot", page_icon=":shopping_cart:")

# Set up Session State
if "messages" not in st.session_state:
    logger.info("Initializing new chat session")
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm your Neo4j Chatbot! How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """Submit handler that processes user input and generates response"""
    
    logger.info("Processing submit request")
    
    # Handle the response
    with st.spinner('Thinking...'):
        # Create a container for the callback visualization
        callback_container = st.container()
        
        # Call the agent with streamlit callback
        response = generate_response(message, show_intermediate_steps=True, st_container=callback_container)
        
        print("Response received in bot.py:", response)  # Debug print
        
        # Write the final response
        if isinstance(response, dict) and 'output' in response:
            cleaned_output = response['output'].strip('`').strip()
            write_message('assistant', cleaned_output)
        elif isinstance(response, str):
            cleaned_output = response.strip('`').strip()
            write_message('assistant', cleaned_output)

# Display messages in Session State
for message in st.session_state.messages:
    if message.get('content', '').strip():  # Only display non-empty messages
        write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What would you like to know about movies?"):
    logger.info(f"Received user input: {prompt}")
    # Display user message in chat message container
    write_message('user', prompt)
    # Generate a response
    handle_submit(prompt)