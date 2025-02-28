import streamlit as st
from utils import write_message
from agent import generate_response
from logger import log
import json

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
        
        # Create an expander for the tool usage
        if isinstance(response, dict) and 'intermediate_steps' in response and response['intermediate_steps']:
            with st.expander("View thinking process", expanded=False):
                for step in response['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        # Extract the tool and observation
                        tool_action = step[0]
                        tool_observation = step[1]
                        
                        # Display tool action in bold
                        st.markdown(f"**Tool:** {tool_action.tool}")
                        st.markdown(f"**Input:** {tool_action.tool_input}")
                        
                        # Format the observation for better display
                        st.markdown("**Result:**")
                        if isinstance(tool_observation, dict) and 'result' in tool_observation:
                            if isinstance(tool_observation['result'], list) and len(tool_observation['result']) > 0:
                                st.table(tool_observation['result'])
                            else:
                                st.code(json.dumps(tool_observation['result'], indent=2), language="json")
                        else:
                            st.write(tool_observation)
                        
                        # Add a separator between steps
                        st.markdown("---")
        
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
if prompt := st.chat_input("What would you like to know about Northwind Store?"):
    logger.info(f"Received user input: {prompt}")
    # Display user message in chat message container
    write_message('user', prompt)
    # Generate a response
    handle_submit(prompt)