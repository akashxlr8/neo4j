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
        # Call the agent with verbose=True
        response = generate_response(message, show_intermediate_steps=True)
        
        print("Response received in bot.py:", response)  # Debug print
        
        # Create an expander for showing the intermediate steps
        if isinstance(response, dict) and 'intermediate_steps' in response:
            print("Found intermediate steps:", response['intermediate_steps'])  # Debug print
            tool_count = 0
            current_tool_steps = []
            
            for step in response['intermediate_steps']:
                print("Processing step:", step)  # Debug print
                # If we see an Action, it's the start of a new tool use
                if 'Action' in step:
                    print("Found Action in step")  # Debug print
                    # If we have previous steps, show them in an expander
                    if current_tool_steps:
                        tool_count += 1
                        print(f"Showing Tool Use #{tool_count}")  # Debug print
                        
                        # Get the tool name from the AgentAction object
                        tool_name = current_tool_steps[0][0].tool if isinstance(current_tool_steps[0], tuple) and hasattr(current_tool_steps[0][0], 'tool') else 'Unknown'
                        
                        with st.expander(f"🔧️ Tool: {tool_name} #{tool_count}", expanded=False):
                            for prev_step in current_tool_steps:
                                col1, col2 = st.columns([1, 4])
                                # Extract information from AgentAction
                                if isinstance(prev_step, tuple) and hasattr(prev_step[0], 'tool'):
                                    thought_log = prev_step[0].log
                                    action = prev_step[0].tool
                                    action_input = prev_step[0].tool_input
                                    observation = str(prev_step[1])
                                    
                                    # Display the components
                                    if thought_log:
                                        col1.markdown("**🤔 Thought:**")
                                        col2.markdown(thought_log)
                                    col1.markdown("**⚡ Action:**")
                                    col2.markdown(action)
                                    col1.markdown("**📥 Input:**")
                                    col2.markdown(f"```json\n{action_input}\n```")
                                    col1.markdown("**👁️ Observation:**")
                                    col2.markdown(f"```\n{observation}\n```")
                        current_tool_steps = []
                current_tool_steps.append(step)
            
            # Show any remaining steps
            if current_tool_steps:
                tool_count += 1
                print(f"Showing Tool Use #{tool_count}")  # Debug print
                # Get the tool name from the first step that has an Action
                tool_name = next((step.get('Action', 'Tool Use') for step in current_tool_steps 
                                 if isinstance(step, dict) and 'Action' in step), 'Tool Use')
                
                # Check if step is a dictionary before calling get()
                action = step.get('Action', 'Unknown') if isinstance(step, dict) else 'Unknown'
                
                # Modified expander label to show both tool name and tool used
                with st.expander(f"🔧️ Tool: {tool_name} | Used: {action} #{tool_count}", expanded=False):
                    for prev_step in current_tool_steps:
                        col1, col2 = st.columns([1, 4])
                        if isinstance(prev_step, dict):  # Only process dictionary steps
                            for key, (emoji, label) in {
                                'Thought': ('🤔', 'Thought'),
                                'Action': ('⚡', 'Action'),
                                'Action Input': ('📥', 'Input'),
                                'Observation': ('👁️', 'Observation')
                            }.items():
                                if key in prev_step:
                                    col1.markdown(f"**{emoji} {label}:**")
                                    if key == 'Action Input':
                                        col2.markdown(f"```json\n{prev_step[key]}\n```")
                                    elif key == 'Observation':
                                        col2.markdown(f"```\n{prev_step[key]}\n```")
                                    else:
                                        col2.markdown(prev_step[key])
        else:
            print("No intermediate steps found in response")  # Debug print
        
        # Write the final response outside the expanders
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