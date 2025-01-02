from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id
from langchain.schema.runnable import RunnableMap
from logger import log
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain_core.runnables import RunnablePassthrough

from tools.cypher import cypher_qa
from tools.vector import retriever

logger = log('agent')

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a company internal service expert chatbot providing information about products, orders, and customers in the Northwind database. "),
        ("human", "{input}"),
    ]
)

general_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat about the company and its products and services not covered by other tools",
        func=general_chat.invoke,
    ),
    Tool.from_function(
        name="Northwind information",
        description="Provide information about products, orders, and customers in the Northwind database using Cypher",
        func=cypher_qa
    ),
    Tool.from_function(
        name="Product Search",
        description="For finding similar products or searching product descriptions semantically",
        func=lambda q: [{"page_content": doc.page_content, "metadata": doc.metadata} 
                       for doc in retriever.get_relevant_documents(q)]
    )
]

def get_memory(session_id):
    memory = Neo4jChatMessageHistory(session_id=session_id, graph=graph)
    # Get all messages and keep only last 3
    messages = memory.messages[-3:] if memory.messages else []
    # Clear and add back only last 3 with simplified format
    memory.clear()
    for msg in messages:
        # Store only the content without additional metadata
        simplified_msg = msg.__class__(content=msg.content)
        memory.add_message(simplified_msg)
    return memory

agent_prompt = PromptTemplate.from_template("""
You are a store expert providing information about products, orders, and customers in the Northwind database.
Do not engage in general conversation or provide information that is not related to the store.
IMPORTANT TOOL SELECTION GUIDELINES:
1. Use "Product Search" tool for:
   - Finding products by description
   - Semantic similarity searches
   - Questions about product details or ingredients
   - Any natural language queries about products

2. Use "Northwind information" tool for:
   - Exact counts or numerical queries
   - Customer information
   - Order details
   - Specific product lookups by ID or exact name
   - Relationship queries (e.g., which supplier supplies what)

3. Use "General Chat" tool for:
   - General conversation
   - Questions not requiring specific data lookup

Previous conversation history:
{chat_history}

Remember to maintain context from the previous messages when answering follow-up questions.
If a question seems incomplete, try to understand it in the context of previous messages.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# Convert AgentExecutor to Runnable
runnable_agent = RunnablePassthrough() | agent_executor

chat_agent = RunnableWithMessageHistory(
    runnable_agent,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

def generate_response(user_input, show_intermediate_steps=False):
    logger.info(f"Starting response generation for input: {user_input}")
    
    try:
        logger.info("Invoking chat agent")
        logger.info(f"Chat history: {get_memory(get_session_id()).messages}")
        response = chat_agent.invoke(
            {
                "input": user_input,
                "chat_history": get_memory(get_session_id()).messages  # Pass messages directly
            },
            {"configurable": {"session_id": get_session_id()}},
        )
        
        logger.debug(f"Raw chat agent response: {response}")
        
        # Log agent thoughts and actions
        if isinstance(response, dict) and 'intermediate_steps' in response:
            logger.info("Tools Used:")
            for step in response['intermediate_steps']:
                if isinstance(step, dict) and 'Action' in step:
                    logger.info(f"🔧 Tool Used: {step['Action']}")
                    logger.info(f"📥 Input: {step.get('Action Input', '')}")
                    logger.info(f"👁️ Observation: {step.get('Observation', '')}\n")
        
        # Extract the actual response
        output = response.get('output', '')
        steps = response.get('intermediate_steps', [])
        logger.info(f"Extracted output and {len(steps)} intermediate steps")
        
        logger.info(f"Final response: {output}")
        return {
            'output': output,
            'intermediate_steps': steps if show_intermediate_steps else []
        }
    
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        return {
            'output': "I apologize, but I encountered an error while processing your request.",
            'intermediate_steps': []
        }
