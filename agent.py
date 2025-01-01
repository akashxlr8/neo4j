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

from tools.cypher import cypher_qa
from tools.vector import retriever

logger = log('agent')

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a company internal service expert chatbot providing information about products, orders, and customers in the Northwind database. "),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat about the company and its products and services not covered by other tools",
        func=movie_chat.invoke,
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
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a store expert providing information about products, orders, and customers in the Northwind database.
You can use semantic search to find similar products and provide detailed product information.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to products, orders, or customers.

Previous conversation context:
{chat_history}

Remember to maintain context from the previous messages when answering follow-up questions.
If a question seems incomplete, try to understand it in the context of previous messages.

For each step, you should:
1. Think about whether you need to use a tool
2. Choose the appropriate tool
3. Provide the input to the tool
4. Use the observation to form your final answer

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
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input, show_intermediate_steps=False):
    logger.info(f"Starting response generation for input: {user_input}")
    
    try:
        logger.info("Invoking chat agent")
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}},
        )
        logger.debug(f"Raw chat agent response: {response}")
        
        # Log agent thoughts and actions
        if isinstance(response, dict) and 'result' in response:
            result = response['result']
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    logger.info(f"Agent Thought: {step.get('thought', '')}")
                    logger.info(f"Agent Action: {step.get('action', '')}")
                    logger.info(f"Agent Action Input: {step.get('action_input', '')}")
                    logger.info(f"Agent Observation: {step.get('observation', '')}")
        
        # Extract the actual response
        if isinstance(response, dict) and 'result' in response:
            logger.info("Extracting result from response")
            response = response['result']
        
        # Get final output
        if isinstance(response, dict):
            output = response.get('output', '')
            if not output and 'response' in response:
                output = response['response']
            steps = response.get('intermediate_steps', [])
            logger.info(f"Extracted output and {len(steps)} intermediate steps")
        else:
            output = str(response)
            steps = []
            logger.info("Converted response to string output")

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
