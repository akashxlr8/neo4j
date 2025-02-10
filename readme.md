<div>
    <a href="https://www.loom.com/share/109aa9e9768e4929ad4fb559831f47fe">
      <p>Neo4j Chatbot · Streamlit - Google Chrome - 10 February 2025 - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/109aa9e9768e4929ad4fb559831f47fe">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/109aa9e9768e4929ad4fb559831f47fe-5dee3d221d60b96c-full-play.gif">
    </a>
  </div>

# Chatbot Application Documentation

This document provides a comprehensive overview of the chatbot application that utilizes Neo4j Graph Database for data storage and retrieval. The application is built using the Northwind dataset, which is a relational and open-source database. The chatbot interacts with the database stored in graph form, enabling advanced querying capabilities. The chatbot can be use to perform natural language queries over the database


---

### Python Code Snippet (`graph.py`)

This snippet is responsible for establishing a connection to a Neo4j graph database using the `langchain_neo4j` library.

```python
# Connect to Neo4j
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
    database="neo4j"
)
```

- **Import Statement**: The code imports `Neo4jGraph` from the `langchain_neo4j` library, which is used to interact with a Neo4j database.
- **Graph Initialization**: 
  - `Neo4jGraph` is instantiated with several parameters:
    - `url`: The URI of the Neo4j database, retrieved from Streamlit's secrets management (`st.secrets["NEO4J_URI"]`).
    - `username` and `password`: Credentials for accessing the database, also stored securely in Streamlit's secrets.
    - `database`: Specifies the database name, which is "neo4j" in this case.

This setup allows the application to securely connect to a Neo4j database using credentials stored in a secure manner.

---


### Product Information

The following table provides detailed information about the first 5 products in the Northwind dataset:

| productID | productName                     | supplierID | categoryID | quantityPerUnit     | unitPrice | unitsInStock | unitsOnOrder | reorderLevel | discontinued |
|-----------|---------------------------------|------------|------------|---------------------|-----------|--------------|--------------|--------------|--------------|
| 1         | Chai                            | 1          | 1          | 10 boxes x 20 bags  | 18.00     | 39           | 0            | 10           | 0            |
| 2         | Chang                           | 1          | 1          | 24 - 12 oz bottles  | 19.00     | 17           | 40           | 25           | 0            |
| 3         | Aniseed Syrup                   | 1          | 2          | 12 - 550 ml bottles | 10.00     | 13           | 70           | 25           | 0            |
| 4         | Chef Anton's Cajun Seasoning    | 2          | 2          | 48 - 6 oz jars      | 22.00     | 53           | 0            | 0            | 0            |
| 5         | Chef Anton's Gumbo Mix          | 2          | 2          | 36 boxes            | 21.35     | 0            | 0            | 0            | 1            |


### Cypher Code Snippet (`neo4j-data-import.cypher`)

This snippet is a sample Cypher script used to import data from a CSV file into the Neo4j database.

```cypher
LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/products.csv" AS row
MERGE (n:Product {productID:row.productID})
SET n += row,
n.unitPrice = toFloat(row.unitPrice),
n.unitsInStock = toInteger(row.unitsInStock), n.unitsOnOrder = toInteger(row.unitsOnOrder),
n.reorderLevel = toInteger(row.reorderLevel), n.discontinued = (row.discontinued <> "0");
```

- **LOAD CSV**: This command loads data from a CSV file located at the specified URL. The `WITH HEADERS` option indicates that the first row of the CSV file contains column headers.
- **MERGE**: This command ensures that a node with a specific `productID` exists. If it doesn't, a new node is created. This is similar to an "upsert" operation.
- **SET**: This command updates the properties of the node:
  - `n += row`: This syntax updates the node `n` with all properties from the CSV row.
  - `n.unitPrice`, `n.unitsInStock`, `n.unitsOnOrder`, `n.reorderLevel`: These properties are explicitly set with type conversions to ensure they are stored as the correct data types (e.g., `toFloat`, `toInteger`).
  - `n.discontinued`: This property is set to a boolean value based on whether the `discontinued` field in the CSV is not equal to "0".

This script effectively imports product data from a CSV file into the Neo4j database, ensuring that each product is represented as a node with the appropriate properties.

Like this cypher script was used to load products.csv into the db, we have several others defined in the `neo4j-data-import.cypher` file that help with the loading of different relevant csvs into the database.

---

## Product Descriptions and Semantic Search

A new property, `description`, has been added to the products in `products.csv`. This property is used for semantic searching, allowing users to search for products based on characteristics.
```txt
1. Chai: Chai is a popular spiced tea beverage made with black tea leaves, milk, and a blend of spices such as cinnamon, ginger, and cardamom. Our Chai product is sourced from the tea gardens of Assam and Darjeeli...
2. Chang: Chang is a type of Chinese beer brewed with a combination of ingredients such as barley, rice, and hops. Our Chang product is sourced from a reputable brewery in China, where the art of brewing has been perfected over centuries. ...
3. Aniseed Syrup: Aniseed Syrup is a sweet and fragrant liquid extract made from the seeds of the anise plant, which is native to the Mediterranean region. Our Aniseed Syrup product is sourced from a ...
4. Chef Anton's Cajun Seasoning: Chef Anton's Cajun Seasoning is a unique blend of spices and herbs that originated in the southern United States, particularly in the state of Louisiana. Our Chef Anton's Cajun Seasoning ...
5. Chef Anton's Gumbo Mix: Chef Anton's Gumbo Mix is a convenient and easy-to-use mix of ingredients that can be used to make a delicious and authentic gumbo. Our Chef Anton's Gumbo Mix product is sourced from a reputable manufacturer in the United ...

```
### Embedding Conversion
The `embed.py` script is used to convert text descriptions into vector embeddings, which are then loaded into the graph database.

```python:embed.py

def generate_embeddings(tx, node_label, descriptions):
    query = """
    MATCH (n:Product)
    WHERE n.productID IS NOT NULL
    WITH n, 
    CASE WHEN n:Product = true 
         THEN $descriptions[toString(n.productID)] +
              COALESCE(' - Category: ' + 
              [(n)-[:PART_OF]->(c:Category) | c.categoryName][0], '') +
              COALESCE(' - Supplier: ' + 
              [(s:Supplier)-[:SUPPLIES]->(n) | s.companyName][0], '')
    END as enriched_text
    RETURN enriched_text AS text, id(n) AS nodeId
    """
```

#### Explanation

The `generate_embeddings` function is designed to generate enriched text descriptions for products in a graph database. Here's a breakdown of what each part of the function does:

1. **Function Definition**: 
   - `generate_embeddings(tx, node_label, descriptions)`: This function takes three parameters: `tx` (the transaction object), `node_label` (the label of the nodes to match), and `descriptions` (a dictionary of descriptions keyed by product ID).

2. **Cypher Query**:
   - The query is a Cypher query used to interact with the graph database.
   - `MATCH (n:Product)`: This part of the query matches all nodes with the label `Product`.
   - `WHERE n.productID IS NOT NULL`: This filters the nodes to only include those with a non-null `productID`.

3. **WITH Clause**:
   - `WITH n, CASE WHEN n:Product = true THEN ... END as enriched_text`: This clause constructs an enriched text description for each product node.
   - `CASE WHEN n:Product = true`: This checks if the node is a `Product`.
   - `$descriptions[toString(n.productID)]`: This retrieves the description for the product from the `descriptions` dictionary using the product ID.
   - `COALESCE(' - Category: ' + [(n)-[:PART_OF]->(c:Category) | c.categoryName][0], '')`: This appends the category name to the description if the product is part of a category.
   - `COALESCE(' - Supplier: ' + [(s:Supplier)-[:SUPPLIES]->(n) | s.companyName][0], '')`: This appends the supplier name to the description if the product has a supplier.

4. **RETURN Clause**:
   - `RETURN enriched_text AS text, id(n) AS nodeId`: This returns the enriched text and the node ID for each product node.

This function enriches the product descriptions by including additional information such as category and supplier, which can be useful for generating more informative vector embeddings.

## Cypher generation
The `cypher.py` file is responsible for translating user queries into Cypher queries and executing them against a Neo4j database. Here's a detailed explanation of the code:

### Explanation

1. **Imports**:
   - The file imports necessary modules and classes, including `streamlit`, `llm`, `graph`, `GraphCypherQAChain`, `PromptTemplate`, and various Neo4j exceptions for error handling.

2. **Cypher Generation Template**:
   - A template string, `CYPHER_GENERATION_TEMPLATE`, is defined to guide the translation of user questions into Cypher queries. It includes example Cypher statements for common queries, such as finding products by category or retrieving customer orders.

3. **Prompt Template**:
   - `cypher_prompt` is created using `PromptTemplate.from_template`, which utilizes the `CYPHER_GENERATION_TEMPLATE` to generate Cypher queries based on user input.

4. **Logging**:
   - A logger is initialized to log information and errors related to Cypher query execution.

5. **Function: `execute_cypher_query`**:
   - This function takes a Cypher query as input and logs the query execution.
   - It attempts to execute the query using `cypher_qa` and handles various exceptions:
     - **ClientError**: Indicates a syntax or entity reference issue in the query.
     - **TransientError**: Suggests temporary unavailability of the database.
     - **DatabaseError**: Points to a persistent issue with the database.
     - **General Exception**: Catches any unexpected errors.
   - The function returns the query result or an error message with suggestions for resolution.

6. **Cypher QA Chain**:
   - `cypher_qa` is an instance of `GraphCypherQAChain`, configured with the language model (`llm`), the graph database connection (`graph`), and the `cypher_prompt`.
   - It is set to be verbose and allows direct execution of potentially dangerous requests. This is due to the fact that the cypher query that is generated can also execute malicious code resulting in unwanted deletion and modification of data. we can prevent it using a middleware or configuring only read access to the DB for this user.


````python:tools/cypher.py
import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from neo4j.exceptions import ClientError, TransientError, DatabaseError
from logger import log

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about products, orders, and customers in the Northwind database.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statements:

1. To find products by category:
```
MATCH (p:Product)-[:PART_OF]->(c:Category {{categoryName: "Beverages"}})
RETURN p.productName, p.unitPrice
```

2. To find orders for a customer by customerID (say ALFKI):
```
MATCH (c:Customer {{customerID: "ALFKI"}}-[:PURCHASED]->(o:Order)
RETURN o.orderID, o.orderDate, o.shipAddress
```

3. To find product details with supplier:
```
MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)
WHERE p.productName CONTAINS 'Chai'
RETURN s.companyName, p.productName, p.unitPrice
```

4. To find total number of customers:
```
MATCH (c:Customer)
RETURN COUNT(c) as customerCount
```

Schema:
{schema}

Question:
{question}
"""
cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

logger = log('cypher')
def execute_cypher_query(query):
    logger.info(f"Executing Cypher query: {query}")
    try:
        response = cypher_qa(query)
        
        # Check if the response contains an error message
        if isinstance(response, str) and "ERROR:" in response:
            return {
                "error": {
                    "type": "invalid_query",
                    "message": response.split("ERROR: ")[1],
                    "suggestion": "Please ask about entities that exist in the database schema."
                }
            }
        
        logger.info(f"Query results: {response}")
        return {"result": response}
        
    except ClientError as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return {
            "error": {
                "type": "client_error",
                "message": str(e),
                "suggestion": "Please check your query syntax and ensure all referenced entities exist."
            }
        }
        
    except TransientError as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return {
            "error": {
                "type": "transient_error",
                "message": str(e),
                "suggestion": "The database is temporarily unavailable. Please try again in a few moments."
            }
        }
    except DatabaseError as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return {
            "error": {
                "type": "database_error",
                "message": str(e),
                "suggestion": "There was an issue with the database. Please contact support if this persists."
            }
        }
        
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        return {
            "error": {
                "type": "unknown_error",
                "message": str(e),
                "suggestion": "An unexpected error occurred. Please try a different query."
            }
        }
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,
    return_direct=True
)
````


This code is integral to the chatbot's ability to process natural language queries and interact with the Neo4j database, providing users with accurate and relevant information.


## User Interface Development

The user interface for the chatbot is developed using Streamlit. The main code for the chat interface is located in `bot.py`.

```python:bot.py

def handle_submit(message):
    """Submit handler that processes user input and generates response"""
    
    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent without showing intermediate steps
        response = generate_response(message, show_intermediate_steps=False)
        
        print("Response received in bot.py:", response)  # Debug print
```



### Running the Application
To run the application locally, use the following command:

```sh
streamlit run bot.py
```

Streamlit will start a server on `http://localhost:8501`.

### Session Management
The application uses Streamlit's session state to maintain conversation history.

```python:bot.py
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Neo4j Chatbot! How can I help you?"},
    ]
```

## Natural Language Processing

The application leverages AI models for natural language processing. An instance of the ChatOpenAI is created to communicate with a GPT model using OpenAI.

### API Key Management
API keys are stored securely in the `secrets.toml` file.

```python
import streamlit as st

openai_api_key = st.secrets['OPENAI_API_KEY']
openai_model = st.secrets['OPENAI_MODEL']
```
## Vector Embeddings and Semantic Search

The application uses vector embeddings for semantic search. The embeddings are stored in the graph database and are used to find semantically related texts. Here's how the vector index is created:

```python
def create_vector_index(tx):
    # Create vector index for product names
    query = """
    CREATE VECTOR INDEX productNameIndex IF NOT EXISTS
    FOR (p:Product)
    ON (p.productNameEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """
    tx.run(query)
```

The code above creates a vector index in Neo4j for semantic search capabilities. Let's break down what each part does:



1. The function `create_vector_index(tx)` takes a Neo4j transaction object as a parameter
2. Inside the function, a Cypher query is defined that:
   - Creates a vector index named "productNameIndex" if it doesn't already exist
   - Applies the index to nodes with the "Product" label
   - Uses the "productNameEmbedding" property of those nodes to store vector embeddings
   - Configures the index with:
     - 768 dimensions for the vectors (matching the embedding model's output size)
     - Cosine similarity as the comparison function between vectors
3. The query is executed using `tx.run(query)`

This vector index enables efficient similarity searches across product descriptions by comparing their vector embeddings, which is essential for semantic search functionality in the chatbot.


### Embedding Model Initialization
The `llm.py` script is used to configure the embedding model settings.

```python:llm.py
# Create the Embedding model
from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-small",
    dimensions=768
)
```

### Explaination
 `api_key=st.secrets["OPENAI_API_KEY"]`: This parameter provides the API key required to authenticate with OpenAI's services. The key is securely retrieved from Streamlit's secrets management system (st.secrets).

`model="text-embedding-3-small"`:This specifies the model to be used for generating embeddings. The model "text-embedding-3-small" is likely a version optimized for creating text embeddings.

`dimensions=768`: This sets the dimensionality of the embeddings. A dimension of 768 is common for many language models, providing a balance between detail and computational efficiency.


## Agent Configuration

The main component of the application is the agent, defined in `agent.py`. The agent processes user queries and generates responses using various tools and models.
## Understanding Agents

An agent is a system that uses a Language Model (LLM) to determine the control flow of an application. More specifically, agents enable LLMs to interact with external tools and make decisions about which actions to take in which order.

The key components of an agent system include:

1. A base Language Model (LLM)
2. Tools that the agent can interact with (like calculators, search, code execution)
3. An agent to control the interaction

### ReAct Agent Architecture

The chatbot uses a ReAct (Reasoning and Acting) agent architecture, which is based on the paper ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629). This architecture enables the agent to:

1. Think through its reasoning process step-by-step
2. Take actions using available tools when needed
3. Observe the results of those actions
4. Continue reasoning based on observations

-----
Let's explore the relevant code snippets from `agent.py` and `agent.md` to understand how the agent is configured and operates within the codebase.

### Key Components of the Agent

#### 1. Language Model (LLM)

The agent uses a language model to process user queries and generate responses. This is configured using a chat prompt system.


```python:agent.py
from langchain_core.runnables import RunnablePassthrough

from tools.cypher import cypher_qa
from tools.vector import retriever

chat_prompt = ChatPromptTemplate.from_messages(
    [
```


- **Chat Prompt Template**: The `ChatPromptTemplate` is used to define the interaction when the agent doesnt need to query the database or do a product search. It sets the context for the chatbot as a company internal service expert.

#### 2. Tools

The agent uses specialized tools to handle different types of queries. These tools are defined to perform specific tasks such as general chat, database queries, and semantic product searches.


```python:agent.py
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
```

### Explanation

1. **Tool Definitions**:
   - The code defines a list called `tools`, which contains three tools, each created using `Tool.from_function`.
   - Each tool has a `name`, `description`, and `func` (function) that it uses to perform its task.

2. **Tools**:
   - **General Chat**:
     - **Name**: "General Chat"
     - **Description**: Handles general chat about the company and its products/services not covered by other tools.
     - **Function**: `general_chat.invoke` is the function that processes these queries.
   
   - **Northwind Information**:
     - **Name**: "Northwind information"
     - **Description**: Provides information about products, orders, and customers in the Northwind database using Cypher queries.
     - **Function**: `cypher_qa` is the function that executes Cypher queries to interact with the database.
   
   - **Product Search**:
     - **Name**: "Product Search"
     - **Description**: Finds similar products or searches product descriptions semantically.
     - **Function**: A lambda function that retrieves relevant documents using `retriever.get_relevant_documents(q)`, and formats them into a list of dictionaries containing `page_content` and `metadata`.

#### 3. Memory Management

The agent maintains conversation history using Neo4j, which helps in providing context for follow-up questions.


```python:agent.py
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

```


- **Memory Function**: The `get_memory` function retrieves and manages the last three messages in the conversation history, simplifying them for storage.

### Agent Configuration and Execution

#### 1. Agent Setup

The agent is configured using the ReAct (Reasoning and Acting) framework, which allows it to reason through its actions and use tools as needed.


````python:agent.py

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

---
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
---

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

--
Thought: Do I need to use a tool? No
Final Answer: [your response here]
--

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
````


- **Agent Prompt**: The `agent_prompt` defines guidelines for tool selection and response generation. It instructs the agent on when to use specific tools based on the query type.

#### 2. Message History Integration

The agent is converted into a runnable format and integrated with a message history system to maintain context.


```python:agent.py

# Convert AgentExecutor to Runnable
runnable_agent = RunnablePassthrough() | agent_executor

chat_agent = RunnableWithMessageHistory(
    runnable_agent,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)
```


- **Runnable Agent**: The `RunnableWithMessageHistory` wraps the agent to handle input and output messages, maintaining the chat history.

### Response Generation

The `generate_response` function is responsible for processing user input and generating a response using the agent.


```python:agent.py
def generate_response(user_input, show_intermediate_steps=False):
{user_input}")
    
    try:
        (get_session_id()).messages}")
        response = chat_agent.invoke(
            {
                "input": user_input,
                "chat_history": get_memory(get_session_id()).messages  # Pass messages directly
            },
            {"configurable": {"session_id": get_session_id()}},
        )
                
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
         return {
            'output': output,
            'intermediate_steps': steps if show_intermediate_steps else []
        }
```


- **Response Flow**: The function logs the input, invokes the chat agent, and processes the response. It also logs intermediate steps if required, providing transparency in the agent's decision-making process.

### Explanation of How It Works

- **Input Handling**: User input is processed through the `generate_response` function, which uses the agent prompt to determine the appropriate tool and action.
- **Tool Selection**: The agent evaluates the input against predefined guidelines to select the right tool for the task.
- **Response Generation**: Using the ReAct framework, the agent reasons through its actions, uses tools as needed, and generates a final response.
- **Memory Management**: The agent maintains a simplified conversation history to provide context for follow-up questions, ensuring coherent interactions.

This setup allows the agent to effectively handle a variety of queries related to the Northwind database, leveraging both language models and database tools for comprehensive responses.

-----



This documentation provides a detailed overview of the chatbot application, with code snippets illustrating key components and processes. For further details, refer to the respective files in the codebase.
