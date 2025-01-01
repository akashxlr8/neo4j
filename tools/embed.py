import streamlit as st
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

# Initialize the Neo4j driver
driver = GraphDatabase.driver(
    st.secrets["NEO4J_URI"],
    auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
)

# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-small",
    dimensions=768
)

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
    
    for record in tx.run(query, descriptions=descriptions):
        text = record["text"]
        node_id = record["nodeId"]
        
        embedding = embeddings_model.embed_query(text)
        
        update_query = """
        MATCH (n)
        WHERE id(n) = $nodeId
        SET n.productDescription = $text,
            n.productNameEmbedding = $embedding
        """
        tx.run(update_query, nodeId=node_id, text=text, embedding=embedding)

def load_product_descriptions():
    descriptions = {}
    with open('northwind-data-importer-mode-data/product_desc.txt', 'r') as file:
        for line in file:
            if line.strip():
                # Extract product ID from the start of line (e.g., "1. Chai:" -> "1")
                product_id = line.split('.')[0].strip()
                # Extract description (everything after the colon)
                description = line.split(':', 1)[1].strip() if ':' in line else ''
                descriptions[product_id] = description
    return descriptions

def main():
    with driver.session() as session:
        # First create the vector index
        session.execute_write(create_vector_index)
        
        # Then generate embeddings using product descriptions
        session.execute_write(
            generate_embeddings, 
            "Product", 
            load_product_descriptions()
        )

    driver.close()

if __name__ == "__main__":
    main()


#####################

# import os
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from openai import OpenAI
# from neo4j import GraphDatabase

# COURSES_PATH = "llm-vectors-unstructured/data/asciidoc"

# loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
# docs = loader.load()

# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1500,
#     chunk_overlap=200,
# )

# chunks = text_splitter.split_documents(docs)

# #For each chunk, you have to create an embedding of the text and extract the metadata.

# # Create a function to create and return an embedding using the OpenAI API:
# def get_embedding(llm, text):
#     response = llm.embeddings.create(
#             input=chunk.page_content,
#             model="text-embedding-ada-002"
#         )
#     return response.data[0].embedding

# # Create a 2nd function, which will extract the data from the chunk:
# def get_course_data(llm, chunk):
#     data = {}

#     path = chunk.metadata['source'].split(os.path.sep)

#     data['course'] = path[-6]
#     data['module'] = path[-4]
#     data['lesson'] = path[-2]
#     data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
#     data['text'] = chunk.page_content
#     data['embedding'] = get_embedding(llm, data['text'])

#     return data

# llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # Connect to the Neo4j sandbox:
# driver = GraphDatabase.driver(
#     st.secrets["NEO4J_URI"],
#     auth=(
#         st.secrets["NEO4J_USERNAME"],
#         st.secrets["NEO4J_PASSWORD"]
#     )
# )
# driver.verify_connectivity()

# #To create the data in the graph, you will need a function that incorporates the course data into a Cypher statement and runs it in a transaction.
# def create_chunk(tx, data):
#     tx.run("""
#         MERGE (c:Course {name: $course})
#         MERGE (c)-[:HAS_MODULE]->(m:Module{name: $module})
#         MERGE (m)-[:HAS_LESSON]->(l:Lesson{name: $lesson, url: $url})
#         MERGE (l)-[:CONTAINS]->(p:Paragraph{text: $text})
#         WITH p
#         CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
#         """, 
#         data
#         )
    
#  # terate through the chunks and execute the create_chunk function:   
# for chunk in chunks:
#     with driver.session(database="neo4j") as session:
        
#         session.execute_write(
#             create_chunk,
#             get_course_data(llm, chunk)
#         )
        
    
# driver.close()