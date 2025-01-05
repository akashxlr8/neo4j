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

