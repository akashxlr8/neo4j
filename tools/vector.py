import streamlit as st
from llm import llm, embeddings
from graph import graph

# tag::import_vector[]
from langchain_neo4j import Neo4jVector
# end::import_vector[]
# tag::import_chain[]
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# end::import_chain[]

# tag::import_chat_prompt[]
from langchain_core.prompts import ChatPromptTemplate
# end::import_chat_prompt[]


# tag::vector[]
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="productNameIndex",
    node_label="Product",
    text_node_property="productDescription",
    embedding_node_property="productNameEmbedding",
    retrieval_query="""
    RETURN
        node.productDescription AS text,
        score,
        {
            productID: node.productID,
            unitPrice: node.unitPrice,
            category: [(node)-[:PART_OF]->(c:Category) | c.categoryName][0],
            supplier: [(s:Supplier)-[:SUPPLIES]->(node) | s.companyName][0],
            unitsInStock: node.unitsInStock
        } AS metadata
    """
)
# end::vector[]

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]

# tag::prompt[]
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)
# end::prompt[]

# tag::chain[]
question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)
# end::chain[]

# tag::get_movie_plot[]
def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})
# end::get_movie_plot[]