import streamlit as st
import os
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import nltk
nltk.download('punkt')

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("Graph Database Question Answering")

# Set Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize the Neo4j graph object
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Display the graph schema
st.subheader("Graph Schema")
schema = graph.refresh_schema()
st.write(schema)

# Load Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM (Large Language Model) using Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Initialize the GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# Input box for user queries
st.subheader("Ask a Question")
query = st.text_input("Enter your query:", value="Who was the director of the movie Casino")

if st.button("Get Answer"):
    if query:
        # Process the query and get the response
        response = chain.invoke({"query": query})
        
        # Check if the response has the expected format
        if response and isinstance(response, dict):
            # Extract the relevant answer from the response
            answer = response.get('result', 'No answer found')
            
            # Display the answer in a more user-friendly format
            st.subheader("Answer")
            st.write(answer)
        else:
            st.error("Unexpected response format. Please try again.")
    else:
        st.warning("Please enter a query to get an answer.")