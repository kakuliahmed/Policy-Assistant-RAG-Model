import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_mistralai import ChatMistralAI

# -------------------
# CONFIG
# -------------------
DB_PATH = "lic_vector_db_v2"
COLLECTION_NAME = "lic_policies"
MISTRAL_API_KEY = "MGPRMWWZ8E9eV9FNSY74VL9ybMHW4cfc"  # <-- Replace with key

# -------------------
# Load Vector DB
# -------------------
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

# -------------------
# Mistral Model
# -------------------
llm = ChatMistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)

# -------------------
# UI
# -------------------
st.title("🤖 LIC IntelliSearch Assistant")
st.write("Ask about LIC policies and I will answer only using the official policy database.")

# Friendly greeting
st.info(" Hello! You can ask things like:\n- Best term insurance plan\n- Policy with lifelong coverage\n- Plans for children")

question = st.text_input("🔍 What policy information would you like to know?")

if question:
    with st.spinner("Checking database..."):
        result = collection.query(
            query_texts=[question],
            n_results=1,
            include=["metadatas", "distances"]
        )

    # Extract result
    distance = result["distances"][0][0] if result and "distances" in result else None
    policy = result["metadatas"][0][0] if result and "metadatas" in result else None

    # Safety rule: If match is weak, do NOT answer
    if policy is None or distance is None or distance > 0.40:
        st.warning("Sorry, I couldn't find relevant information in the LIC database.\n\nPlease try rephrasing your question.")
    else:
        # Force formatted short answer
        prompt = f"""
        You MUST answer ONLY based on the policy metadata below.
        Do NOT add any new external information.
        Keep the answer short and useful.

        POLICY DATA:
        Policy Name: {policy['Policy Name']}
        Policy ID: {policy['Policy ID']}
        Description: {policy['Description']}
        Duration: {policy['Duration']}

        USER QUESTION: {question}

        FORMAT STRICTLY LIKE THIS:

        Policy Name:
        Policy ID:
        Description:
        Duration:
        """

        response = llm.invoke(prompt).content

        st.success(response)