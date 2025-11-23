import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import os

# Load Gemini API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Load FAISS Vector Store
vector_store = FAISS.load_local(
    "scholarship_faiss_store",
    embeddings=None,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# RAG Answer Function
def rag_answer(query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the following scholarship PDF information to answer the question.

{context}

Question: {query}

Answer clearly.
"""

    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)
    return response.text


# Streamlit App UI
st.set_page_config(page_title="Scholarship Chatbot", page_icon="🎓")
st.title("🎓 Scholarship Assistant")
st.write("Ask anything about the scholarship from the official document!")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = rag_answer(query)
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {e}")
