import streamlit as st
import pandas as pd
from google import genai
import chromadb
import time

st.title("Company Policy Assistant")

policies = pd.read_csv("company_policies.csv")

st.write("Company policies loaded:", len(policies))

st.subheader("1. Rules-Based Search")

question = st.text_input("Ask a company policy question:")

if question:
    start_time = time.perf_counter()
    stop_words = {"i", "can", "a", "the", "is", "are", "do", "does",
              "from", "to", "in", "on", "my", "me", "what",
              "how", "where", "when", "for"}

    keywords = [
        word.strip("?!.,").lower()
        for word in question.split()
        if word.strip("?!.,").lower() not in stop_words
    ]

    def relevance_score(row):
        title = str(row["title"]).lower()
        policy_text = str(row["policy_text"]).lower()

        return sum(
            3 * (word in title) + (word in policy_text)
            for word in keywords
        )

    matches = policies.copy()
    matches["score"] = matches.apply(relevance_score, axis=1)
    matches = matches[matches["score"] > 0]
    matches = matches.sort_values("score", ascending=False)

    if not matches.empty:
        policy = matches.iloc[0]
        st.write("Answer:", policy["policy_text"])
        st.write("Relevant policy:", policy["title"])
    else:
        st.write("No matching policy found.")
    elapsed_time = time.perf_counter() - start_time
    st.write(f"Response time: {elapsed_time:.4f} seconds")

st.subheader("2. LLM Without Vector Index")

if question:
    if st.button("Ask LLM"):
        start_time_2 = time.perf_counter()
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

        policy_context = "\n\n".join(
            f"Policy: {row['title']}\n"
            f"Department: {row['department']}\n"
            f"Text: {row['policy_text']}"
            for _, row in policies.iterrows()
        )

        prompt = f"""
You are a company policy assistant.
Answer using only the company policies below.
Identify the relevant policy by its exact title.
If the policies do not answer the question, say so.
Do not invent policies.

COMPANY POLICIES:
{policy_context}

EMPLOYEE QUESTION:
{question}
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        st.write(response.text)
        if response.usage_metadata:
            st.write("Total tokens:", response.usage_metadata.total_token_count)
        elapsed_time_2 = time.perf_counter() - start_time_2
        st.write(f"Response time: {elapsed_time_2:.4f} seconds")

st.subheader("3. LLM With Vector Index")

if question:
    if st.button("Ask LLM With Vector Index"):
        start_time_3 = time.perf_counter()
        # Open the existing vector database
        vector_client = chromadb.PersistentClient(path="./chroma_db")

        collection = vector_client.get_collection(
            name="company_policies"
        )

        # Find the 3 most relevant policies
        results = collection.query(
            query_texts=[question],
            n_results=3
        )

        relevant_policies = "\n\n".join(
            results["documents"][0]
        )

        # Ask Gemini using the retrieved policies
        gemini_client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"],
            http_options={"timeout": 30000}
        )

        prompt = f"""
You are a company policy assistant.

Answer using only the company policies below.
Identify the relevant policy by its exact title.
If the policies do not answer the question, say so.
Do not invent policies.

RELEVANT COMPANY POLICIES:
{relevant_policies}

EMPLOYEE QUESTION:
{question}
"""
        st.write("Vector search completed. Waiting for Gemini...")

        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        st.write(response.text)
        if response.usage_metadata:
            st.write("Total tokens:", response.usage_metadata.total_token_count)
        elapsed_time_3 = time.perf_counter() - start_time_3
        st.write(f"Response time: {elapsed_time_3:.4f} seconds")