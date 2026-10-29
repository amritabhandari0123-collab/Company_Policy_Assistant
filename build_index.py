import pandas as pd
import chromadb

# Load company policies
policies = pd.read_csv("company_policies.csv")

# Create a local vector database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="company_policies"
)

# Prepare policy documents
documents = []
ids = []
metadatas = []

for i, row in policies.iterrows():
    documents.append(
        f"Policy: {row['title']}\n"
        f"Department: {row['department']}\n"
        f"Text: {row['policy_text']}"
    )

    ids.append(str(i))

    metadatas.append({
        "title": str(row["title"])
    })

# Create embeddings and store them
collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("Vector index created successfully!")
print("Number of policies:", collection.count())