import chromadb

client = chromadb.PersistentClient(path="data/vector_store")
collection = client.get_collection("documents")

result = collection.get(include=['documents','metadatas']) 

with open("chunk_mapping.txt","w", encoding="utf-8") as f:
    for i, doc_id in enumerate(result['ids']):
        f.write(f"--- DOC ID: {doc_id} ---\n")
        f.write(f"{result['documents'][i]}\n\n")

print("Success! chunk_mapping.txt is created.")
