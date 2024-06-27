import chromadb

client = chromadb.PersistentClient(path = "embeddings")


# collection = client.create_collection(name="my_collection",metadata={"hnsw:space": "l2"})
collection = client.get_collection(name = "my_collection")


collection.upsert(
    documents=["doc1", "doc2", "doc3"],
    embeddings=[[1.7, 3.3, 0.2], [4.5, 6.9, 4.4], [1.7, 2.3, 3.2]],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
    ids=["id1", "id2", "id3"]
)

results = collection.query(
    query_embeddings=[[1.7, 2.3, 3.2]],
    n_results=1
)

print(results)