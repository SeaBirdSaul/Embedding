from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("google/embeddinggemma-300m")

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

query_embedding = model.encode(query, convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)
print(query_embedding.shape, document_embeddings.shape)

similarities = util.cos_sim(query_embedding, document_embeddings)
print(similarities)