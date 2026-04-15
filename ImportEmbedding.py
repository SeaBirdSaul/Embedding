'''
Imports an Obsidian vault, embedding the text files within it and saving the embeddings to a file.
Takes user inputted word or phrase and finds the most similar text files within the vault, printing the results to the console.
'''

import os
import pickle
from sentence_transformers import SentenceTransformer, util

# Use a public model since EmbeddingGemma is gated
model = SentenceTransformer("google/embeddinggemma-300m")

def embed_vault(vault_path, embedding_file='vault_embeddings.pkl'):
    """
    Embeds all .md files in the Obsidian vault and saves embeddings to a pickle file.
    """
    embeddings = {}
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():  # Skip empty files
                        embedding = model.encode(content, convert_to_tensor=True)
                        embeddings[file_path] = embedding
                        print(f"Embedded: {file_path}")
                    else:
                        print(f"Skipped empty file: {file_path}")
                except Exception as e:
                    print(f"Error embedding {file_path}: {e}")
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {embedding_file}")
    return embeddings

def load_embeddings(embedding_file='vault_embeddings.pkl'):
    """
    Loads embeddings from the pickle file.
    """
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def find_most_similar(query, embeddings, top_k=5):
    """
    Finds the top_k files with the highest similarity to the query.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = {}
    for file_path, emb in embeddings.items():
        sim = util.cos_sim(query_embedding, emb).item()
        similarities[file_path] = sim
    # Sort by similarity descending
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_k]

if __name__ == "__main__":
    vault_path = input("Enter the path to your Obsidian vault folder: ").strip()
    if not os.path.isdir(vault_path):
        print("Invalid directory path.")
        exit(1)

    embedding_file = 'vault_embeddings.pkl'
    embeddings = load_embeddings(embedding_file)
    if embeddings is None:
        print("Embedding vault...")
        embeddings = embed_vault(vault_path, embedding_file)

    while True:
        query = input("Enter a text or phrase to search (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        if query:
            top_results = find_most_similar(query, embeddings)
            print("Top 5 similar files:")
            for i, (file_path, score) in enumerate(top_results, 1):
                print(f"{i}. {file_path} (similarity: {score:.4f})")
        else:
            print("Please enter a valid query.")
 

