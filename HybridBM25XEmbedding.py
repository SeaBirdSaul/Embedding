import os
import pickle
import math
import re
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Use a public model since EmbeddingGemma is gated
embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

# File paths for cached data
BM25_CACHE_FILE = "bm25_retriever.pkl"
FAISS_INDEX_FOLDER = "faiss_index"

def load_documents(vault_path):
    """
    Loads all .md files in the Obsidian vault and returns a list of Document objects.
    """
    docs = []
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():
                        # Create a doc with metadata containing the file path
                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path}
                        )
                        docs.append(doc)
                        print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return docs


def save_bm25_retriever(retriever, file_path=BM25_CACHE_FILE):
    """Save BM25 retriever to disk."""
    with open(file_path, 'wb') as f:
        pickle.dump(retriever, f)
    print(f"BM25 retriever saved to {file_path}")


def load_bm25_retriever(file_path=BM25_CACHE_FILE):
    """Load BM25 retriever from disk if it exists."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            retriever = pickle.load(f)
        print(f"BM25 retriever loaded from {file_path}")
        return retriever
    return None


def save_faiss_index(vector_db, folder=FAISS_INDEX_FOLDER):
    """Save FAISS vector store to disk."""
    vector_db.save_local(folder)
    print(f"FAISS index saved to {folder}")


def load_faiss_index(folder=FAISS_INDEX_FOLDER):
    """Load FAISS vector store from disk if it exists."""
    if os.path.exists(folder):
        vector_db = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
        print(f"FAISS index loaded from {folder}")
        return vector_db
    return None


def tokenize(text):
    """Simple tokenization: lowercase and split on non-alphanumeric characters."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_bm25_score(query, doc_text, docs, k1=1.5, b=0.75):
    """
    Compute BM25 score for a document given a query.
    
    Args:
        query: The search query
        doc_text: The document text
        docs: List of all documents (for computing IDF)
        k1: Term frequency scaling parameter (default 1.5)
        b: Document length normalization parameter (default 0.75)
    
    Returns:
        BM25 score
    """
    query_terms = tokenize(query)
    doc_terms = tokenize(doc_text)
    
    N = len(docs)  # Total number of documents
    doc_len = len(doc_terms)  # Length of current document
    
    # Calculate average document length
    avgdl = sum(len(tokenize(d.page_content)) for d in docs) / N if N > 0 else 1
    
    # Count term frequencies in document
    term_freq = {}
    for term in doc_terms:
        term_freq[term] = term_freq.get(term, 0) + 1
    
    # Count documents containing each term
    doc_count = {}
    for d in docs:
        d_terms = set(tokenize(d.page_content))
        for term in query_terms:
            if term in d_terms:
                doc_count[term] = doc_count.get(term, 0) + 1
    
    score = 0.0
    for term in query_terms:
        if term not in term_freq:
            continue
            
        # IDF: log((N - nt + 0.5) / (nt + 0.5))
        nt = doc_count.get(term, 0)
        if nt == 0:
            continue
        idf = math.log((N - nt + 0.5) / (nt + 0.5) + 1)  # Added +1 to avoid log(0)
        
        # TF: freq / (freq + k1 * (1 - b + b * (|d| / avgdl)))
        freq = term_freq[term]
        tf = freq / (freq + k1 * (1 - b + b * (doc_len / avgdl)))
        
        score += idf * tf
    
    return score


def calculate_similarity(query, results, docs, alpha=0.4):
    """
    Calculate combined similarity scores (BM25 + vector) for retrieved documents.
    
    Args:
        query: The search query
        results: List of retrieved documents
        docs: All documents (for BM25 calculation)
        alpha: Weight for BM25 score (0.4 = 40% BM25, 60% vector)
    
    Returns:
        List of (document, score) tuples sorted by score descending
    """
    # Get the query embedding for vector similarity
    query_embedding = embeddings.embed_query(query)
    
    # Calculate BM25 scores for normalization
    bm25_scores = []
    for doc in results:
        bm25_score = compute_bm25_score(query, doc.page_content, docs)
        bm25_scores.append(bm25_score)
    
    # Normalize BM25 scores to [0, 1] range
    max_bm25 = max(bm25_scores) if bm25_scores else 1
    min_bm25 = min(bm25_scores) if bm25_scores else 0
    bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
    
    scored_results = []
    for i, doc in enumerate(results):
        # Vector similarity (cosine)
        doc_text = doc.page_content
        doc_embedding = embeddings.embed_query(doc_text[:1000])
        
        dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
        query_magnitude = sum(a * a for a in query_embedding) ** 0.5
        doc_magnitude = sum(b * b for b in doc_embedding) ** 0.5
        
        if query_magnitude > 0 and doc_magnitude > 0:
            vector_score = dot_product / (query_magnitude * doc_magnitude)
        else:
            vector_score = 0.0
        
        # Normalized BM25 score
        bm25_score = (bm25_scores[i] - min_bm25) / bm25_range if bm25_range > 0 else 0
        
        # Combined score: alpha * BM25 + (1 - alpha) * vector
        combined_score = alpha * bm25_score + (1 - alpha) * vector_score
        
        scored_results.append((doc, combined_score, bm25_score, vector_score))
    
    # Sort by combined score descending
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results


if __name__ == "__main__":
    vault_path = input("Enter the path to your Obsidian vault folder: ").strip()
    if not os.path.isdir(vault_path):
        print("Invalid directory path.")
        exit(1)

    # Load documents
    print("Loading documents...")
    docs = load_documents(vault_path)

    # Try to load cached BM25 retriever, otherwise create and save it
    print("Loading/creating BM25 retriever...")
    bm25_retriever = load_bm25_retriever()
    if bm25_retriever is None:
        bm25_retriever = BM25Retriever.from_documents(docs)
        save_bm25_retriever(bm25_retriever)
    bm25_retriever.k = 5

    # Try to load cached FAISS index, otherwise create and save it
    print("Loading/creating vector store...")
    vector_db = load_faiss_index()
    if vector_db is None:
        vector_db = FAISS.from_documents(docs, embeddings) # vector_db = FAISS.from_documents(docs, OpenAIEmbeddings)
        save_faiss_index(vector_db)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # Combine the retrievers
    print("Creating hybrid retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] # BM25: 40%, Vector: 60%
    )

    while True:
        query = input("Enter a text or phrase to search (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        if query:
            results = hybrid_retriever.invoke(query)
            # Calculate similarity scores
            scored_results = calculate_similarity(query, results, docs)
            
            print("\nTop 5 similar files:")
            for i, (doc, score, bm25, vector) in enumerate(scored_results, 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"{i}. {source}")
                print(f"   Combined: {score:.4f} | BM25: {bm25:.4f} | Vector: {vector:.4f}")
        else:
            print("Please enter a valid query.")