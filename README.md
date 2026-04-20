# A repository to learn how embedding works 

This repo will be used to document my journey in learning how embedding works.
I will be using my Obsidian vault as the source of data for the embedding

## What is embedding?

Embedding is a process of converting data (texts) into a numerical format (vector). These vectors can then be used for various tasks such as similarity search, clustering, and classification. The idea is to capture the semantic meaning of the text in a way that similar texts have similar vector representations. 

## What's being used?
 - Python: Language used for coding
 - Qwen3: The embedding model used to convert text into vectors.
 - Langchain: A framework to build applications with language models, used here to manage the embedding and retrieval process.
 - BM25: A ranking function used by search engines to estimate the relevance of documents to a given search query, used here to enhance the retrieval process by combining it with embedding-based similarity.
 - ChromaDB: A vector database used to store and retrieve the vector representations of the documents.

## Files:
 - FirstEmbedding.py: A simple script to test the embedding model and see how it works.
 - ImportingEmbedding.py: A script to test how to import the embedding model and use it in a more complex way.
 - HybridBM25XEmbedding.py: A script that combines BM25 and embedding-based retrieval to enhance search results. It calculates similarity scores based on both BM25 and vector embeddings to provide more relevant search results.
