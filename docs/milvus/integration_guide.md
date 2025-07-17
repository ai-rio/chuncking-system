# Milvus Integration Guide

This document outlines the rationale and technical steps for integrating the Milvus vector database into our project.

---

## 1. Why Integrate Milvus?

*Contributed by: John, Product Manager*

Milvus is a high-performance, open-source vector database designed for scale. Integrating it into our system offers several key advantages:

- **Efficient Unstructured Data Management:** Our project deals with large volumes of unstructured and multi-modal data (text, documents, etc.). Milvus is purpose-built to organize, store, and search this type of data efficiently.
- **Powerful Similarity Search:** At its core, Milvus enables lightning-fast similarity searches on massive-scale vector datasets. This will be instrumental in features like semantic document search, finding related content, and powering future AI-driven capabilities.
- **Scalability and Performance:** Milvus features a distributed, cloud-native architecture (running on Kubernetes) that can scale horizontally to handle billions of vectors and thousands of queries per second. It also leverages hardware acceleration (CPU/GPU) to ensure top-tier performance.
- **Real-time Data Ingestion:** The system supports real-time streaming updates, ensuring that our vector indexes are always fresh and searches reflect the most current data without costly re-indexing.

By leveraging Milvus, we can offload the complex task of vector management and search to a specialized, battle-tested system, allowing us to focus on building our core application features.

---

## 2. Architectural Considerations

*Contributed by: Winston, Architect*

From an architectural standpoint, Milvus will be introduced as a new, independent service that our application interacts with via its Python SDK. 

- **Deployment Model:** We will start with the `Milvus Lite` deployment for local development and testing. This lightweight version is bundled with the Python SDK and runs in-process, requiring zero external dependencies. For production, we will deploy Milvus as a standalone, containerized service within our Kubernetes cluster.
- **Data Flow:**
  1. Our `chunking-system` will process and chunk documents as it currently does.
  2. An embedding model will convert each chunk into a vector embedding.
  3. The vector embedding, along with its source metadata (e.g., document ID, chunk number, original text), will be inserted into a Milvus collection.
  4. When a user performs a search, their query will be converted into a vector, and Milvus will be queried to find the most similar vectors (and their associated metadata) in the collection.
- **Integration Point:** A new service or module, `VectorDBService`, will be created to encapsulate all interactions with Milvus. This service will handle connection management, collection creation, data insertion, and search queries. Other parts of the application will interact with this service rather than directly with the Milvus client, adhering to the principle of separation of concerns.

---

## 3. How to Integrate Milvus (Python SDK)

*Contributed by: James, Developer*

Here is a practical guide to integrating Milvus using the `pymilvus` Python SDK.

### 3.1. Installation

First, add `pymilvus` to our project's dependencies and install it.

```bash
# Add to requirements.txt or pyproject.toml
pymilvus

# Install
pip install -U pymilvus
```

### 3.2. Connecting to Milvus

The `MilvusClient` is the main entry point for all operations.

**For Local Development (Milvus Lite):**
Simply provide a local file path to persist the data. This is ideal for local testing and development.

```python
from pymilvus import MilvusClient

# This will create a local file named 'milvus_demo.db' to store data.
client = MilvusClient("milvus_demo.db")
```

**For Production (Connecting to a Server):**
When connecting to a deployed Milvus server or a managed cloud instance, provide the URI and authentication token.

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="<your_milvus_endpoint>",
    token="<your_username_and_password_or_api_key>"
)
```

### 3.3. Core Operations

Here are the fundamental operations you'll perform with the client.

**1. Create a Collection:**
A collection in Milvus is analogous to a table in a relational database. We need to define its structure, including the vector dimension.

```python
# The vectors we will use in this demo have 768 dimensions
client.create_collection(
    collection_name="document_chunks",
    dimension=768,  
)
```

**2. Insert Data:**
Data is inserted as a list of dictionaries or a dictionary of lists.

```python
# Assume `data` is a list of dictionaries like:
# data = [
#   {'id': 1, 'vector': [0.1, ..., 0.2], 'text': 'This is the first chunk.'},
#   {'id': 2, 'vector': [0.3, ..., 0.4], 'text': 'This is the second chunk.'}
# ]

res = client.insert(
    collection_name="document_chunks", 
    data=data
)

print(res)
```

**3. Perform a Vector Search:**
To search, provide a list of query vectors. Milvus will return the most similar results.

```python
# Assume `embedding_fn` is a function that converts text to vectors
query_vectors = embedding_fn.encode_queries(["What is vector search?"])

results = client.search(
    collection_name="document_chunks",
    data=query_vectors,  # A list of one or more query vectors
    limit=5,  # Return the top 5 most similar results
    output_fields=["text", "source_document"],  # Specify which fields to return
)

# Process results
for result_set in results:
    for hit in result_set:
        print(hit)
```

### 3.4. Next Steps

- Create the `VectorDBService` module.
- Integrate the Milvus client initialization into the application's startup sequence.
- Modify the document processing pipeline to insert chunk embeddings into Milvus.
- Build a search endpoint that leverages the `VectorDBService` to perform similarity searches.