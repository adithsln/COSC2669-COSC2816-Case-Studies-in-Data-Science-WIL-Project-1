from langchain_core.documents import Document
import os

doc = Document(
    page_content = "This is the main text content I am using for RAG",
    metadata = {
        "source":"example.txt",
        "pages":1,
        "author":"Adithya Lakshmi Narasimhan",
        "data_created":"2026-08-05"
    }
)
doc

os.makedirs("C:/rag_chatbot/data/text_file", exist_ok=True)



### TextLoader
from langchain_community.document_loaders import TextLoader




### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

dir_loader=DirectoryLoader(
    "C:/rag_chatbot/data/text_file",
    glob="**/*.txt",
    loader_cls = TextLoader,
    loader_kwargs={'encoding':'utf-8'},
    show_progress=False
)

documents = dir_loader.load()
print(documents)


from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

dir_loader=DirectoryLoader(
    "C:/rag_chatbot/data/pdf_file",
    glob="**/*.pdf",
    loader_cls = PyMuPDFLoader,
    show_progress=False
)

documents=dir_loader.load()
documents

# Creating Data Chunks 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """
    Split documents into smaller chunks for better RAG performance.
    
    Parameters:
    - chunk_size: Maximum characters per chunk (adjust based on your LLM)
    - chunk_overlap: Characters to overlap between chunks (preserves context)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Each chunk: ~1000 characters
        chunk_overlap=chunk_overlap, # 200 chars overlap for context
        length_function=len, # How to measure length
        separators=["\n\n", "\n", " ", ""] # Split hierarchy
    )
    # Actually split the documents
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show what a chunk looks like
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs


chunks = split_documents(documents)

### embedding and vectorstores

import numpy as np
from sentence_transformers import SentenceTransformer 
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple 
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
                model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model is loaded successfully. Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
                texts: List of text strings to embed
        
        Returns:
                numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not Loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generating embeddings with shape: {embeddings.shape}")
        return embeddings

        def get_embedding_dimension(self) -> int:
            """Get the embedding dimension of the model"""
            if not self.model:
                raise ValueError("Model not loaded")
            return self.model.get_sentence_embedding_dimension()

## initialize the embedding manager
embedding_manager = EmbeddingManager()
embedding_manager

### VectorStore

import os
class VectorStore:
    """Manager document embedding in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "../data/vector_store"):
        """
        Initializing the vector store

        Args:
                collection_name: Name of the ChromaDB collection
                persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistant ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG",
                "hnsw:space": "cosine"}
            )
            print(f"Vector store initalized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
                documents: List of LangChain documents
                embeddings: Corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the embeddings")
        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document Content
            documents_text.append(doc.page_content)

            #Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text)
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding doucments to vector store: {e}")
            raise

vectorstore = VectorStore()
vectorstore


### Convert the texts to embeddings
texts=[doc.page_content for doc in chunks]

### Generate the Embeddings

embeddings = embedding_manager.generate_embeddings(texts)

### Store into the vecotor database
vectorstore.add_documents(chunks, embeddings)


### Retriever 
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the reteiver
        
        Args:
            vecotor_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vecotor_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimu, similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")

        ### Generating query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in the vector
        try:
            results = self.vecotor_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i+1
                        })
                    print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever = RAGRetriever(vectorstore, embedding_manager)


rag_retriever

### Prompt to find the source document tester\

results = rag_retriever.retrieve("Who is Andrea Liss?")

for result in results:
    print("\n=================================")
    print("ID:", result["id"])
    print("Similarity Score:", result["similarity_score"])
    print("Metadata:", result["metadata"])
    print("Content:")
    print(result["content"][:500])


from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
load_dotenv()

### Initialize LLM

llm = ChatOllama(model="llama3.2", temperature=0.3)

## 2. Simple RAG function: retrieve context + generate response

def rag_simple(query, reteiver, llm, top_k=3):
    ### Retrieve the context
    results = reteiver.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question"

    ### Generate the answer using the llm
    prompt=f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}
        
        Answer:"""
    
    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content



answer=rag_simple("How to deal with anxiety as a young person?", rag_retriever,llm)
print(answer)
