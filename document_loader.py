from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import numpy as np
import os
import uuid
from typing import List, Dict, Any

# Walert uses a relative DATA_DIR pattern.
# The same pattern is used here instead of a hard-coded C: drive.
DATA_DIR = "data"
TEXT_DIR = DATA_DIR + "/txt_file"
PDF_DIR = DATA_DIR + "/pdf_file"
VECTOR_DIR = DATA_DIR + "/vector_store"

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better RAG performance.

    Parameters:
    - chunk_size: Maximum characters per chunk (adjust based on your LLM)
    - chunk_overlap: Characters to overlap between chunks (preserves context)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs

class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                "Model is loaded successfully. Embedding dimensions: "
                f"{self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

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
        """Get the embedding dimension of the model."""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()

class VectorStore:
    """Manager document embedding in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = VECTOR_DIR,
    ):
        """
        Initializing the vector store.

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
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embedding for RAG",
                    "hnsw:space": "cosine",
                },
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

                print(
                    f"Retrieved {len(retrieved_docs)} documents "
                    "(after filtering)"
                )
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

def rag_simple(query, retriever, llm, top_k=3):
    """
    Retrieve context and generate a response using the team's RAG function.
    """
    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [doc["content"] for doc in results]
    ) if results else ""

    if not context:
        return "No relevant context found to answer the question"

    prompt = f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""

    response = llm.invoke(
        [prompt.format(context=context, query=query)]
    )

    return response.content

# Walert uses the same execution-guard structure.
if __name__ == "__main__":

    load_dotenv()

    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    # TXT loading — current team repository approach.
    text_loader = DirectoryLoader(
        TEXT_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )

    documents = text_loader.load()

    # PDF loading — current team repository approach.
    pdf_loader = DirectoryLoader(
        PDF_DIR,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False,
    )

    pdf_documents = pdf_loader.load()

    # Combine TXT and PDF documents using the append() pattern available in Walert.
    for pdf_document in pdf_documents:
        documents.append(pdf_document)

    # Create data chunks.
    chunks = split_documents(documents)

    # Initialize embedding manager.
    embedding_manager = EmbeddingManager()

    # Initialize vector store.
    vectorstore = VectorStore()

    # Convert chunk text to embeddings.
    texts = [doc.page_content for doc in chunks]

    # Generate embeddings.
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store embeddings in ChromaDB.
    vectorstore.add_documents(chunks, embeddings)

    # Initialize retriever.
    rag_retriever = RAGRetriever(
        vectorstore,
        embedding_manager,
    )

    # Initialize Ollama LLM.
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
    )

    # Three-question smoke test.
    questions = [
        "According to the review, which mental health risks were most commonly studied in social-media machine-learning research?",
        "According to the review, why should social-media mental-health models be treated as risk-detection tools rather than clinical diagnostic systems?",
        "What exact steps should an Australian university student follow to report cyberbullying to the eSafety Commissioner?",
    ]

    for question in questions:
        print("\n=================================")
        print("Question:", question)
        print("=================================")

        results = rag_retriever.retrieve(question, top_k=3)

        for result in results:
            print("\nRank:", result["rank"])
            print("Similarity Score:", result["similarity_score"])
            print("Metadata:", result["metadata"])
            print("Content:", result["content"][:500])

        print("\nGenerated Answer:")
        print(rag_simple(question, rag_retriever, llm, top_k=3))