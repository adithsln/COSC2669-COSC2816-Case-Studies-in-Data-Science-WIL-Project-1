import os
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import chromadb


DATA_DIR = "data"
TEXT_DIR = DATA_DIR + "/txt_file"
PDF_DIR = DATA_DIR + "/pdf_file"
VECTOR_DIR = DATA_DIR + "/vector_store"

def split_documents(documents, chunk_size=1500, chunk_overlap=200):
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
    return split_docs

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = VECTOR_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings", "hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the embeddings")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
            doc_id = f"doc{content_hash}_{i}"

            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )
        print(f"Tool unique chunks are store in collection: {self.collection.count()}")

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score
                    })
        return retrieved_docs

def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answe the question"

    prompt = f"""You are a friendly assistant, answer the question using ONLY the provided Context. Do not add external knowledge, advice, example or conversational filler. if context does not contain the exact answer, reply 'I do not have enough information'.
    Context:
    {context}

    Question: {query}
    Answer:"""

    response = llm.invoke(prompt)
    return response.content

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot_product/(norm_a * norm_b)) if (norm_a * norm_b) != 0 else 0.0

def evaluate_faithfulness(llm, answer: str, context: str) -> float:
    prompt = f"""Context:
    {context}
    
    Statement:
    {answer}
    
    Task: Is every factual claim in the statement supported by the context above?
    Respond with '1' if supported, or '0' if unsupported claims are present. Output only single digit:"""

    resp = llm.invoke(prompt).content.strip()
    return 1.0 if "1" in resp else 0.0

def evaluate_context_precision(llm, question: str, retrieved_contexts: List[str]) -> float:
    if not retrieved_contexts:
        return 0.0
    scores = []
    for ctx in retrieved_contexts:
        prompt = f"""Question: {question}
        Excerpt: {ctx}
        
        Task: Does this excerpt contain information relevant to answering the question?
        Respond with '1' for yes, '0' for no. Output only the single digit:"""
        resp = llm.invoke(prompt).content.strip()
        scores.append(1.0 if "1" in resp else 0.0)
    return float(np.mean(scores))

if __name__ == "__main__":
    load_dotenv()
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    text_loader = DirectoryLoader(TEXT_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})
    documents = text_loader.load()

    pdf_loader = DirectoryLoader(
    PDF_DIR,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False)

    pdf_documents = pdf_loader.load()
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
        temperature=0.0,
    )

    questions = [
    "I'm an RMIT University student and I am expreienceing severe anxiety and it is late at night. Who can I contact for urgent help?",
    "I'm an Monash University student and I am expreienceing severe anxiety from a social media post interaction. What should I do?",
    "Suggest me some ways to get my mind of social media?",
    "Someone is sending me hateful online post to humiliate me. Should I give retaliate? who can I contact in Monash University?",
    "I am current University of melbourne student, I am experiencing immediet mental health crisis from a social media post. Is there someone I can talk in the University during night time?",
    "I am 24 years old studying in RMIT University, which external services can I contact?",
    "I am a student, How much does it cost to attend wellbeing workshop at University of Melbourne CAPS?",
    "I am a RMIT UP student. Can I call Benestar phone counselling service?",
    "Social Media FOMO is making me anxious. What are some specific apps or features I can use to limit my screen time?"
    ]
    ground_truths = [
        "For urgent assistance outside business gours, contact RMIT Urgent mental health support by calling 1300 305 737 or texting 0488 884 162.",
        "You should not attempt to retaliate. You should block the bully, report the behavior to the safety center of the social media platform, and speak with a trusted adult. For support, you can contact the Monash Safer Community Unit, the Monash Counselling Service, or the 24/7 Benestar phone counselling service at 1300 788 336.",
        "Move apps away from your home screen into folders, or delete them entirely. Use apps or features like iPhone's 'Screen Time' or 'Forest' to set timelimits. Spend atleast an hour a week on a screen-free hobby, such as a yoga class, reading a book, or planting a garden. Leave your phone in your bag during dinner with friends. Leave your phone on the charger outside your bedroom.",
        "You should not try to retaliate. Instead you should block the bully, report the behavior to the platform's safety center, and speak with trusted adults. For support you should call 24/7 Benestar phone counselling service on 1300 788 336",
        "Yes, you can access the urgent mental health support available 24/7 through the university dedicated Mental Health Crisis Support Service.",
        "RMIT recommends Headspace, which provides free online and telephone support to young people aged 12 to 25",
        "The Counselling and Psychological Services (CAPS) provides students with free support, including wellbeing workshops.",
        "RMIT UP students who require conselling supports should contact RMIT UP Student Wellbeing at wellbeing@rmit.edu.au or call 03 9657 5800",
        "You can use apps like Forestm which grows a digital plabt while you stay off your phone, or Daywise to schedule notifications. You can also use Do Not Disturb feature, set daily reminder on Instagramm or turn 'Screen Time' in iPhone settings."
    ]

    results_data = []
    print("\nRunning RAG pipeline & local metric evaluation...")

    for q, gt in zip(questions, ground_truths):
        retrieved_docs = rag_retriever.retrieve(q, top_k=3)
        contexts = [doc["content"] for doc in retrieved_docs]
        joined_context = "\n\n".join(contexts)
        answer = rag_simple(q, rag_retriever, llm, top_k=3)

        faith_score = evaluate_faithfulness(llm, answer, joined_context)

        q_emb = embedding_manager.generate_embeddings([q])[0]
        ans_emb = embedding_manager.generate_embeddings([answer])[0]
        relevancy_socre = calculate_cosine_similarity(q_emb, ans_emb)

        precision_score = evaluate_context_precision(llm, q, contexts)

        gt_emb = embedding_manager.generate_embeddings([gt])[0]
        ctx_emb = embedding_manager.generate_embeddings([joined_context])[0]
        recall_score = calculate_cosine_similarity(gt_emb, ctx_emb)

        results_data.append({
            "question":q,
            "response":answer,
            "faithfulness": faith_score,
            "answer_relevancy": round(relevancy_socre, 4),
            "context_precision": round(precision_score, 4),
            "context_recall": round(recall_score, 4)
            })
    
    eval_df = pd.DataFrame(results_data)
    eval_df.to_csv("RAGAS_METRIC_CLEAN.csv", index=False)

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")
    print(eval_df[["question","faithfulness","answer_relevancy","context_precision","context_recall"]].to_string())