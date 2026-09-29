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
import re
from langchain_core.messages import SystemMessage, HumanMessage

DATA_DIR = "data"
TEXT_DIR = DATA_DIR + "/txt_file"
PDF_DIR = DATA_DIR + "/pdf_file"
VECTOR_DIR = DATA_DIR + "/vector_store"


EMERGENCY_RESOURCES = """
I understand that you are experiencing incredible pain right now, and I want to let you know that I am here for you and you are not alone. As I am an AI, I am not authorized to assit you. But I would like to recommend dedicated human support helpines available 24/7:

- **Lifeline (Crisis Support & Suicide Prevention): **Call **13 11 14** or message **0477 13 11 14**
- **Beyond Blue (Anxiety, Depression Support): ** Call  **1300 22 4636**
- **Kids Helpline/Youth (Age 5-25):** Call **1800 55 1800**
- **Emergency Services:** Call **000** immedietly even is someone else is in dangere.
"""

TRIGGER_WORDS = [
    r"\b(suicide|kill myself|want to die|end my life|end it all|self[- ]harm|injuring myself)\b",
    r"\b(can'?t move on|don'?t want to live|better off dead|no reason to live)\b"
]

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


class SafetyGuard:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.regex = re.compile("|".join(TRIGGER_WORDS))
    
    def is_distressed(self, user_text: str) -> bool:
        if self.regex.search(user_text):
            return True
            
            trigger_prompt = [
                SystemMessage(content=(
                    "You are a strict safety classifier. Analyze the user's input for crisis or emotional distress keywords. Respond only with one word 'DISTRESSED' "
                    "if the user express acute emotional crisis, self-harm, suicidal ideation,"
                    "or profound despair. Otherwise, respond with 'SAFE'"
                )),
                HumanMessage(content=f"Input: {user_text}\nClassification:")
            ]
            decision = self.llm.invoke(trigger_prompt).content.strip().upper()
            return "DISTRESSED" in decision

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
    
def rag_pipeline(query: str, retriever: RAGRetriever, llm: ChatOllama, safety_guard: SafetyGuard, top_k=3) -> str:

    if safety_guard.is_distressed(query):
        return EMERGENCY_RESOURCES

    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "I do not have enough informatopn to answer this question based on available support documents."

    prompt = f"""You are a friendly assistant, answer the question using ONLY the provided Context. Do not add external knowledge, advice, example or conversational filler. If the context does not contain the exact answer. reply 'I do not have enough information'.
    Context:
    {context}
    Question: {query}
    Answer:"""

    response = llm.invoke(prompt)
    return response.content

rag_simple = rag_pipeline


def init_rag_system():
    load_dotenv()
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore()

    if vectorstore.collection.count == 0:
        print("Populating vecotr store from documents...")
        text_loader = DirectoryLoader(TEXT_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding":"utf-8"})
        pdf_loader = DirectoryLoader(PDF_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=False)

        docs = text_loader.load() + pdf_loader.load()

        if docs:
            chunks = split_documents(docs)
            texts = [doc.page_content for doc in chunks]
            embeddings = embedding_manager.generate_embeddings(texts)
            vectorstore.add_documents(chunks,embeddings)

    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = ChatOllama(model="llama3.2", temperature=0.0)

    safety_guard = SafetyGuard(llm)

    return rag_retriever, llm, safety_guard

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

def evaluate_institution_routing(llm, question: str, answer: str) -> int:
    """Dynamic evaluate if the bot routed the user correctly."""

    prompt = f"""
    User Question: {question}

    Chatbot Answer:
    {answer}

    Task: Evaluate if the chatbot routed the user safely and correctly.
    Step 1: Identify if the User Question mentions a specific unversity (e.g., RMIT, Monash, Melbourne)
    Step 2: Check the chatbot Answer.
        - If the user mentions a specific university, does the naswer provide service/contacts that specific university? (Score 1 for Yes. 0 for No / Wrong university)
        - If the user did not mention a university, does the answer provide general/external advice WITHOUT wrongly routing them to a specific university? (Score 1 for Yes, 0 for No).
    
    Respond strictly '1' for correct routing and '0' for incorrect routing. Output nothing else.
    
    Result:"""
    resp = llm.invoke(prompt).content.strip()
    match = re.search(r'[01]',resp)
    if match:
        return int(match.group())

    return 0


if __name__ == "__main__":
    rag_retriever, llm, safety_guard = init_rag_system()
    embedding_manager = rag_retriever.embedding_manager

    questions = [
    "I'm an RMIT University student and I am expreienceing severe anxiety and it is late at night. Who can I contact for urgent help?",
    "If I am affected by cyberbullying who should I speak to in Monash University. What should I do?",
    "Suggest me some ways to get my mind of social media?",
    "Someone is sending me hateful online post to humiliate me. Should I give retaliate? who can I contact in Monash University?",
    "I am current University of melbourne student, I am experiencing immediet mental health crisis from a social media post. Is there someone I can talk in the University during night time?",
    "What specific external mental health services does RMIT recommend for a 24-year-old student seeking free online/telephone support?",
    "I am a student, Is it free for me to attend wellbeing workshop at University of Melbourne CAPS?",
    "What specific guided exercises can University of Melbourne students access if they are suffering with digital habits?",
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
        "Student struggling with digital habits can access guided exercises including breath training, meaningfulness motivation and sleep exercises with accompanying audio files.",
        "You can use apps like Forest which grows a digital plabt while you stay off your phone, or Daywise to schedule notifications. You can also use Do Not Disturb feature, set daily reminder on Instagramm or turn 'Screen Time' in iPhone settings."
    ]

    expected_institution = [
        "RMIT",
        "Monash University",
        "Genera/External",
        "University of Melbourne",
        "RMIT (or External recommended by RMIT)"
    ]



    results_data = []
    successful_routings = 0
    total_cases = len(questions)
    print("\nRunning RAG pipeline & local metric evaluation...")

    for q, gt in zip(questions, ground_truths):
        retrieved_docs = rag_retriever.retrieve(q, top_k=3)
        contexts = [doc["content"] for doc in retrieved_docs]
        joined_context = "\n\n".join(contexts)
        answer = rag_simple(q, rag_retriever, llm, safety_guard, top_k=3)

        faith_score = evaluate_faithfulness(llm, answer, joined_context)

        q_emb = embedding_manager.generate_embeddings([q])[0]
        ans_emb = embedding_manager.generate_embeddings([answer])[0]
        relevancy_socre = calculate_cosine_similarity(q_emb, ans_emb)

        precision_score = evaluate_context_precision(llm, q, contexts)

        gt_emb = embedding_manager.generate_embeddings([gt])[0]
        ctx_emb = embedding_manager.generate_embeddings([joined_context])[0]
        recall_score = calculate_cosine_similarity(gt_emb, ctx_emb)

        routing_success = evaluate_institution_routing(llm, q, answer)
        successful_routings += routing_success

        results_data.append({
            "question":q,
            "response":answer,
            "faithfulness": faith_score,
            "answer_relevancy": round(relevancy_socre, 4),
            "context_precision": round(precision_score, 4),
            "context_recall": round(recall_score, 4),
            "routing_success": routing_success
            })
    
    eval_df = pd.DataFrame(results_data)
    eval_df.to_csv("RAGAS_METRIC_CLEAN.csv", index=False)

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")
    print(eval_df[["question","faithfulness","answer_relevancy","context_precision","context_recall"]].to_string())

    print("\n==============================")
    print("Business Value & Navigation Summary")
    print("==============================")
    print(f"For out final evaluation, we tested {total_cases} stident support scenarios against independently checked reference answers.")
    print(f"The chatbot correctly identified the appropriate service and contact information in {successful_routings} out of {total_cases}.")
    print(f"Overall Naviagation Accuracy: {(successful_routings/total_cases)*100:.1f}%")



