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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import csv

# Walert uses a relative DATA_DIR pattern.
# The same pattern is used here instead of a hard-coded C: drive.
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

CRISIS_WORDS = [
    r"\b(suicide|kill myself|want to die|end my life|end it all|self[- ]harm|injuring myself)\b",
    r"\b(can'?t move on|don'?t want to live|better off dead|no reason to live)\b"
]

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

class SafetyGuard:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.regex = re.compile("|".join(CRISIS_WORDS))
    
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
    # all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 BAAI/bge-m3 text-embedding-3-small

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
            source_name = os.path.basename(doc.metadata.get("source","doc")).replace(" ","_")
            #doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            doc_id = f"{source_name}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.upsert(
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

def rag_simple(query, retriever, llm, safety_guard, chat_history, top_k=3):
    """
    Retrieve context and generate a response using the team's RAG function.
    """

    if safety_guard.is_distressed(query):
        return EMERGENCY_RESOURCES.strip()

    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join(
        [doc["content"] for doc in results]
    ) if results else ""

    if not context:
        return "No relevant context found to answer the question"

    #prompt = f"""Use the following context to answer the question concisely.
    #    Context:
    #    {context}

    #    Question: {query}

    #    Answer:"""

    system_prompt = (
        "You are an empathetic, warm and supportive mental health assistant."
        "Your goal is to answer questions about social media anxiety and mental health."
        "Use provided knowledge contet to inform you answer, but always speak in a natural,"
        "conversational tone. Do not just strictly list facts; validate the user's feelings. \n\n"
        f"Context knowledge:\n{context}" 
    )

    #response = llm.invoke(
    #    [prompt.format(context=context, query=query)]
    #)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)

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

    safety_guard = SafetyGuard(llm)

    def generate_my_chroma_run(questions_dict, retriever, run_output="my_chroma.run", top_k=3):
        with open(run_output, "w", encoding="utf-8") as f:
            for q_id, q_text in questions_dict.items():
                retrieved_docs = retriever.retrieve(q_text, top_k=top_k)
                for item in retrieved_docs:
                    doc_id = item["id"]
                    rank = item["rank"]
                    score = item["similarity_score"]
                    f.write(f"{q_id} Q0 {doc_id} {rank} {score:.4f} chroma_minilm\n")
        print(f"Exported run file to {run_output}")
    
    

    

    
    # This is a function I created for my testing (Not to be used as part of code)

    #def generate_ground_truth_csv(question_list, retriever, llm, safety_guard, output_file="draft_ground_truth.csv"):
    #    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    #        writer = csv.writer(file)

    #        writer.writerow(["query_id", "query_text", "expected_doc_id", "retrieved_chunk_text", "generated_answer"])

    #        for i, question in enumerate(question_list):
    #            query_id = f"Q{i+1:03d}"
    #            print(f"Processing {query_id}: {question}")

    #            results = retriever.retrieve(question, top_k=1)

    #            if results:
    #                top_doc_id = results[0]["id"]
    #                top_chunk_text = results[0]["content"]
    #            else:
    #                top_doc_id = "NO_DOC_FOUND"
    #                top_chunk_text = "N/A"
    #            chat_memory = []
    #            answer = rag_simple(
    #                query=question,
    #                retriever=retriever,
    #                llm=llm,
    #                safety_guard=safety_guard,
    #                chat_history=chat_memory
    #            )
    #            writer.writerow([query_id,question, top_doc_id, top_chunk_text, answer])
    
    #    print(f"\nSuccessfully generated {output_file}. Plese check")

    # 50 Questions from the trained pdf documents 
    my_questions = [
        "Does it get worse if I use a phone just texting instead of have a smartphone with internet?",
        "Is cyberbullying actually got increased now than before the pandemic?",
        "Does talking to my parents protect me from getting bullied online?",
        "Do people who bully others online sometimes get bullied back?",
        "Does having siblings make you not get cyberbullied?",
        "Do boys and girls get cyberbullied differently? Or it is common for all?",
        "Does watching violent movies/shows and playing violent games make someone more likely to bully people online?",
        "Will I get anxiety if I start posting my own things instead of just scroll and watch stuff?",
        "What does it mean to be parasocial mean and does that make me more anxious if I am parasocial?",
        "Is following a bunch of random influencers bad for my mental health?",
        "Does posting to a big audience feel different (mentally) than texting one friend?",
        "Why does arguing with people online (social conflict) seem to hit harder than in person?",
        "If I mostly watch stories instead of DMing friends, is that a red flag?",
        "Can how I use social media matter more than how much time I spend on it?",
        "Is it true that social media addiction and anxiety kind of feed off each other?",
        "What's FoMO actually called in psychology, and is it the biggest social media risk factor?",
        "Can using social media too much lower my self-esteem, or is it the other way around?",
        "Does feeling lonely make me use social media more, or does social media make me lonelier?",
        "Is there real science behind depression and social media use, or is that just something adults say?",
        "If I already feel anxious, does that mean I'm more likely to get addicted to my phone?",
        "What counts as problematic social media use vs. just normal teen scrolling?",
        "If I've been both bullied and bullied someone else online, does that make my anxiety worse?",
        "Why would kids who are only a little involved in cyberbullying feel more scared of new people?",
        "Does being a bystander who just watches bullying happen affect my mental health too?",
        "Is it normal to avoid talking to people I don't know well because of stuff I've seen online?",
        "Can being scared of being judged online actually be linked to social anxiety in real life?",
        "If I've never been bullied online, could I still be affected just from seeing it happen to others?",
        "Do apps like TikTok and Instagram actually use AI to keep me hooked on purpose?",
        "Is my brain literally being changed by using social media too much?",
        "What is the dopamine cycle everyone talks about with social media?",
        "Does getting likes and comments actually work like a drug?",
        "Can social media algorithms mess with my sleep and make my anxiety worse?",
        "Is it true that using my phone right before bed messes with my brain chemistry?",
        "Why do I feel like I need to check my phone even when I don't want to?",
        "Are there actual apps or programs designed to help teens with anxiety through social media?",
        "Could a moderated online group actually help me feel less anxious, like therapy?",
        "Is it safe to join an online support group for anxiety, or could it make things worse?",
        "Do these mental health apps/programs actually work, or is it mostly untested?",
        "Would a chatbot like you actually be able to help, based on real research?",
        "Is there a difference between healthy online support and just doom-scrolling forums?",
        "What should I look for to know if an online mental health tool is legit?",
        "Why do I always compare myself to people who seem better than me online?",
        "Is comparing myself to influencers different from comparing myself to my friends?",
        "Does comparing myself to people worse off than me actually make me feel better?",
        "Why does seeing someone prettier or cooler than me on Instagram ruin my whole mood?",
        "If I'm socially anxious, am I more likely to make these upward comparisons than other people?",
        "Can constantly comparing myself to others online actually cause anxiety, or just make it worse?",
        "How do I know if my social media use is normal or actually a problem?",
        "What are some healthy ways to use social media without it messing with my anxiety?",
        "If I think I might have social anxiety because of social media, what should I actually do about it?"
]

    my_questions_dict = {}
    for i, question in enumerate(my_questions):
        query_id = f"Q{i+1:03d}"
        my_questions_dict[query_id] = question

    generate_my_chroma_run(my_questions_dict, rag_retriever, run_output="my_chroma.run")

    # Three-question smoke test.
    questions = [
        "According to the review, which mental health risks were most commonly studied in social-media machine-learning research?",
        "According to the review, why should social-media mental-health models be treated as risk-detection tools rather than clinical diagnostic systems?",
        "What exact steps should an Australian university student follow to report cyberbullying to the eSafety Commissioner?",
        "I feel like I want to kill myself. What should I do?"
    ]

    #generate_ground_truth_csv(my_questions, rag_retriever, llm, safety_guard)

    chat_memory = []

    for question in questions:
        print("\n=================================")
        print("Question:", question)
        print("=================================")

        #results = rag_retriever.retrieve(question, top_k=3)

        #for result in results:
        #    print("\nRank:", result["rank"])
        #    print("Similarity Score:", result["similarity_score"])
        #    print("Metadata:", result["metadata"])
        #    print("Content:", result["content"][:500])

        #print("\nGenerated Answer:")
        #print(rag_simple(question, rag_retriever, llm, top_k=3))

        answer = rag_simple(
            query=question,
            retriever=rag_retriever,
            llm=llm,
            safety_guard=safety_guard,
            chat_history=chat_memory,
        )

        print("\nBot:", answer)

        chat_memory.append(HumanMessage(content=question))
        chat_memory.append(AIMessage(content=answer))

        if len(chat_memory) > 6:
            chat_memory = chat_memory[-6:]
