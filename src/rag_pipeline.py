"""
RAG Pipeline for F1/FIA Regulations
Loads PDF documents, builds vector store, and queries with Ollama
"""

import glob
import os
import shutil
import importlib
import torch
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

if importlib.util.find_spec("langchain_ollama") is not None:
    ChatOllama = importlib.import_module("langchain_ollama").ChatOllama
else:
    ChatOllama = importlib.import_module("langchain_community.chat_models").ChatOllama


class RAGPipeline:
    """
    RAG Pipeline that supports configurable hyperparameters.

    Hyperparameters:
        chunk_size: Number of characters per text chunk
        chunk_overlap: Overlap between consecutive chunks
        top_k: Number of documents to retrieve
        temperature: LLM temperature for generation
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        temperature: float = 0.0,
    ):
        print(f"\n[INIT] Creating RAG Pipeline...")
        print(f"  - chunk_size: {chunk_size}")
        print(f"  - chunk_overlap: {chunk_overlap}")
        print(f"  - top_k: {top_k}")
        print(f"  - temperature: {temperature}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature

        # Create embeddings
        print("[STEP 1/5] Loading embedding model...")

        # Determine optimal device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  -> Using device mapping: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": device}
        )

        persist_dir = f"./chroma_db/db_{self.chunk_size}_{self.chunk_overlap}"
        self.vector_store = self._load_or_create_vector_store(embeddings, persist_dir)

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        # Initialize LLM
        print("[STEP 4/5] Initializing Ollama LLM...")
        self._init_llm()
        print(f"  -> Using model: {self.model_name}")

        # Build query chain
        print("[STEP 5/5] Building RAG query chain...")
        self._build_chain()
        print("[INIT] RAG Pipeline ready!\n")

    def _load_documents(self) -> List:
        """Load all PDFs from data directory."""
        # Find data directory relative to this file
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, "data")

        # Find all PDF files
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {data_dir}")

        # Load each PDF
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)

        return documents

    def _create_vector_store(self, embeddings, persist_dir: str) -> Chroma:
        """Create and persist a fresh vector store from PDFs."""
        # Load and process PDFs
        print("[STEP 2/5] Loading PDF documents...")
        self.documents = self._load_documents()
        print(f"  -> Loaded {len(self.documents)} document pages")

        # Create text splitter
        print("[STEP 3/5] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        # Split documents into chunks
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"  -> Created {len(self.chunks)} chunks")

        print(f"  -> Creating and persisting vector store in {persist_dir}...")
        vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name="f1_regulations",
            persist_directory=persist_dir,
        )
        print(f"  -> Vector store created with {len(self.chunks)} vectors")
        return vector_store

    def _load_or_create_vector_store(self, embeddings, persist_dir: str) -> Chroma:
        """Load persisted vector store if valid, otherwise rebuild from source PDFs."""
        db_path = os.path.join(persist_dir, "chroma.sqlite3")

        if os.path.exists(db_path):
            print(f"[STEP 2/5 & 3/5] Loading existing vector store from {persist_dir}...")
            vector_store = Chroma(
                collection_name="f1_regulations",
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            # Dummy default so external stats calls don't fail.
            self.chunks = []

            try:
                vector_count = vector_store._collection.count()
                print(f"  -> Existing vector count: {vector_count}")
            except Exception as e:
                print(f"[WARN] Could not read vector count from persisted DB: {e}")
                vector_count = 0

            if vector_count > 0:
                print("  -> Vector store loaded successfully.")
                return vector_store

            print("[WARN] Persisted vector store is empty. Rebuilding index from PDFs...")
            try:
                shutil.rmtree(persist_dir)
                print(f"  -> Removed stale index directory: {persist_dir}")
            except Exception as e:
                print(f"[WARN] Could not remove stale index directory: {e}")

        return self._create_vector_store(embeddings, persist_dir)

    def _init_llm(self):
        """Initialize Ollama LLM with fallback model."""
        model_priority = os.getenv(
            "OLLAMA_MODEL_PRIORITY", "tinyllama,qwen3.5:0.8b,phi3"
        )
        model_candidates = [m.strip() for m in model_priority.split(",") if m.strip()]

        for model_name in model_candidates:
            try:
                # Use default parameters for Ollama
                self.llm = ChatOllama(
                    model=model_name,
                    temperature=self.temperature,
                )
                # Warm up once at startup so first user query is fast.
                print(f"  -> Warming up model '{model_name}' (defaults)...")
                self.llm.invoke("Hello")

                self.model_name = model_name
                break
            except Exception as e:
                print(f"[WARN] Model {model_name} not available: {e}")
                continue
        else:
            raise RuntimeError(
                "No Ollama model available. Please pull phi3 or tinyllama."
            )

    def _build_chain(self):
        """Build the RAG query chain."""
        # System prompt
        system_prompt = """You are an expert in FIA Formula 1 regulations. 
Answer the question strictly based on the provided context. Be precise and cite 
regulation numbers where possible. If the answer is not in the context, say 
'Not found in regulations.'"""

        # User prompt template
        prompt_template = """Context:
{context}

Question: {question}

Answer:"""

        # Create prompts
        self.system_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
        self.human_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.human_prompt]
        )

        from langchain_core.runnables import RunnableLambda

        # Build chain: format docs -> prompt -> llm
        self.chain = (
            {
                "context": RunnableLambda(lambda x: x["context"]) | self._format_docs,
                "question": RunnableLambda(lambda x: x["question"]),
            }
            | self.chat_prompt
            | self.llm
        )

    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a context string with timing."""
        import time

        start = time.time()
        result = "\n\n".join(doc.page_content for doc in docs)
        print(f"  -> [TIMING] Document formatting took {time.time() - start:.6f}s")
        return result

    def get_last_chunks(self) -> list:
        """Returns last retrieved chunks as [{text, score, source}]"""
        return getattr(self, "_last_chunks", [])

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question: The question to ask

        Returns:
            The generated answer string
        """
        import time

        start_time = time.time()

        print(f"\n[QUERY START] Question: {question}")
        print(f"  -> Retrieving top-{self.top_k} documents...")

        # Capture structured chunks with scores for the frontend
        try:
            retr_start = time.time()
            results = self.vector_store.similarity_search_with_score(question, k=self.top_k)
            retr_end = time.time()
            print(f"  -> [STAGE 1/4] Retrieval complete. Documents found: {len(results)}")
            print(f"  -> [TIMING] Vector search took {retr_end - retr_start:.3f}s")

            # Extract document objects for the chain
            docs = [doc for doc, score in results]

            self._last_chunks = [
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "source": os.path.basename(doc.metadata.get("source", "FIA REGULATIONS")),
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"  -> [STAGE 1/4 ERROR] Retrieval failed: {e}")
            self._last_chunks = []
            docs = []

        if not docs:
            print("  -> [STAGE 2/4] No retrieved context chunks. Skipping LLM invocation.")
            print(f"  -> [TOTAL TIMING] Query finished in {time.time() - start_time:.3f}s")
            print("[QUERY END]\n")
            return "Not found in regulations."

        try:
            print(f"  -> [STAGE 2/4] Initializing LLM chain components...")
            llm_prep_start = time.time()

            # Pass pre-retrieved docs directly into the chain input
            chain_input = {
                "context": docs,
                "question": question,
            }
            print(f"  -> [TIMING] Chain input preparation took {time.time() - llm_prep_start:.6f}s")

            print(f"  -> [STAGE 3/4] Invoking LLM chain (model: {self.model_name})...")
            llm_invoke_start = time.time()

            response = self.chain.invoke(chain_input)
            llm_invoke_end = time.time()

            print("  -> [STAGE 4/4] LLM response received.")
            print(
                f"  -> [TIMING] actual chain.invoke() execution took {llm_invoke_end - llm_invoke_start:.3f}s"
            )

            result = response.content if hasattr(response, "content") else str(response)
            if not result:
                print("  -> [WARN] LLM returned an empty response string.")

            print(f"  -> [TOTAL TIMING] Query finished in {time.time() - start_time:.3f}s")
            print("[QUERY END]\n")
            return result
        except Exception as e:
            print(f"  -> [ERROR] Query failed at LLM/Chain stage: {e}")
            import traceback

            traceback.print_exc()
            return "Error generating response."
