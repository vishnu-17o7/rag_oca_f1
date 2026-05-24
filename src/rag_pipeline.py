"""
RAG Pipeline for F1/FIA Regulations
Loads PDF documents, builds vector store, and queries via Groq
"""

import glob
import os
import shutil
import sys
import time
import torch
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_groq import ChatGroq


def _ts():
    """Return a wall-clock timestamp string for debug logs."""
    return time.strftime("%H:%M:%S", time.localtime())

def _log(msg):
    """Print with timestamp and immediate flush for HF Space log collector."""
    _log(f" {msg}")
    sys.stdout.flush()


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
        _log("[INIT] Creating RAG Pipeline...")
        _log(f"  - chunk_size: {chunk_size}")
        _log(f"  - chunk_overlap: {chunk_overlap}")
        _log(f"  - top_k: {top_k}")
        _log(f"  - temperature: {temperature}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        _log(f"[INIT] Attributes assigned. chunk_size={self.chunk_size}")

        # Create embeddings
        _log("[STEP 1/5] Loading embedding model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"  -> Using device mapping: {device}")

        t0 = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        _log(f"[TIME] Embedding model loaded in {time.time() - t0:.1f}s")

        model_short = "bge-base"
        persist_dir = f"./chroma_db/db_{model_short}_{self.chunk_size}_{self.chunk_overlap}"
        _log(f"[STEP 2-3] persist_dir={persist_dir}")
        self.vector_store = self._load_or_create_vector_store(embeddings, persist_dir)

        _log(f"[STEP 3b] Creating retriever (top_k={self.top_k})...")
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )
        _log("[STEP 3b] Retriever created.")

        self._reranker = None

        _log("[STEP 4/5] Initializing Groq LLM...")
        self._init_llm()
        _log(f"  -> Using model: {self.model_name}")

        _log("[STEP 5/5] Building RAG query chain...")
        self._build_chain()
        _log("[INIT] RAG Pipeline ready!\n")

    def _load_documents(self) -> List:
        """Load all PDFs from data and data2 directories (deduplicated by basename)."""
        project_root = os.path.dirname(os.path.dirname(__file__))

        search_dirs = [
            os.path.join(project_root, "data"),
            os.path.join(project_root, "data2"),
        ]

        seen_basenames = set()
        pdf_files = []
        for d in search_dirs:
            label = os.path.basename(d)
            if not os.path.isdir(d):
                _log(f" [LOAD] Directory '{label}/' not found, skipping")
                continue
            _log(f" [LOAD] Scanning '{label}/' directory...")
            found = glob.glob(os.path.join(d, "*.pdf"))
            for f in sorted(found):
                bn = os.path.basename(f)
                if bn in seen_basenames:
                    _log(f" [LOAD]   SKIPPING duplicate: {label}/{bn} (already loaded from other dir)")
                    continue
                seen_basenames.add(bn)
                pdf_files.append(f)
                _log(f" [LOAD]   FOUND: {label}/{bn}")
            _log(f" [LOAD] '{label}/' -> {len(found)} PDFs ({len([f for f in found if os.path.basename(f) not in seen_basenames or f == found[-1]])} new)")

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in data/ or data2/")

        # Load each PDF
        documents = []
        total_start = time.time()
        for pdf_file in pdf_files:
            bn = os.path.basename(pdf_file)
            _log(f" [LOAD] Parsing {bn} ...")
            t0 = time.time()
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            elapsed = time.time() - t0
            documents.extend(docs)
            _log(f" [LOAD]   -> {len(docs)} pages in {elapsed:.1f}s from {bn}")
        _log(f" [LOAD] TOTAL: {len(documents)} pages from {len(pdf_files)} PDFs in {time.time() - total_start:.1f}s")

        return documents

    def _create_vector_store(self, embeddings, persist_dir: str) -> Chroma:
        """Create and persist a fresh vector store from PDFs."""
        # Load and process PDFs
        _log(f" [STEP 2/5] Loading PDF documents...")
        t0 = time.time()
        self.documents = self._load_documents()
        _log(f" [TIME] PDF loading took {time.time() - t0:.1f}s")

        # Create text splitter
        _log(f" [STEP 3/5] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n## ", "\n\n### ", "\n\n# ",
                "\n\nArticle ", "\n\nARTICLE ",
                "\n\n", "\n", ". ", " ", "",
            ],
        )

        # Split documents into chunks
        t1 = time.time()
        self.chunks = text_splitter.split_documents(self.documents)
        _log(f"   -> Created {len(self.chunks)} chunks in {time.time() - t1:.1f}s")

        _log(f"   -> Creating and persisting vector store in {persist_dir}...")
        _log(f"   -> Embedding {len(self.chunks)} chunks with bge-base-en-v1.5 (this is the slow step)...")
        sys.stdout.flush()  # ensure the above is visible before the long embedding call
        t2 = time.time()
        vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name="f1_regulations",
            persist_directory=persist_dir,
        )
        _log(f"   -> Vector store created with {len(self.chunks)} vectors in {time.time() - t2:.1f}s")
        return vector_store

    def _load_or_create_vector_store(self, embeddings, persist_dir: str) -> Chroma:
        """Load persisted vector store if valid, otherwise rebuild from source PDFs."""
        db_path = os.path.join(persist_dir, "chroma.sqlite3")

        if os.path.exists(db_path):
            _log(f" [CACHE] Found existing vector store at {persist_dir}")
            _log(f" [CACHE] Loading Chroma from {persist_dir}...")
            t0 = time.time()
            vector_store = Chroma(
                collection_name="f1_regulations",
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            # Dummy default so external stats calls don't fail.
            self.chunks = []

            try:
                vector_count = vector_store._collection.count()
                _log(f" [CACHE] Existing vector count: {vector_count} (loaded in {time.time() - t0:.1f}s)")
            except Exception as e:
                _log(f" [WARN] Could not read vector count from persisted DB: {e}")
                vector_count = 0

            if vector_count > 0:
                _log(f" [CACHE] Vector store loaded successfully.")
                return vector_store

            _log(f" [WARN] Persisted vector store is empty. Rebuilding index from PDFs...")
            try:
                shutil.rmtree(persist_dir)
                _log(f"   -> Removed stale index directory: {persist_dir}")
            except Exception as e:
                _log(f" [WARN] Could not remove stale index directory: {e}")

        return self._create_vector_store(embeddings, persist_dir)

    def _init_llm(self):
        """Initialize Groq LLM with fallback model."""
        model_priority = os.getenv(
            "GROQ_MODEL_PRIORITY", "llama-3.3-70b-versatile,llama-3.1-8b-instant"
        )
        model_candidates = [m.strip() for m in model_priority.split(",") if m.strip()]
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or set the environment variable."
            )

        for model_name in model_candidates:
            try:
                self.llm = ChatGroq(
                    model=model_name,
                    temperature=self.temperature,
                    api_key=api_key,
                )
                # Warm up once at startup so first user query is fast.
                _log(f"   -> Warming up model '{model_name}'...")
                t0 = time.time()
                self.llm.invoke("Hello")
                _log(f"   -> Model '{model_name}' warmed up in {time.time() - t0:.1f}s")

                self.model_name = model_name
                break
            except Exception as e:
                _log(f" [WARN] Model {model_name} not available: {e}")
                continue
        else:
            raise RuntimeError(
                "No Groq model available. Check GROQ_API_KEY and model names."
            )

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            _log(f"   -> Loading cross-encoder re-ranker (BAAI/bge-reranker-base)...")
            t0 = time.time()
            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
            _log(f"   -> Re-ranker loaded in {time.time() - t0:.1f}s")
        return self._reranker

    def _rerank_docs(self, query: str, docs_with_scores: list) -> list:
        docs = [doc for doc, _ in docs_with_scores]
        if len(docs) <= 1:
            return docs_with_scores

        try:
            reranker = self._get_reranker()
            pairs = [(query, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)

            scored = list(zip(docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(doc, float(score)) for doc, score in scored]
        except Exception as e:
            _log(f"[WARN] Re-ranking failed: {e}. Using original order.")
            return docs_with_scores

    def _build_chain(self):
        """Build the RAG query chain."""
        # System prompt
        system_prompt = """You are an expert in FIA Formula 1 regulations. 
Answer the question based exclusively on the provided context. Be precise, cite 
specific regulation numbers, articles, or clauses when applicable. Synthesize 
information from multiple retrieved passages if they cover different facets of 
the same regulation. If the context does not contain the answer, say 'Not found 
in regulations.' Do not use external knowledge."""

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
            _log(f"[TIME] Document formatting took {time.time() - start:.6f}s")
        return result

    def get_last_chunks(self) -> list:
        """Returns last retrieved chunks as [{text, score, source}]"""
        return getattr(self, "_last_chunks", [])

    def query(self, question: str) -> str:
        start_time = time.time()

        initial_k = min(self.top_k * 2, 15)
        _log(f" [QUERY START] Question: {question}")
        _log(f" [QUERY] Retrieving top-{initial_k} documents (will re-rank to top-{self.top_k})...")

        try:
            retr_start = time.time()
            results = self.vector_store.similarity_search_with_score(
                question, k=initial_k
            )
            retr_end = time.time()
            _log(f" [STAGE 1/5] Initial retrieval complete. Documents found: {len(results)}")
            _log(f" [TIME] Vector search took {retr_end - retr_start:.3f}s")

            if results:
                rerank_start = time.time()
                reranked = self._rerank_docs(question, results)
                top_results = reranked[:self.top_k]
                rerank_end = time.time()
                _log(f" [STAGE 2/5] Re-ranking complete. Kept top-{self.top_k} of {len(reranked)}.")
                _log(f" [TIME] Re-ranking took {rerank_end - rerank_start:.3f}s")
            else:
                top_results = []

            docs = [doc for doc, _ in top_results]

            self._last_chunks = [
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "source": os.path.basename(doc.metadata.get("source", "FIA REGULATIONS")),
                }
                for doc, score in top_results
            ]
        except Exception as e:
            _log(f" [STAGE 1/5 ERROR] Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            self._last_chunks = []
            docs = []

        if not docs:
            _log(f" [STAGE 3/5] No retrieved context chunks. Skipping LLM invocation.")
            _log(f" [TIME] Query finished in {time.time() - start_time:.3f}s")
            _log(f" [QUERY END]\n")
            return "Not found in regulations."

        try:
            _log(f" [STAGE 3/5] Preparing LLM chain input...")
            llm_prep_start = time.time()

            chain_input = {
                "context": docs,
                "question": question,
            }
            _log(f" [TIME] Chain input preparation took {time.time() - llm_prep_start:.6f}s")

            _log(f" [STAGE 4/5] Invoking LLM chain (model: {self.model_name})...")
            llm_invoke_start = time.time()

            response = self.chain.invoke(chain_input)
            llm_invoke_end = time.time()

            _log(f" [STAGE 5/5] LLM response received.")
            _log(f" [TIME] chain.invoke() took {llm_invoke_end - llm_invoke_start:.3f}s")

            result = response.content if hasattr(response, "content") else str(response)
            if not result:
                _log(f" [WARN] LLM returned an empty response string.")

            _log(f" [TIME] Query finished in {time.time() - start_time:.3f}s")
            _log(f" [QUERY END]\n")
            return result
        except Exception as e:
            _log(f" [ERROR] Query failed at LLM/Chain stage: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return "Error generating response."
