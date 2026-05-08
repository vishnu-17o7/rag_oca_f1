"""
RAG Pipeline for F1/FIA Regulations
Uses LEANN for vector storage/retrieval, LangChain for orchestration,
and OpenRouter/Ollama for LLM generation. LangSmith traces all steps
when LANGSMITH_TRACING=true is set in the environment.
"""

import glob
import os
import logging
import shutil
import importlib
import time
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable

# Suppress noisy LangSmith client warnings on auth failures
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from leann import LeannBuilder
from src.leann_retriever import LeannLangChainRetriever

if importlib.util.find_spec("langchain_ollama") is not None:
    ChatOllama = importlib.import_module("langchain_ollama").ChatOllama
else:
    ChatOllama = importlib.import_module("langchain_community.chat_models").ChatOllama

LEANN_INDEX_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leann_index")


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
        self._print_config_banner(chunk_size, chunk_overlap, top_k, temperature)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature

        index_dir = f"db_{self.chunk_size}_{self.chunk_overlap}"
        self.index_path = os.path.join(LEANN_INDEX_ROOT, index_dir)

        print("[STEP 1/5] Loading or building LEANN index...")
        self._load_or_build_index()

        print("[STEP 2/5] Creating LEANN retriever...")
        self.retriever = LeannLangChainRetriever(
            index_path=self.index_path, top_k=self.top_k
        )

        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        print(f"[STEP 3/5] Initializing LLM (provider: {provider})...")
        self._init_llm(provider)
        print(f"  -> Using model: {self.model_name}")

        print("[STEP 4/5] Building RAG query chain...")
        self._build_chain()

        print("[INIT] RAG Pipeline ready!\n")

    @staticmethod
    def _print_config_banner(chunk_size, chunk_overlap, top_k, temperature):
        """Print a detailed configuration banner showing all runtime choices."""
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        openrouter_emb_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        ollama_models = os.getenv("OLLAMA_MODEL_PRIORITY", "tinyllama,qwen3.5:0.8b,phi3")
        openrouter_llm = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
        langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "rag-oca-f1")
        startup_mode = os.getenv("RAG_STARTUP_MODE", "lazy")

        print("\n" + "=" * 62)
        print("  RAG PIPELINE CONFIGURATION")
        print("=" * 62)
        print(f"  Chunk size:          {chunk_size}")
        print(f"  Chunk overlap:       {chunk_overlap}")
        print(f"  Top-K retrieved:     {top_k}")
        print(f"  LLM temperature:     {temperature}")
        print("-" * 62)
        print(f"  Embedding provider:  {'OpenRouter' if openrouter_key else 'local (SentenceTransformer)'}")
        if openrouter_key:
            print(f"  Embedding model:     {openrouter_emb_model}")
            print(f"  Embedding API key:   sk-or-v1-...{openrouter_key[-4:] if len(openrouter_key) > 4 else '???'}")
        else:
            print(f"  Embedding model:     BAAI/bge-small-en-v1.5 (local)")
        print("-" * 62)
        print(f"  LLM provider:        {llm_provider}")
        if llm_provider == "openrouter":
            print(f"  LLM model:           {openrouter_llm}")
            print(f"  LLM API key:         sk-or-v1-...{openrouter_key[-4:] if len(openrouter_key) > 4 else '???'}")
        else:
            print(f"  LLM model priority:  {ollama_models}")
        print("-" * 62)
        print(f"  LangSmith tracing:   {'ON' if langsmith_key else 'OFF'}")
        if langsmith_key:
            print(f"  LangSmith project:   {langsmith_project}")
            print(f"  LangSmith API key:   ls_...{langsmith_key[-4:] if len(langsmith_key) > 4 else '???'}")
        print(f"  Startup mode:        {startup_mode}")
        print("=" * 62 + "\n")

    def _load_documents(self) -> List:
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, "data")
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {data_dir}")

        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)

        return documents

    @traceable(run_type="chain")
    def _build_index(self) -> None:
        print("  -> Loading PDF documents...")
        documents = self._load_documents()
        print(f"  -> Loaded {len(documents)} document pages")

        print("  -> Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "  ", " ", ""],
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"  -> Created {len(self.chunks)} chunks")

        print(f"  -> Building LEANN index at {self.index_path}...")
        os.makedirs(self.index_path, exist_ok=True)

        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            embedding_model = os.getenv(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large"
            )
            print(f"  -> Using OpenRouter embeddings: {embedding_model}")
            builder = LeannBuilder(
                "hnsw",
                embedding_model=embedding_model,
                embedding_mode="openai",
                embedding_options={
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": openrouter_key,
                },
                is_recompute=False,
            )
        else:
            embedding_model = "BAAI/bge-small-en-v1.5"
            print(f"  -> Using local embeddings: {embedding_model}")
            builder = LeannBuilder(
                "hnsw",
                embedding_model=embedding_model,
                is_recompute=False,
            )

        for i, chunk in enumerate(self.chunks):
            builder.add_text(chunk.page_content, metadata={"id": str(i)})

        builder.build_index(self.index_path)
        print(f"  -> LEANN index built with {len(self.chunks)} documents")

    def _load_or_build_index(self) -> None:
        if os.path.isdir(self.index_path) and os.listdir(self.index_path):
            print(f"  -> Loading existing LEANN index from {self.index_path}")
            self.chunks = []
        else:
            if os.path.isdir(self.index_path):
                print("  -> Index directory exists but is empty; rebuilding.")
                shutil.rmtree(self.index_path)
            else:
                print("  -> No existing index found; building from PDFs.")
            self._build_index()

    def _init_llm(self, provider: str) -> None:
        if provider == "openrouter":
            self._init_openrouter()
        else:
            self._init_ollama()

    def _init_ollama(self) -> None:
        model_priority = os.getenv(
            "OLLAMA_MODEL_PRIORITY", "tinyllama,qwen3.5:0.8b,phi3"
        )
        model_candidates = [m.strip() for m in model_priority.split(",") if m.strip()]

        for model_name in model_candidates:
            try:
                self.llm = ChatOllama(
                    model=model_name,
                    temperature=self.temperature,
                )
                print(f"  -> Warming up model '{model_name}'...")
                self.llm.invoke("Hello")
                self.model_name = model_name
                return
            except Exception as e:
                print(f"[WARN] Model {model_name} not available: {e}")
                continue

        raise RuntimeError(
            "No Ollama model available. Please pull phi3 or tinyllama."
        )

    def _init_openrouter(self) -> None:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is required when "
                "LLM_PROVIDER=openrouter. Get your key at https://openrouter.ai/keys"
            )

        self.llm = ChatOpenAI(
            model=model,
            temperature=self.temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model
        print(f"  -> Using OpenRouter model: {model}")

    def _build_chain(self) -> None:
        system_prompt = """You are an expert in FIA Formula 1 regulations. 
Answer the question strictly based on the provided context. Be precise and cite 
regulation numbers where possible. If the answer is not in the context, say 
'Not found in regulations.'"""

        prompt_template = """Context:
{context}

Question: {question}

Answer:"""

        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(prompt_template)
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_last_chunks(self) -> list:
        return getattr(self, "_last_chunks", [])

    @traceable(run_type="chain")
    def query(self, question: str) -> str:
        start_time = time.time()

        print(f"\n[QUERY START] Question: {question}")
        print(f"  -> Retrieving top-{self.top_k} documents...")

        try:
            retr_start = time.time()
            results = self.retriever._searcher.search(question, k=self.top_k)
            retr_end = time.time()
            print(f"  -> [STAGE 1/3] Retrieval complete. Documents found: {len(results)}")
            print(f"  -> [TIMING] Vector search took {retr_end - retr_start:.3f}s")

            self._last_chunks = [
                {
                    "text": res.text,
                    "score": getattr(res, "distance", getattr(res, "score", 0.0)),
                    "source": f"chunk-{res.id}" if hasattr(res, "id") else "LEANN",
                }
                for res in results
            ]
        except Exception as e:
            print(f"  -> [STAGE 1/3 ERROR] Retrieval failed: {e}")
            self._last_chunks = []
            return "Not found in regulations."

        if not results:
            print("  -> No retrieved context chunks. Skipping LLM invocation.")
            print(f"  -> [TOTAL TIMING] Query finished in {time.time() - start_time:.3f}s")
            print("[QUERY END]\n")
            return "Not found in regulations."

        try:
            print(f"  -> [STAGE 2/3] Invoking LLM chain (model: {self.model_name})...")
            llm_start = time.time()

            response = self.chain.invoke(question)
            llm_end = time.time()
            print(f"  -> [TIMING] Chain invocation took {llm_end - llm_start:.3f}s")

            print(f"  -> [STAGE 3/3] LLM response received.")
            print(f"  -> [TOTAL TIMING] Query finished in {time.time() - start_time:.3f}s")
            print("[QUERY END]\n")
            return response
        except Exception as e:
            print(f"  -> [ERROR] Query failed at LLM/Chain stage: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating response."
