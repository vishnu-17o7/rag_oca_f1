"""
RAG Pipeline for F1/FIA Regulations
Loads PDF documents, builds vector store, and queries with Ollama
"""

import glob
import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


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
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
        )

        persist_dir = f"./chroma_db/db_{self.chunk_size}_{self.chunk_overlap}"
        
        if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            print(f"[STEP 2/5 & 3/5] Loading existing vector store from {persist_dir}...")
            self.vector_store = Chroma(
                collection_name="f1_regulations",
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            # Dummy load so it doesn't crash on .chunks later if queried for stats
            self.chunks = []
            print(f"  -> Vector store loaded successfully.")
        else:
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
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings,
                collection_name="f1_regulations",
                persist_directory=persist_dir
            )
            print(f"  -> Vector store created with {len(self.chunks)} vectors")

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

    def _init_llm(self):
        """Initialize Ollama LLM with fallback model."""
        # Try qwen2.5 first, then fall back to phi3/tinyllama
        for model_name in ["qwen3.5:0.8b", "phi3", "tinyllama"]:
            try:
                self.llm = ChatOllama(
                    model=model_name, temperature=self.temperature, verbose=False
                )
                # Test the model
                self.llm.invoke("test")
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

        # Build chain: retriever -> format docs -> llm
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.chat_prompt
            | self.llm
        )

    @staticmethod
    def _format_docs(docs) -> str:
        """Format retrieved documents into a context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_last_chunks(self) -> list:
        """Returns last retrieved chunks as [{text, score, source}]"""
        return getattr(self, '_last_chunks', [])

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question: The question to ask

        Returns:
            The generated answer string
        """
        print(f"  -> Retrieving top-{self.top_k} documents...")
        
        # Capture structured chunks with scores for the frontend
        try:
            results = self.vector_store.similarity_search_with_score(question, k=self.top_k)
            self._last_chunks = [
                {
                    "text": doc.page_content,
                    "score": float(score),
                    "source": os.path.basename(doc.metadata.get("source", "FIA REGULATIONS"))
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"[WARN] Error capturing doc chunks: {e}")
            self._last_chunks = []

        try:
            response = self.chain.invoke(question)
            result = response.content if hasattr(response, "content") else str(response)
            print(f"  -> LLM response received ({len(result)} chars)")
            return result
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return "Error generating response."
