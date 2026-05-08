"""
Custom LangChain Retriever wrapping LEANN's LeannSearcher.
Ensures every retrieval step is logged as a distinct span in LangSmith.
"""

import os
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from leann import LeannSearcher


class LeannLangChainRetriever(BaseRetriever):
    index_path: str
    top_k: int = 5
    _searcher: LeannSearcher | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, index_path: str, top_k: int = 5, **kwargs):
        super().__init__(index_path=index_path, top_k=top_k, **kwargs)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"LEANN index not found at: {index_path}")
        self._searcher = LeannSearcher(index_path, recompute_embeddings=False)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self._searcher.search(query, k=self.top_k)
        docs = []
        for res in results:
            docs.append(
                Document(
                    page_content=res.text,
                    metadata={
                        "source_id": res.id if hasattr(res, "id") else "",
                    },
                )
            )
        return docs
