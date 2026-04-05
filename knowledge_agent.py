"""
knowledge_agent.py — Knowledge Agent (RAG)

Retrieves relevant information from the knowledge base using:
- Semantic similarity search (ChromaDB + OpenAI embeddings)
- Intent-aware query rewriting for better retrieval
- Confidence scoring based on similarity scores
- Result re-ranking for quality
"""

from __future__ import annotations

import hashlib
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeAgent:
    """
    RAG-powered knowledge retrieval agent.
    
    Uses ChromaDB as the vector store with OpenAI embeddings.
    Supports intent-aware query rewriting to improve retrieval quality.
    """

    # Similarity threshold: docs below this score are discarded
    SIMILARITY_THRESHOLD = 0.65

    # Collection names per intent for targeted retrieval
    COLLECTION_MAP = {
        "refund": "policies",
        "order_status": "orders",
        "technical": "technical_docs",
        "billing": "billing",
        "general": "general_faq",
        "account_action": "account_docs",
    }

    def __init__(self):
        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )

        # OpenAI embedding function for ChromaDB
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )

        # LLM for query rewriting
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
        )

        # Ensure default collections exist
        self._initialize_collections()

    def _initialize_collections(self):
        """Create ChromaDB collections if they don't exist."""
        collection_names = set(self.COLLECTION_MAP.values())
        for name in collection_names:
            self.chroma_client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(f"Initialized {len(collection_names)} ChromaDB collections")

    async def retrieve(
        self,
        query: str,
        intent: str,
        category: str,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve relevant knowledge base documents.

        Args:
            query: Original customer message
            intent: Classified intent (for collection routing)
            category: Fine-grained category (for metadata filtering)
            top_k: Number of documents to retrieve

        Returns:
            dict with 'docs' (list of text chunks) and 'confidence' (float)
        """
        # Step 1: Rewrite query for better retrieval
        rewritten_query = await self._rewrite_query(query, intent)
        logger.debug(f"Query rewritten: '{query[:50]}...' → '{rewritten_query[:50]}...'")

        # Step 2: Determine which collection(s) to search
        primary_collection = self.COLLECTION_MAP.get(intent, "general_faq")
        collections_to_search = [primary_collection]

        # Always include general FAQ as fallback
        if primary_collection != "general_faq":
            collections_to_search.append("general_faq")

        # Step 3: Search each collection
        all_results = []
        for collection_name in collections_to_search:
            try:
                collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                )
                results = collection.query(
                    query_texts=[rewritten_query],
                    n_results=min(top_k, collection.count() or 1),
                    include=["documents", "distances", "metadatas"],
                )
                all_results.append((results, collection_name))

            except Exception as e:
                logger.warning(f"Error searching collection '{collection_name}': {e}")

        # Step 4: Parse, filter, and deduplicate results
        documents = []
        seen_hashes = set()

        for results, coll_name in all_results:
            if not results["documents"] or not results["documents"][0]:
                continue

            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score (1 = perfect match)
                similarity = 1 - (distance / 2)

                if similarity < self.SIMILARITY_THRESHOLD:
                    continue  # Skip low-quality matches

                # Deduplicate by content hash
                doc_hash = hashlib.md5(doc.encode()).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

                # Add source metadata to document
                source = metadata.get("source", coll_name)
                documents.append({
                    "text": doc,
                    "similarity": similarity,
                    "source": source,
                    "collection": coll_name,
                })

        # Step 5: Sort by similarity and take top_k
        documents.sort(key=lambda x: x["similarity"], reverse=True)
        top_docs = documents[:top_k]

        # Step 6: Calculate overall confidence
        if top_docs:
            avg_similarity = sum(d["similarity"] for d in top_docs) / len(top_docs)
            # Boost confidence if we have multiple high-quality results
            confidence = min(avg_similarity * (1 + 0.1 * (len(top_docs) - 1)), 1.0)
        else:
            confidence = 0.0

        # Format docs as plain text for the supervisor
        formatted_docs = [
            f"[Source: {d['source']}] {d['text']}"
            for d in top_docs
        ]

        logger.info(
            f"Retrieved {len(formatted_docs)} docs "
            f"(confidence={confidence:.2f}, collections={collections_to_search})"
        )

        return {
            "docs": formatted_docs,
            "confidence": confidence,
            "doc_count": len(formatted_docs),
        }

    async def _rewrite_query(self, original_query: str, intent: str) -> str:
        """
        Rewrite the customer's message into an optimal retrieval query.
        
        This improves RAG performance by removing emotional language,
        normalizing terminology, and making the query more document-like.
        """
        prompt = f"""Rewrite the following customer support message as an optimal search query 
for a knowledge base. The intent is: {intent}.

Rules:
- Remove emotional language and pleasantries
- Keep only the factual/technical core of the question
- Use standard terminology (e.g., "refund policy" not "get my money back")
- Output ONLY the rewritten query, nothing else (no quotes, no explanation)

Original message: {original_query}

Rewritten query:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip().strip('"').strip("'")
            return rewritten if rewritten else original_query

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original")
            return original_query

    async def add_document(
        self,
        text: str,
        collection_name: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text content
            collection_name: Target collection
            metadata: Optional metadata (source, title, etc.)
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            Document ID
        """
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        # Generate ID from content hash if not provided
        if not doc_id:
            doc_id = f"doc_{hashlib.sha256(text.encode()).hexdigest()[:16]}"

        collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

        logger.debug(f"Added document {doc_id} to collection '{collection_name}'")
        return doc_id

    def get_collection_stats(self) -> dict:
        """Return stats for all collections."""
        stats = {}
        for intent, collection_name in self.COLLECTION_MAP.items():
            try:
                collection = self.chroma_client.get_collection(collection_name)
                stats[collection_name] = {"count": collection.count()}
            except Exception:
                stats[collection_name] = {"count": 0}
        return stats
