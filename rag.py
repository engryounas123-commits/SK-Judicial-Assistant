"""
Legal RAG Module
Retrieval-Augmented Generation system for Pakistani legal precedents and statutes.
Loads from data/legal_knowledge.json, builds a FAISS index, and retrieves
the most relevant documents at query time.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from modules.embeddings import LegalEmbeddings

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "legal_knowledge.json"
INDEX_PATH = str(Path(__file__).parent.parent / "data" / "legal_index.faiss")


class LegalRAG:
    """
    Retrieval system over Pakistani legal knowledge (statutes + precedents).
    Provides grounded context to AI judge agents.
    """

    def __init__(self):
        self.embeddings = LegalEmbeddings()
        self.knowledge_base: List[Dict] = []
        self.is_initialised = False
        logger.info("LegalRAG created (not yet initialised)")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Load legal knowledge, build (or reload) FAISS index.
        Called once at application start-up.
        """
        if self.is_initialised:
            return True

        # 1. Try to load existing index
        if self.embeddings.load(INDEX_PATH):
            self._load_knowledge_base()
            self.is_initialised = True
            logger.info("LegalRAG: loaded cached FAISS index")
            return True

        # 2. Build fresh index
        kb_loaded = self._load_knowledge_base()
        if not kb_loaded:
            logger.error("LegalRAG: no knowledge base loaded – RAG disabled")
            return False

        logger.info(f"Building FAISS index over {len(self.knowledge_base)} documents…")
        success = self.embeddings.build_index(self.knowledge_base)

        if success:
            self.embeddings.save(INDEX_PATH)
            self.is_initialised = True
            logger.info("LegalRAG: index built and saved")
        else:
            logger.warning("LegalRAG: index build failed – will use fallback search")
            self.is_initialised = True  # fallback still works

        return True

    def _load_knowledge_base(self) -> bool:
        """Load legal knowledge from JSON file."""
        if not DATA_PATH.exists():
            logger.error(f"Knowledge base not found: {DATA_PATH}")
            return False

        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.knowledge_base = []

            # Flatten statutes
            for statute in data.get("statutes", []):
                self.knowledge_base.append(
                    {
                        "type": "statute",
                        "content": f"{statute['title']}: {statute['content']}",
                        "title": statute.get("title", ""),
                        "id": statute.get("id", ""),
                        "category": statute.get("category", "general"),
                        "court_source": statute.get("court_source", "All Courts"),
                    }
                )

            # Flatten precedents
            for prec in data.get("precedents", []):
                self.knowledge_base.append(
                    {
                        "type": "precedent",
                        "content": (
                            f"{prec['case_title']} ({prec['citation']}): "
                            f"{prec['legal_principle']}"
                        ),
                        "title": prec.get("case_title", ""),
                        "citation": prec.get("citation", ""),
                        "court": prec.get("court", ""),
                        "year": str(prec.get("year", "")),
                        "judge": prec.get("judge", ""),
                        "legal_principle": prec.get("legal_principle", ""),
                        "category": prec.get("category", "general"),
                        "keywords": prec.get("keywords", []),
                        "verified": prec.get("verified", False),
                    }
                )

            # Flatten legal principles / maxims
            for principle in data.get("legal_principles", []):
                self.knowledge_base.append(
                    {
                        "type": "maxim",
                        "content": (
                            f"{principle['maxim']} – '{principle['translation']}'. "
                            f"{principle['application']}"
                        ),
                        "title": principle.get("maxim", ""),
                        "category": principle.get("category", "general"),
                    }
                )

            logger.info(
                f"Knowledge base loaded: {len(self.knowledge_base)} documents "
                f"({sum(1 for d in self.knowledge_base if d['type']=='statute')} statutes, "
                f"{sum(1 for d in self.knowledge_base if d['type']=='precedent')} precedents)"
            )
            return True

        except Exception as exc:
            logger.error(f"Failed to load knowledge base: {exc}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve_precedents(
        self,
        query: str,
        case_type: str = "general",
        top_k: int = 8,
    ) -> List[Dict]:
        """
        Retrieve the most relevant legal precedents and statutes for a given query.
        Returns a list of enriched result dicts suitable for the judge agents.
        """
        if not self.is_initialised:
            self.initialize()

        if not query:
            return self._get_type_defaults(case_type, top_k)

        # Build a focused query by prepending case-type keywords
        enhanced_query = self._enhance_query(query, case_type)

        # Retrieve from embedding index
        raw_results: List[Tuple[float, Dict]] = self.embeddings.search(
            enhanced_query, top_k=top_k * 2  # retrieve more, then filter
        )

        # Also do keyword-based retrieval as a complement
        keyword_results = self._keyword_search(query, case_type, top_k)

        # Merge, deduplicate, rank
        merged = self._merge_results(raw_results, keyword_results, top_k)

        logger.info(
            f"RAG retrieved {len(merged)} docs for query "
            f"(case_type={case_type}, top_k={top_k})"
        )
        return merged

    def retrieve_statutes(
        self, query: str, case_type: str = "general", top_k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant statutory provisions only."""
        results = self.retrieve_precedents(query, case_type, top_k * 2)
        statutes = [r for r in results if r.get("type") == "statute"]
        return statutes[:top_k]

    def get_all_by_category(self, category: str) -> List[Dict]:
        """Return all knowledge-base items matching a category."""
        return [d for d in self.knowledge_base if d.get("category") == category]

    def format_context_for_agent(self, results: List[Dict]) -> str:
        """
        Format retrieved results as a concise context block for AI agents.
        """
        if not results:
            return "No specific precedents retrieved from the knowledge base."

        lines = ["=== RETRIEVED LEGAL CONTEXT ===\n"]

        precedents = [r for r in results if r.get("type") == "precedent"]
        statutes = [r for r in results if r.get("type") == "statute"]
        maxims = [r for r in results if r.get("type") == "maxim"]

        if precedents:
            lines.append("--- RELEVANT PRECEDENTS ---")
            for p in precedents:
                verified_tag = "✔ VERIFIED" if p.get("verified") else "⚠ REQUIRES VERIFICATION"
                lines.append(
                    f"• {p['title']}\n"
                    f"  Citation : {p.get('citation', 'N/A')}\n"
                    f"  Court    : {p.get('court', 'N/A')} ({p.get('year', '')})\n"
                    f"  Judge    : {p.get('judge', 'N/A')}\n"
                    f"  Principle: {p.get('legal_principle', '')}\n"
                    f"  Status   : {verified_tag}\n"
                )

        if statutes:
            lines.append("\n--- APPLICABLE STATUTES ---")
            for s in statutes:
                lines.append(
                    f"• {s['title']}\n"
                    f"  {s.get('content', '')[:300]}\n"
                )

        if maxims:
            lines.append("\n--- LEGAL MAXIMS ---")
            for m in maxims:
                lines.append(f"• {m.get('content', '')}")

        lines.append("\n=== END OF LEGAL CONTEXT ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enhance_query(self, query: str, case_type: str) -> str:
        """Prepend domain-specific terms to improve retrieval."""
        type_terms = {
            "criminal": "Pakistan Penal Code criminal offence qatl murder theft robbery",
            "civil": "civil suit specific performance injunction damages Pakistan CPC",
            "family": "Muslim Family Laws divorce maintenance custody inheritance",
            "property": "land property sale deed possession mutation pre-emption",
            "banking": "banking recovery loan finance cheque dishonour",
            "tax": "income tax FBR assessment penalty ITO 2001",
            "constitutional": "fundamental rights Article constitution writ petition",
            "corporate": "companies act SECP director shareholders",
            "shariah": "Islamic law riba interest Quran Hadith Fiqh Federal Shariat Court",
            "accountability": "NAB corruption assets accountability court",
        }
        prefix = type_terms.get(case_type, "Pakistani law legal precedent")
        return f"{prefix} {query}"[:1000]

    def _keyword_search(
        self, query: str, case_type: str, top_k: int
    ) -> List[Tuple[float, Dict]]:
        """Keyword overlap scoring as a complement to embeddings."""
        query_words = set(query.lower().split())
        results = []

        for doc in self.knowledge_base:
            # Category bonus
            cat_score = 2.0 if doc.get("category") == case_type else 1.0

            content_words = set(doc.get("content", "").lower().split())
            keyword_words = set(
                kw.lower() for kw in doc.get("keywords", [])
            )

            overlap = len(query_words & content_words)
            kw_overlap = len(query_words & keyword_words) * 3  # keywords weighted higher

            score = (overlap + kw_overlap) * cat_score / max(len(query_words), 1)
            if score > 0:
                results.append((score, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        embedding_results: List[Tuple[float, Dict]],
        keyword_results: List[Tuple[float, Dict]],
        top_k: int,
    ) -> List[Dict]:
        """Merge embedding and keyword results, deduplicate, pick best top_k."""
        seen_titles: set = set()
        merged: List[Tuple[float, Dict]] = []

        # Normalise and combine
        for score, doc in embedding_results:
            title = doc.get("title", doc.get("content", "")[:50])
            if title not in seen_titles:
                seen_titles.add(title)
                merged.append((score * 1.5, doc))  # Weight embeddings higher

        for score, doc in keyword_results:
            title = doc.get("title", doc.get("content", "")[:50])
            if title not in seen_titles:
                seen_titles.add(title)
                merged.append((score, doc))

        merged.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in merged[:top_k]]

    def _get_type_defaults(self, case_type: str, top_k: int) -> List[Dict]:
        """Return default documents for a case type when no query is given."""
        matching = [d for d in self.knowledge_base if d.get("category") == case_type]
        return matching[:top_k] if matching else self.knowledge_base[:top_k]
