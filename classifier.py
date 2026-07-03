"""
Case Classifier Module
Classifies case type (civil/criminal/family/property/banking/tax/corporate/constitutional)
and determines the best court template using keyword analysis + embeddings.
"""

import logging
import json
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword rule-sets for each case category
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "criminal": [
        "murder", "qatl", "ppc", "fir", "accused", "arrested", "sessions court",
        "bailâ€", "convicted", "acquitted", "theft", "robbery", "dacoity",
        "hurt", "kidnapping", "abduction", "rape", "zina", "hudood", "narcotics",
        "cnsa", "terrorism", "anti-terrorism", "offence", "challan", "police station",
    ],
    "civil": [
        "suit", "plaintiff", "defendant", "cpc", "decree", "injunction",
        "specific performance", "damages", "breach of contract", "tortious",
        "negligence", "trespass", "nuisance", "declaratory", "permanent injunction",
    ],
    "family": [
        "divorce", "talaq", "khul", "dissolution", "maintenance", "iddat",
        "mehr", "dower", "custody", "guardian", "ward", "marriage", "nikah",
        "muslim family laws", "family court", "inheritance", "succession",
        "miras", "will", "hiba", "gift",
    ],
    "property": [
        "land", "plot", "property", "sale deed", "registry", "transfer",
        "mutation", "khasra", "khatooni", "patwari", "revenue", "tenancy",
        "pre-emption", "waqf", "possession", "eviction", "rent", "landlord",
        "tenant", "lease", "dha", "housing scheme", "bahria town",
    ],
    "banking": [
        "bank", "loan", "finance", "recovery", "promissory note", "cheque",
        "dishonour", "mortgage", "hypothecation", "pledge", "guarantor",
        "financial institution", "nbfc", "leasing", "musharaka", "murabaha",
        "ijarah", "state bank", "sbp", "banking ordinance",
    ],
    "tax": [
        "income tax", "sales tax", "customs", "fбp", "fbr", "ito",
        "assessment", "demand notice", "tribunal", "income tax ordinance",
        "tax", "withholding", "penalty", "evasion",
    ],
    "constitutional": [
        "fundamental rights", "article 10", "article 25", "article 184",
        "writ petition", "habeas corpus", "mandamus", "certiorari",
        "constitutional", "parliament", "legislation", "ultra vires",
        "due process", "equal protection",
    ],
    "corporate": [
        "company", "director", "shareholder", "secp", "companies act",
        "securities", "stock exchange", "psx", "listed company", "winding up",
        "liquidation", "debenture", "shares",
    ],
    "accountability": [
        "nab", "corruption", "corrupt practices", "reference", "plea bargain",
        "assets beyond means", "benami", "accountability court", "misconduct",
    ],
    "shariah": [
        "shariah", "islamic", "riba", "interest", "federal shariat court",
        "fsc", "quran", "hadith", "fiqh", "un-islamic", "repugnant",
    ],
}

# Court-to-category mapping (which courts typically hear which matters)
COURT_CATEGORY_MAP: Dict[str, List[str]] = {
    "Supreme Court Pakistan": ["constitutional", "civil", "criminal"],
    "Lahore High Court": ["civil", "criminal", "property", "family", "constitutional"],
    "Islamabad High Court": ["civil", "criminal", "constitutional", "corporate"],
    "Sindh High Court": ["civil", "criminal", "banking", "corporate", "tax"],
    "Peshawar High Court": ["civil", "criminal", "property", "family"],
    "Balochistan High Court": ["civil", "criminal", "constitutional"],
    "Federal Shariat Court": ["shariah", "hudood", "criminal"],
}

# Templates per court
COURT_TEMPLATES: Dict[str, str] = {
    "Supreme Court Pakistan": "supreme",
    "Lahore High Court": "lahore_hc",
    "Islamabad High Court": "islamabad_hc",
    "Peshawar High Court": "peshawar_hc",
    "Balochistan High Court": "balochistan_hc",
    "Sindh High Court": "sindh_hc",
    "Federal Shariat Court": "fsc",
}


class CaseClassifier:
    """
    Classifies case type and selects an appropriate court template.
    Uses keyword scoring with optional embedding-based similarity.
    """

    def __init__(self):
        self._embedding_model = None
        logger.info("CaseClassifier initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str, case_data: Optional[Dict] = None) -> Dict:
        """
        Classify case type from text and parsed case_data.
        Returns a rich classification result dictionary.
        """
        text_lower = text.lower() if text else ""

        scores = self._score_categories(text_lower, case_data)
        primary_type, secondary_types, confidence = self._rank_categories(scores)
        court = self._suggest_court(primary_type, case_data)
        template = COURT_TEMPLATES.get(court, "supreme")
        sub_type = self._detect_sub_type(primary_type, text_lower)

        result: Dict = {
            "primary_type": primary_type,
            "secondary_types": secondary_types,
            "sub_type": sub_type,
            "confidence": confidence,
            "suggested_court": court,
            "template": template,
            "scores": scores,
        }

        logger.info(
            f"Classification: {primary_type} ({confidence}%) | court={court}"
        )
        return result

    def get_domain_agent_focus(self, case_type: Dict) -> str:
        """
        Returns the domain-specific focus area string for the Domain Specialist Agent.
        """
        primary = (
            case_type.get("primary_type", "civil")
            if isinstance(case_type, dict)
            else str(case_type)
        )
        focus_map = {
            "criminal": "Criminal Law – PPC, CrPC, Evidence",
            "civil": "Civil Litigation – CPC, Contract, Torts",
            "family": "Family Law – Muslim Family Laws, Guardian/Wards, Succession",
            "property": "Property Law – Transfer of Property, Land Revenue, Pre-emption",
            "banking": "Banking & Finance – Banking Ordinance, FIO, NIBAF regulations",
            "tax": "Tax Law – ITO 2001, Sales Tax Act, Customs Act",
            "constitutional": "Constitutional Law – Fundamental Rights, Writ Jurisdiction",
            "corporate": "Corporate Law – Companies Act 2017, SECP regulations",
            "accountability": "Accountability Law – NAB Ordinance, Asset Declaration",
            "shariah": "Islamic Law – Quran, Hadith, Fiqh, FSC Jurisdiction",
        }
        return focus_map.get(primary, "General Law Practice")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_categories(
        self, text_lower: str, case_data: Optional[Dict]
    ) -> Dict[str, int]:
        """Count keyword hits per category."""
        scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}

        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                scores[category] += text_lower.count(kw)

        # Boost from parsed case data
        if case_data:
            sections = case_data.get("legal_sections", [])
            for sec in sections:
                law = sec.get("law", "").lower()
                self._boost_from_law(scores, law)

            # FIR detected → boost criminal
            if case_data.get("fir_details"):
                scores["criminal"] += 10

        return scores

    def _boost_from_law(self, scores: Dict[str, int], law_name: str):
        boosts = {
            "ppc": ("criminal", 8),
            "crpc": ("criminal", 6),
            "qanun-e-shahadat": ("criminal", 4),
            "cpc": ("civil", 6),
            "family court": ("family", 8),
            "muslim family": ("family", 10),
            "transfer of property": ("property", 8),
            "banking": ("banking", 8),
            "income tax": ("tax", 8),
            "sales tax": ("tax", 6),
            "companies act": ("corporate", 8),
            "nab": ("accountability", 10),
            "shariat": ("shariah", 10),
            "constitution": ("constitutional", 6),
        }
        for keyword, (cat, pts) in boosts.items():
            if keyword in law_name:
                scores[cat] += pts

    def _rank_categories(
        self, scores: Dict[str, int]
    ) -> Tuple[str, List[str], int]:
        """Return (primary_type, secondary_types, confidence_pct)."""
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_cats[0][0]
        primary_score = sorted_cats[0][1]
        total_score = sum(s for _, s in sorted_cats)

        # Confidence: ratio of top score to total
        confidence = 0
        if total_score > 0:
            raw = (primary_score / total_score) * 100
            # Non-linear scaling to avoid over-confident results
            confidence = min(95, max(40, int(raw * 1.4)))

        secondary = [c for c, s in sorted_cats[1:4] if s > 0]
        return primary, secondary, confidence

    def _suggest_court(
        self, primary_type: str, case_data: Optional[Dict]
    ) -> str:
        """Determine most appropriate court from case type."""
        # If court was already mentioned in the document, prefer it
        if case_data:
            detected_court = case_data.get("court", "")
            for court_name in COURT_TEMPLATES:
                if court_name.lower() in detected_court.lower():
                    return court_name

        # Shariah matters → FSC
        if primary_type == "shariah":
            return "Federal Shariat Court"

        # Default suggestions per type
        default_court_map = {
            "constitutional": "Supreme Court Pakistan",
            "criminal": "Lahore High Court",
            "civil": "Lahore High Court",
            "family": "Lahore High Court",
            "property": "Lahore High Court",
            "banking": "Sindh High Court",
            "tax": "Islamabad High Court",
            "corporate": "Islamabad High Court",
            "accountability": "Islamabad High Court",
        }
        return default_court_map.get(primary_type, "Supreme Court Pakistan")

    def _detect_sub_type(self, primary_type: str, text_lower: str) -> str:
        """Identify specific sub-type within the primary category."""
        sub_types = {
            "criminal": {
                "murder / qatl": ["murder", "qatl", "302"],
                "robbery / theft": ["robbery", "theft", "dacoity", "379", "392"],
                "fraud / cheating": ["fraud", "cheating", "420", "forgery"],
                "narcotics": ["narcotic", "drug", "heroin", "cnsa"],
                "terrorism": ["terrorism", "ata", "militant"],
                "sexual offence": ["rape", "zina", "assault", "modesty"],
            },
            "civil": {
                "breach of contract": ["breach", "contract", "agreement"],
                "specific performance": ["specific performance"],
                "injunction": ["injunction", "restraint", "stay"],
                "damages": ["damages", "compensation", "loss"],
            },
            "family": {
                "divorce / dissolution": ["divorce", "talaq", "khul", "dissolution"],
                "maintenance": ["maintenance", "nafaqa"],
                "custody / guardianship": ["custody", "guardian"],
                "inheritance / succession": ["inheritance", "succession", "miras"],
            },
            "property": {
                "possession / ejection": ["possession", "ejection", "ejectment"],
                "sale deed dispute": ["sale deed", "bai-nama"],
                "land acquisition": ["acquisition", "compensation"],
                "tenancy / rent": ["rent", "tenant", "landlord"],
                "mutation / registration": ["mutation", "registry", "registration"],
            },
            "banking": {
                "loan recovery": ["recovery", "loan", "finance"],
                "cheque dishonour": ["cheque", "dishonour", "bounce"],
                "mortgage enforcement": ["mortgage", "hypothecation"],
                "islamic banking": ["murabaha", "musharaka", "ijarah"],
            },
            "tax": {
                "income tax": ["income tax", "ito"],
                "sales tax": ["sales tax"],
                "customs duty": ["customs", "import", "export"],
            },
            "constitutional": {
                "fundamental rights": ["fundamental", "article 9", "article 10", "article 25"],
                "writ / habeas corpus": ["habeas corpus", "writ"],
                "election": ["election", "ecp"],
            },
        }

        for sub_label, keywords in sub_types.get(primary_type, {}).items():
            if any(kw in text_lower for kw in keywords):
                return sub_label

        return "General " + primary_type.title()
