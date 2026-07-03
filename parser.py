"""
Case Parser Module
Extracts structured case data from raw document text:
plaintiff/defendant, dates, amounts, legal claims, evidence, relief sought.
Outputs a clean JSON-serialisable dictionary.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CaseParser:
    """
    Extracts structured information from raw case text.
    Works via regex patterns first, then uses OpenAI for deeper extraction.
    """

    def __init__(self):
        self._compile_patterns()
        logger.info("CaseParser initialised")

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    def _compile_patterns(self):
        flags = re.IGNORECASE | re.MULTILINE

        # Party names
        self.pat_plaintiff = re.compile(
            r"(?:plaintiff|petitioner|appellant|complainant|applicant)[:\s]+([A-Z][A-Za-z\s\./,'-]{2,60})",
            flags,
        )
        self.pat_defendant = re.compile(
            r"(?:defendant|respondent|accused|opposite\s+party)[:\s]+([A-Z][A-Za-z\s\./,'-]{2,60})",
            flags,
        )

        # Case numbers
        self.pat_case_no = re.compile(
            r"(?:case\s+no\.|suit\s+no\.|cr\.|crl\.|civ\.|writ\s+petition|w\.p\.|c\.p\.)\s*"
            r"([\w\-/]{3,30}[/\s]?(?:of|\/)\s*(?:19|20)\d{2})",
            flags,
        )

        # Court names
        self.pat_court = re.compile(
            r"(supreme\s+court|lahore\s+high\s+court|islamabad\s+high\s+court|"
            r"sindh\s+high\s+court|peshawar\s+high\s+court|balochistan\s+high\s+court|"
            r"federal\s+shariat\s+court|district\s+court|sessions\s+court|"
            r"family\s+court|accountability\s+court|anti[-\s]terrorism\s+court)",
            flags,
        )

        # Dates
        self.pat_date = re.compile(
            r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\b"
            r"|\b(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+(?:19|20)\d{2})\b",
            flags,
        )

        # Monetary figures (PKR)
        self.pat_money = re.compile(
            r"(?:rs\.?|pkr|rupees?)\s*"
            r"([\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|lakh|crore|thousand))?)",
            flags,
        )

        # Legal sections
        self.pat_sections = re.compile(
            r"(?:section|s\.)\s*(\d+[-A-Z]*(?:\s*\(\w+\))?)"
            r"(?:\s+of\s+the\s+([\w\s]+?)(?:\.|,|\n|$))?",
            flags,
        )

        # Relief sought
        self.pat_relief = re.compile(
            r"(?:relief\s+sought|prayer(?:s)?|wherefore|it\s+is\s+prayed)[:\s]+"
            r"(.{20,500}?)(?:\n\n|\Z)",
            flags | re.DOTALL,
        )

        # FIR / Police
        self.pat_fir = re.compile(
            r"fir\s+no\.?\s*([\d/]+)\s*(?:dated|dt\.?)?\s*([\d/-]*)\s*"
            r"(?:police\s+station\s+([\w\s]+))?",
            flags,
        )

        # Witnesses
        self.pat_witness = re.compile(
            r"(?:pw|dw|cw|witness)\s*[-:]?\s*(\d+)\s*[:\-]\s*([A-Z][a-zA-Z\s]+?)(?:\n|,|\.)",
            flags,
        )

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def parse_case(self, text: str) -> Dict[str, Any]:
        """
        Parse raw case text and return structured dictionary.
        """
        if not text or len(text.strip()) < 20:
            logger.warning("parse_case: input text is too short")
            return self._empty_case()

        logger.info(f"Parsing case text ({len(text)} chars)")

        case_data: Dict[str, Any] = {
            "raw_text_length": len(text),
            "parties": self._extract_parties(text),
            "case_identifiers": self._extract_identifiers(text),
            "court": self._extract_court(text),
            "dates": self._extract_dates(text),
            "monetary_amounts": self._extract_money(text),
            "legal_sections": self._extract_sections(text),
            "fir_details": self._extract_fir(text),
            "witnesses": self._extract_witnesses(text),
            "relief_sought": self._extract_relief(text),
            "key_facts": self._extract_key_facts(text),
            "legal_claims": self._extract_legal_claims(text),
            "evidence_items": self._extract_evidence(text),
            "timeline": self._build_timeline(text),
            "parsed_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Parsing complete: parties={case_data['parties']}, "
            f"dates={len(case_data['dates'])}, amounts={len(case_data['monetary_amounts'])}"
        )
        return case_data

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_parties(self, text: str) -> Dict[str, str]:
        parties: Dict[str, str] = {
            "plaintiff_petitioner": "",
            "defendant_respondent": "",
            "additional_parties": [],
        }

        for m in self.pat_plaintiff.finditer(text):
            name = m.group(1).strip().rstrip(",.:;")
            if len(name) > 3 and not parties["plaintiff_petitioner"]:
                parties["plaintiff_petitioner"] = self._clean_name(name)
                break

        for m in self.pat_defendant.finditer(text):
            name = m.group(1).strip().rstrip(",.:;")
            if len(name) > 3 and not parties["defendant_respondent"]:
                parties["defendant_respondent"] = self._clean_name(name)
                break

        # Detect "v." / "versus" pattern for additional confidence
        vs_match = re.search(
            r"([A-Z][A-Za-z\s\.]+?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z\s\.]+?)(?:\n|$)",
            text[:500],
        )
        if vs_match:
            if not parties["plaintiff_petitioner"]:
                parties["plaintiff_petitioner"] = self._clean_name(vs_match.group(1))
            if not parties["defendant_respondent"]:
                parties["defendant_respondent"] = self._clean_name(vs_match.group(2))

        return parties

    def _clean_name(self, name: str) -> str:
        name = re.sub(r"\s+", " ", name).strip()
        return name[:80]  # cap length

    def _extract_identifiers(self, text: str) -> Dict[str, str]:
        ids: Dict[str, str] = {"case_number": "", "suit_number": "", "year": ""}

        for m in self.pat_case_no.finditer(text[:2000]):
            raw = m.group(1).strip()
            ids["case_number"] = raw
            year_m = re.search(r"((?:19|20)\d{2})", raw)
            if year_m:
                ids["year"] = year_m.group(1)
            break

        # Try to grab year from first 300 chars if not found
        if not ids["year"]:
            year_m = re.search(r"\b((?:19|20)\d{2})\b", text[:300])
            if year_m:
                ids["year"] = year_m.group(1)

        return ids

    def _extract_court(self, text: str) -> str:
        for m in self.pat_court.finditer(text[:1000]):
            return m.group(0).title()
        return "Court Not Specified"

    def _extract_dates(self, text: str) -> List[str]:
        dates = []
        seen: set = set()
        for m in self.pat_date.finditer(text):
            raw = (m.group(1) or m.group(2) or "").strip()
            if raw and raw not in seen:
                seen.add(raw)
                dates.append(raw)
        return dates[:20]  # return up to 20 dates

    def _extract_money(self, text: str) -> List[Dict[str, str]]:
        amounts = []
        seen: set = set()
        for m in self.pat_money.finditer(text):
            amt = "Rs. " + m.group(1).strip()
            if amt not in seen:
                seen.add(amt)
                # Find surrounding context (up to 60 chars before)
                start = max(0, m.start() - 60)
                ctx = text[start : m.start()].strip().split("\n")[-1]
                amounts.append({"amount": amt, "context": ctx[-80:]})
        return amounts[:15]

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        sections = []
        seen: set = set()
        for m in self.pat_sections.finditer(text):
            sec = m.group(1).strip()
            law = (m.group(2) or "").strip().rstrip(",. ")
            key = f"{sec}_{law}"
            if key not in seen and len(sec) < 15:
                seen.add(key)
                sections.append({"section": sec, "law": law or "Unknown Law"})
        return sections[:20]

    def _extract_fir(self, text: str) -> Optional[Dict[str, str]]:
        for m in self.pat_fir.finditer(text):
            return {
                "fir_number": m.group(1).strip(),
                "fir_date": (m.group(2) or "").strip(),
                "police_station": (m.group(3) or "").strip(),
            }
        return None

    def _extract_witnesses(self, text: str) -> List[Dict[str, str]]:
        witnesses = []
        seen: set = set()
        for m in self.pat_witness.finditer(text):
            wid = m.group(1).strip()
            name = m.group(2).strip()
            if wid not in seen:
                seen.add(wid)
                witnesses.append({"id": wid, "name": name})
        return witnesses[:20]

    def _extract_relief(self, text: str) -> str:
        for m in self.pat_relief.finditer(text):
            raw = m.group(1).strip()
            return self._truncate(raw, 600)
        return ""

    def _extract_key_facts(self, text: str) -> List[str]:
        """
        Extract key factual sentences heuristically.
        Looks for sentences containing date/name/amount keywords.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        key_sents = []
        triggers = re.compile(
            r"\b(alleged|contend|claim|execute|enter|sign|pay|breach|fail|refuse|"
            r"register|file|complain|commit|transfer|arrest|murder|fraud)\b",
            re.IGNORECASE,
        )
        for sent in sentences:
            sent = sent.strip()
            if 30 < len(sent) < 300 and triggers.search(sent):
                key_sents.append(sent)
            if len(key_sents) >= 10:
                break
        return key_sents

    def _extract_legal_claims(self, text: str) -> List[str]:
        claim_triggers = re.compile(
            r"(?:section|under|pursuant|contrary to|violation of|offence under|"
            r"breach of|liable under|entitled under|charged with|accused of)\s+"
            r"(.{10,120}?)(?:\.|,|\n)",
            re.IGNORECASE,
        )
        claims = []
        seen: set = set()
        for m in claim_triggers.finditer(text):
            c = m.group(0).strip()
            if c not in seen:
                seen.add(c)
                claims.append(c)
            if len(claims) >= 10:
                break
        return claims

    def _extract_evidence(self, text: str) -> List[str]:
        evidence_triggers = re.compile(
            r"(?:exhibit|evidence|document|deposition|affidavit|statement|witness|"
            r"fingerprint|dna|expert\s+report|medical\s+report|post\s+mortem|"
            r"sale\s+deed|agreement|invoice|cheque|promissory\s+note)\s+"
            r"(.{0,80}?)(?:\.|,|\n)",
            re.IGNORECASE,
        )
        items = []
        seen: set = set()
        for m in evidence_triggers.finditer(text):
            item = m.group(0).strip()
            if item not in seen and len(item) > 10:
                seen.add(item)
                items.append(item)
            if len(items) >= 12:
                break
        return items

    def _build_timeline(self, text: str) -> List[Dict[str, str]]:
        """
        Build a basic chronological timeline from date + adjacent context.
        """
        timeline = []
        for m in self.pat_date.finditer(text):
            date_str = (m.group(1) or m.group(2) or "").strip()
            if not date_str:
                continue
            # Get a snippet of text after the date (up to 100 chars)
            snippet = text[m.end(): m.end() + 100].strip().split("\n")[0][:100]
            timeline.append({"date": date_str, "event": snippet})
            if len(timeline) >= 10:
                break
        return timeline

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _truncate(self, text: str, max_len: int = 500) -> str:
        return text[:max_len] + ("…" if len(text) > max_len else "")

    def _empty_case(self) -> Dict[str, Any]:
        return {
            "raw_text_length": 0,
            "parties": {
                "plaintiff_petitioner": "Unknown",
                "defendant_respondent": "Unknown",
                "additional_parties": [],
            },
            "case_identifiers": {"case_number": "", "suit_number": "", "year": ""},
            "court": "Not Specified",
            "dates": [],
            "monetary_amounts": [],
            "legal_sections": [],
            "fir_details": None,
            "witnesses": [],
            "relief_sought": "",
            "key_facts": [],
            "legal_claims": [],
            "evidence_items": [],
            "timeline": [],
            "parsed_at": datetime.utcnow().isoformat(),
            "error": "Insufficient text to parse",
        }

    def to_json(self, case_data: Dict) -> str:
        """Serialize case_data to a formatted JSON string."""
        return json.dumps(case_data, indent=2, ensure_ascii=False, default=str)
