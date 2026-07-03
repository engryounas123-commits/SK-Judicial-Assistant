"""
AI Judge Panel Module
Implements 5 specialist AI judge agents that deliberate internally
before the Chief Justice synthesises a final binding judgment.

Agents:
  1. LawExpertAgent        – Pakistani statutory & constitutional law
  2. ShariahJudgeAgent     – Islamic jurisprudence & FSC jurisdiction
  3. DomainSpecialistAgent – Domain-specific regulations (banking/property/tax…)
  4. PrecedentResearchAgent– Citation retrieval & ratio extraction
  5. ChiefJusticeAgent     – Synthesis & final operative order (JSON output)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DEFAULT_MODEL = os.getenv("AI_MODEL", "gpt-4o")
MAX_TOKENS = 2000
TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Helper – load prompt files
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"Prompt file not found: {path}")
    return ""


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseJudgeAgent:
    """
    Generic agent that calls the OpenAI Chat API with a specialist system prompt.
    Wraps the call with exponential-backoff retry logic.
    """

    def __init__(
        self,
        name: str,
        prompt_file: str,
        client,
        model: str = DEFAULT_MODEL,
    ):
        self.name = name
        self.system_prompt = _load_prompt(prompt_file)
        self.client = client
        self.model = model
        logger.info(f"Agent '{name}' initialised (model={model})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, messages: List[Dict]) -> str:
        """Raw API call with retry."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    def deliberate(
        self,
        case_data: Dict,
        case_text: str,
        previous_opinions: Dict[str, str],
        **kwargs,
    ) -> str:
        """
        Generate this agent's opinion given the case data and previous opinions.
        """
        user_message = self._build_user_message(
            case_data, case_text, previous_opinions, **kwargs
        )
        messages = [
            {"role": "system", "content": self.system_prompt or self._default_system()},
            {"role": "user",   "content": user_message},
        ]
        try:
            opinion = self._call_api(messages)
            logger.info(f"Agent '{self.name}' opinion generated ({len(opinion)} chars)")
            return opinion
        except Exception as exc:
            logger.error(f"Agent '{self.name}' API error: {exc}", exc_info=True)
            return self._fallback_opinion(exc)

    def _build_user_message(
        self,
        case_data: Dict,
        case_text: str,
        previous_opinions: Dict[str, str],
        **kwargs,
    ) -> str:
        """Compose the user-turn prompt. Subclasses may override."""
        parties = case_data.get("parties", {})
        sections = [s["section"] for s in case_data.get("legal_sections", [])[:5]]
        money = ", ".join(a["amount"] for a in case_data.get("monetary_amounts", [])[:3])

        prior_block = ""
        if previous_opinions:
            prior_block = "\n\n=== PREVIOUS JUDGE OPINIONS ===\n"
            for agent_name, opinion in previous_opinions.items():
                prior_block += f"\n[{agent_name}]\n{opinion[:600]}…\n"

        return f"""CASE DOCUMENT EXCERPT (first 3000 chars):
{case_text[:3000]}

PARSED CASE DATA:
- Plaintiff/Petitioner : {parties.get('plaintiff_petitioner', 'N/A')}
- Defendant/Respondent : {parties.get('defendant_respondent', 'N/A')}
- Court                : {case_data.get('court', 'N/A')}
- Case No.             : {case_data.get('case_identifiers', {}).get('case_number', 'N/A')}
- Key Sections Cited   : {', '.join(sections) or 'None detected'}
- Monetary Amounts     : {money or 'None detected'}
- Relief Sought        : {case_data.get('relief_sought', 'Not specified')[:200]}
{prior_block}

Please provide your expert legal opinion as per your assigned role and format."""

    def _default_system(self) -> str:
        return f"You are {self.name}, a senior Pakistani judge. Analyse the case and give your expert legal opinion."

    def _fallback_opinion(self, exc: Exception) -> str:
        return (
            f"[{self.name}] API error – opinion unavailable.\n"
            f"Reason: {str(exc)[:200]}\n"
            "Please check your OpenAI API key and try again."
        )


# ---------------------------------------------------------------------------
# Specialist Agents
# ---------------------------------------------------------------------------

class LawExpertAgent(BaseJudgeAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        super().__init__(
            name="Pakistani Law Expert",
            prompt_file="judge_law.txt",
            client=client,
            model=model,
        )


class ShariahJudgeAgent(BaseJudgeAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        super().__init__(
            name="Shariah Judge",
            prompt_file="judge_shariah.txt",
            client=client,
            model=model,
        )


class DomainSpecialistAgent(BaseJudgeAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        super().__init__(
            name="Domain Specialist Judge",
            prompt_file="judge_domain.txt",
            client=client,
            model=model,
        )

    def _build_user_message(
        self,
        case_data: Dict,
        case_text: str,
        previous_opinions: Dict[str, str],
        **kwargs,
    ) -> str:
        domain_focus = kwargs.get("domain_focus", "General Law")
        base = super()._build_user_message(case_data, case_text, previous_opinions, **kwargs)
        return f"DOMAIN FOCUS FOR THIS CASE: {domain_focus}\n\n{base}"


class PrecedentResearchAgent(BaseJudgeAgent):
    def __init__(self, client, model: str = DEFAULT_MODEL):
        super().__init__(
            name="Precedent Research Judge",
            prompt_file="judge_precedent.txt",
            client=client,
            model=model,
        )

    def _build_user_message(
        self,
        case_data: Dict,
        case_text: str,
        previous_opinions: Dict[str, str],
        **kwargs,
    ) -> str:
        legal_context = kwargs.get("legal_context", "No RAG context available.")
        base = super()._build_user_message(case_data, case_text, previous_opinions, **kwargs)
        return f"{base}\n\n=== RAG-RETRIEVED LEGAL CONTEXT ===\n{legal_context}"


class ChiefJusticeAgent(BaseJudgeAgent):
    """
    Reads all panel opinions and returns a final structured JSON judgment.
    """

    def __init__(self, client, model: str = DEFAULT_MODEL):
        super().__init__(
            name="Chief Justice AI",
            prompt_file="chief_justice.txt",
            client=client,
            model=model,
        )

    def synthesize(
        self,
        case_data: Dict,
        case_text: str,
        all_opinions: Dict[str, str],
        court: str,
        judgment_style: str,
        precedents: List[Dict],
        **kwargs,
    ) -> Dict:
        """
        Synthesise all panel opinions into a final JSON judgment.
        Returns a parsed dict (falls back to a default structure on JSON error).
        """
        user_msg = self._build_synthesis_message(
            case_data, case_text, all_opinions, court, judgment_style, precedents
        )
        messages = [
            {"role": "system", "content": self.system_prompt or self._default_system()},
            {"role": "user",   "content": user_msg},
        ]

        try:
            raw = self._call_api(messages)
            logger.info(f"Chief Justice raw output ({len(raw)} chars)")
            return self._parse_json(raw, case_data, court)
        except Exception as exc:
            logger.error(f"Chief Justice synthesis error: {exc}", exc_info=True)
            return self._default_judgment(case_data, court, str(exc))

    def _build_synthesis_message(
        self,
        case_data: Dict,
        case_text: str,
        all_opinions: Dict[str, str],
        court: str,
        judgment_style: str,
        precedents: List[Dict],
    ) -> str:
        opinions_block = "\n\n".join(
            f"=== {name} ===\n{opinion}" for name, opinion in all_opinions.items()
        )

        prec_block = ""
        if precedents:
            prec_lines = []
            for p in precedents[:6]:
                if p.get("type") == "precedent":
                    prec_lines.append(
                        f"- {p.get('title','')} | {p.get('citation','')} | "
                        f"{p.get('legal_principle','')[:120]}"
                    )
            if prec_lines:
                prec_block = "VERIFIED PRECEDENTS FROM RAG:\n" + "\n".join(prec_lines)

        parties = case_data.get("parties", {})

        return f"""You are presiding as Chief Justice over {court}.
Judgment Style: {judgment_style}

PARTIES:
- Petitioner/Plaintiff : {parties.get('plaintiff_petitioner', 'N/A')}
- Respondent/Defendant : {parties.get('defendant_respondent', 'N/A')}

{prec_block}

JUDGE PANEL OPINIONS:
{opinions_block}

CASE DOCUMENT EXCERPT:
{case_text[:2000]}

Based on all opinions above, draft the FINAL JUDGMENT as a single valid JSON object
following the exact schema in your system prompt. Output ONLY the JSON – no preamble,
no markdown fences. The JSON must be parseable."""

    def _parse_json(self, raw: str, case_data: Dict, court: str) -> Dict:
        """Extract and parse JSON from the model output."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        # Find outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            json_str = cleaned[start:end]
            try:
                parsed = json.loads(json_str)
                # Ensure required fields exist
                parsed.setdefault("confidence_score", 75)
                parsed.setdefault("decision", "Requires Further Analysis")
                parsed.setdefault("operative_order", "Judgment pending complete analysis.")
                return parsed
            except json.JSONDecodeError as exc:
                logger.error(f"JSON parse error: {exc}")

        # Fallback: return structured dict with raw text
        parties = case_data.get("parties", {})
        return {
            "case_summary": "AI judgment draft – JSON parsing failed. See raw text.",
            "case_type": "general",
            "primary_court": court,
            "parties": {
                "petitioner_plaintiff": parties.get("plaintiff_petitioner", "N/A"),
                "respondent_defendant": parties.get("defendant_respondent", "N/A"),
            },
            "key_facts": case_data.get("key_facts", [])[:5],
            "issues_framed": ["Issue 1: To be determined"],
            "applicable_laws": [],
            "evidence_analysis": "Evidence analysis not parsed.",
            "legal_reasoning": raw[:1000],
            "shariah_aspects": "N/A",
            "precedents_applied": [],
            "findings": ["Finding pending final analysis"],
            "decision": "RESERVED",
            "operative_order": "Judgment to be re-generated.",
            "relief_granted": [],
            "directives": [],
            "costs": "Not determined",
            "confidence_score": 50,
            "notes": f"JSON parsing failed. Raw output saved. Error: {raw[:200]}",
        }

    def _default_judgment(self, case_data: Dict, court: str, error: str) -> Dict:
        """Fallback judgment structure when the API call fails entirely."""
        parties = case_data.get("parties", {})
        return {
            "case_summary": "AI judgment unavailable due to API error.",
            "case_type": "general",
            "primary_court": court,
            "parties": {
                "petitioner_plaintiff": parties.get("plaintiff_petitioner", "N/A"),
                "respondent_defendant": parties.get("defendant_respondent", "N/A"),
            },
            "key_facts": [],
            "issues_framed": [],
            "applicable_laws": [],
            "evidence_analysis": "",
            "legal_reasoning": "",
            "shariah_aspects": "",
            "precedents_applied": [],
            "findings": [],
            "decision": "API ERROR",
            "operative_order": f"Judgment could not be generated. Error: {error[:200]}",
            "relief_granted": [],
            "directives": [],
            "costs": "N/A",
            "confidence_score": 0,
            "notes": error[:300],
        }


# ---------------------------------------------------------------------------
# Judge Panel Orchestrator
# ---------------------------------------------------------------------------

class JudgePanel:
    """
    Orchestrates the 5-agent deliberation pipeline:
      Law → Shariah → Domain → Precedent → Chief Justice (synthesis)
    """

    AGENT_NAMES = {
        "Pakistani Law Expert":    "law",
        "Shariah Judge":           "shariah",
        "Domain Specialist Judge": "domain",
        "Precedent Research Judge":"precedent",
        "Chief Justice AI":        "chief",
    }

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        self.model = model
        self.law_agent      = LawExpertAgent(self.client, model)
        self.shariah_agent  = ShariahJudgeAgent(self.client, model)
        self.domain_agent   = DomainSpecialistAgent(self.client, model)
        self.precedent_agent= PrecedentResearchAgent(self.client, model)
        self.chief_justice  = ChiefJusticeAgent(self.client, model)

        logger.info(f"JudgePanel ready (model={model})")

    def deliberate(
        self,
        case_data: Dict,
        case_text: str,
        case_type,                    # str OR dict from classifier
        court: str,
        precedents: List[Dict],
        judgment_style: str,
        selected_judges: List[str],
        rag_instance=None,
    ) -> Tuple[Dict[str, str], Dict]:
        """
        Run the full deliberation pipeline.
        Returns (opinions_dict, final_judgment_dict).
        """
        # Normalise case_type
        if isinstance(case_type, dict):
            ct_str       = case_type.get("primary_type", "civil")
            domain_focus = case_type.get("sub_type", "General")
        else:
            ct_str       = str(case_type)
            domain_focus = ct_str.title()

        # Prepare RAG context string
        if rag_instance is not None and precedents:
            legal_context = rag_instance.format_context_for_agent(precedents)
        else:
            legal_context = self._format_precedents_simple(precedents)

        opinions: Dict[str, str] = {}
        active_agents = set(selected_judges) if selected_judges else set(self.AGENT_NAMES.keys())

        # ---- Agent 1: Pakistani Law Expert ----
        if "Pakistani Law Expert" in active_agents or not active_agents:
            self._log_step("Pakistani Law Expert")
            opinions["Pakistani Law Expert"] = self.law_agent.deliberate(
                case_data, case_text, {}
            )

        # ---- Agent 2: Shariah Judge ----
        if "Shariah Judge" in active_agents or not active_agents:
            self._log_step("Shariah Judge")
            opinions["Shariah Judge"] = self.shariah_agent.deliberate(
                case_data, case_text, {"Pakistani Law Expert": opinions.get("Pakistani Law Expert", "")}
            )

        # ---- Agent 3: Domain Specialist ----
        if "Domain Specialist Judge" in active_agents or not active_agents:
            self._log_step("Domain Specialist Judge")
            opinions["Domain Specialist Judge"] = self.domain_agent.deliberate(
                case_data, case_text,
                {k: v for k, v in opinions.items()},
                domain_focus=domain_focus,
            )

        # ---- Agent 4: Precedent Researcher ----
        if "Precedent Research Judge" in active_agents or not active_agents:
            self._log_step("Precedent Research Judge")
            opinions["Precedent Research Judge"] = self.precedent_agent.deliberate(
                case_data, case_text,
                {k: v for k, v in opinions.items()},
                legal_context=legal_context,
            )

        # ---- Agent 5: Chief Justice (synthesis) ----
        self._log_step("Chief Justice AI")
        final_judgment = self.chief_justice.synthesize(
            case_data=case_data,
            case_text=case_text,
            all_opinions=opinions,
            court=court,
            judgment_style=judgment_style,
            precedents=precedents,
        )

        logger.info(
            f"Deliberation complete. Decision={final_judgment.get('decision','?')} "
            f"Confidence={final_judgment.get('confidence_score', 0)}%"
        )
        return opinions, final_judgment

    def _log_step(self, agent_name: str):
        logger.info(f"JudgePanel: convening {agent_name}…")

    @staticmethod
    def _format_precedents_simple(precedents: List[Dict]) -> str:
        if not precedents:
            return "No precedents retrieved."
        lines = []
        for p in precedents[:6]:
            if p.get("type") == "precedent":
                lines.append(
                    f"• {p.get('title','')} – {p.get('citation','')} "
                    f"[{p.get('court','')}]: {p.get('legal_principle','')[:120]}"
                )
            elif p.get("type") == "statute":
                lines.append(f"• STATUTE: {p.get('title','')} – {p.get('content','')[:120]}")
        return "\n".join(lines) if lines else "No relevant precedents found."
