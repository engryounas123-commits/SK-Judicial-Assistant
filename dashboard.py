"""
Dashboard Renderer Module
Renders the full Streamlit case dashboard:
  Tab 1 – Case Snapshot
  Tab 2 – Facts & Timeline
  Tab 3 – Legal Analysis
  Tab 4 – Judge Panel Opinions
  Tab 5 – Precedents
  Tab 6 – Final Judgment
"""

import json
import logging
from typing import Dict, List, Optional, Any

import streamlit as st

logger = logging.getLogger(__name__)


class DashboardRenderer:
    """Renders all dashboard components into Streamlit."""

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def render_full_dashboard(
        self,
        case_data: Dict,
        case_type: Optional[Dict],
        agent_opinions: Dict[str, str],
        final_judgment: Optional[Dict],
        precedents: List[Dict],
        court: str,
        confidence_score: int,
        extracted_text: Dict,
    ):
        if not case_data:
            st.info("Upload case files and click **Generate Judgment** to begin.")
            return

        tabs = st.tabs([
            "📋 Case Snapshot",
            "🗂️ Facts & Timeline",
            "⚖️ Legal Analysis",
            "👨‍⚖️ Judge Panel",
            "📚 Precedents",
            "📜 Final Judgment",
        ])

        with tabs[0]:
            self._render_snapshot(case_data, case_type, court, confidence_score, extracted_text)

        with tabs[1]:
            self._render_facts_timeline(case_data)

        with tabs[2]:
            self._render_legal_analysis(case_data, case_type, final_judgment)

        with tabs[3]:
            self._render_judge_panel(agent_opinions)

        with tabs[4]:
            self._render_precedents(precedents)

        with tabs[5]:
            self._render_final_judgment(final_judgment)

    # ------------------------------------------------------------------
    # Tab 1 – Case Snapshot
    # ------------------------------------------------------------------

    def _render_snapshot(
        self,
        case_data: Dict,
        case_type: Optional[Dict],
        court: str,
        confidence_score: int,
        extracted_text: Dict,
    ):
        st.markdown("### 📋 Case Snapshot")

        parties = case_data.get("parties", {})
        identifiers = case_data.get("case_identifiers", {})

        # Row 1 – Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🏛️ Court", court.replace(" ", "\n") if len(court) > 20 else court)
        with c2:
            pt = parties.get("plaintiff_petitioner", "N/A")
            st.metric("👤 Petitioner / Plaintiff", pt[:30] + "…" if len(pt) > 30 else pt)
        with c3:
            rd = parties.get("defendant_respondent", "N/A")
            st.metric("👤 Respondent / Defendant", rd[:30] + "…" if len(rd) > 30 else rd)
        with c4:
            conf_color = "🟢" if confidence_score >= 80 else "🟡" if confidence_score >= 60 else "🔴"
            st.metric(f"{conf_color} AI Confidence", f"{confidence_score}%")

        st.divider()

        # Row 2 – Case type & identifiers
        c1, c2, c3 = st.columns(3)
        with c1:
            if case_type and isinstance(case_type, dict):
                primary = case_type.get("primary_type", "N/A").upper()
                sub = case_type.get("sub_type", "")
                st.metric("📂 Case Type", primary)
                if sub:
                    st.caption(f"Sub-type: {sub}")
            else:
                st.metric("📂 Case Type", str(case_type or "Unclassified").upper())

        with c2:
            case_no = identifiers.get("case_number", "N/A")
            st.metric("🔢 Case Number", case_no or "N/A")

        with c3:
            year = identifiers.get("year", "N/A")
            st.metric("📅 Year", year or "N/A")

        st.divider()

        # Row 3 – Document summary
        st.markdown("#### 📁 Uploaded Documents")
        if extracted_text:
            cols = st.columns(min(len(extracted_text), 4))
            for i, (fname, fdata) in enumerate(extracted_text.items()):
                with cols[i % 4]:
                    meta = fdata if isinstance(fdata, dict) else {}
                    wc = meta.get("word_count", meta.get("metadata", {}).get("word_count", "?"))
                    ocr = meta.get("ocr_used", meta.get("metadata", {}).get("ocr_used", False))
                    ftype = meta.get("file_type", "").lstrip(".").upper() or "FILE"
                    ocr_badge = " 🔍 OCR" if ocr else ""
                    st.markdown(
                        f"""<div style="background:#f0f4ff;border-left:4px solid #C6922A;
                        padding:10px;border-radius:6px;margin:4px 0;">
                        <b>{fname[:30]}</b><br/>
                        <small>{ftype}{ocr_badge} | {wc} words</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Monetary amounts
        amounts = case_data.get("monetary_amounts", [])
        if amounts:
            st.divider()
            st.markdown("#### 💰 Monetary Amounts Detected")
            amt_cols = st.columns(min(len(amounts), 4))
            for i, a in enumerate(amounts[:8]):
                with amt_cols[i % 4]:
                    st.metric("Amount", a.get("amount", ""), help=a.get("context", ""))

    # ------------------------------------------------------------------
    # Tab 2 – Facts & Timeline
    # ------------------------------------------------------------------

    def _render_facts_timeline(self, case_data: Dict):
        st.markdown("### 🗂️ Key Facts & Timeline")

        c1, c2 = st.columns([3, 2])

        with c1:
            st.markdown("#### 📌 Key Facts")
            facts = case_data.get("key_facts", [])
            if facts:
                for i, fact in enumerate(facts, 1):
                    st.markdown(
                        f"""<div style="background:#fff8e7;border-left:4px solid #C6922A;
                        padding:10px 14px;border-radius:6px;margin:6px 0;">
                        <b>{i}.</b> {fact}
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No key facts automatically extracted. Review case text manually.")

            # Evidence
            st.markdown("#### 🔎 Evidence Items")
            evidence = case_data.get("evidence_items", [])
            if evidence:
                for ev in evidence:
                    st.markdown(f"• {ev}")
            else:
                st.caption("No evidence items detected.")

        with c2:
            st.markdown("#### 🕐 Case Timeline")
            timeline = case_data.get("timeline", [])
            if timeline:
                for event in timeline:
                    date_str = event.get("date", "")
                    ev_text = event.get("event", "")
                    st.markdown(
                        f"""<div class="timeline-item" style="border-left:3px solid #C6922A;
                        padding-left:12px;margin:8px 0;position:relative;">
                        <span style="color:#C6922A;font-weight:bold;">{date_str}</span><br/>
                        <span style="font-size:0.9em;color:#444;">{ev_text[:100]}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No timeline events extracted.")

            # Witnesses
            witnesses = case_data.get("witnesses", [])
            if witnesses:
                st.markdown("#### 👥 Witnesses")
                for w in witnesses:
                    st.markdown(f"**{w.get('id', '')}:** {w.get('name', '')}")

            # FIR
            fir = case_data.get("fir_details")
            if fir:
                st.markdown("#### 🚔 FIR Details")
                st.markdown(f"**FIR No.:** {fir.get('fir_number', 'N/A')}")
                st.markdown(f"**Date:** {fir.get('fir_date', 'N/A')}")
                st.markdown(f"**Police Station:** {fir.get('police_station', 'N/A')}")

    # ------------------------------------------------------------------
    # Tab 3 – Legal Analysis
    # ------------------------------------------------------------------

    def _render_legal_analysis(
        self, case_data: Dict, case_type: Optional[Dict], final_judgment: Optional[Dict]
    ):
        st.markdown("### ⚖️ Legal Analysis")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📜 Legal Sections Cited")
            sections = case_data.get("legal_sections", [])
            if sections:
                for s in sections:
                    sec = s.get("section", "")
                    law = s.get("law", "")
                    st.markdown(
                        f"""<div style="background:#f0f4ff;border-left:3px solid #1a2744;
                        padding:8px 12px;border-radius:5px;margin:4px 0;font-size:0.9em;">
                        <b>Section {sec}</b> – {law or "Unspecified Law"}
                        </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No specific sections detected.")

            # Relief sought
            relief = case_data.get("relief_sought", "")
            if relief:
                st.markdown("#### 🙏 Relief Sought")
                st.markdown(
                    f"""<div style="background:#fff3cd;border:1px solid #C6922A;
                    border-radius:8px;padding:12px;font-size:0.9em;">{relief}</div>""",
                    unsafe_allow_html=True,
                )

        with c2:
            # Legal claims
            st.markdown("#### ⚖️ Legal Claims Identified")
            claims = case_data.get("legal_claims", [])
            if claims:
                for i, cl in enumerate(claims, 1):
                    st.markdown(f"{i}. {cl}")
            else:
                st.info("No specific legal claims parsed.")

            # Case type breakdown
            if case_type and isinstance(case_type, dict):
                st.markdown("#### 📊 Case Classification")
                st.markdown(
                    f"**Primary:** {case_type.get('primary_type','?').title()}"
                )
                secondary = case_type.get("secondary_types", [])
                if secondary:
                    st.markdown(f"**Also involves:** {', '.join(secondary)}")
                st.markdown(
                    f"**Classification confidence:** {case_type.get('confidence', 0)}%"
                )
                st.progress(case_type.get("confidence", 0) / 100)

        # Final judgment applicable laws
        if final_judgment:
            laws = final_judgment.get("applicable_laws", [])
            if laws:
                st.divider()
                st.markdown("#### 📖 Applicable Laws (per AI Judgment)")
                cols = st.columns(2)
                for i, law in enumerate(laws):
                    with cols[i % 2]:
                        st.markdown(
                            f"""<div style="background:#e8f5e9;border-left:3px solid #28a745;
                            padding:8px 12px;border-radius:5px;margin:4px 0;font-size:0.9em;">
                            {law}
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # ------------------------------------------------------------------
    # Tab 4 – Judge Panel Opinions
    # ------------------------------------------------------------------

    def _render_judge_panel(self, agent_opinions: Dict[str, str]):
        st.markdown("### 👨‍⚖️ AI Judge Panel Deliberations")

        if not agent_opinions:
            st.info("Generate judgment to see panel opinions.")
            return

        agent_icons = {
            "Pakistani Law Expert":    "📗",
            "Shariah Judge":           "☪️",
            "Domain Specialist Judge": "🔬",
            "Precedent Research Judge":"📚",
            "Chief Justice AI":        "⚖️",
        }

        agent_colors = {
            "Pakistani Law Expert":    "#1a2744",
            "Shariah Judge":           "#2d6a27",
            "Domain Specialist Judge": "#6b2d2d",
            "Precedent Research Judge":"#2d4a6b",
            "Chief Justice AI":        "#4a1a6b",
        }

        for agent_name, opinion in agent_opinions.items():
            icon = agent_icons.get(agent_name, "👨‍⚖️")
            color = agent_colors.get(agent_name, "#1a2744")

            with st.expander(f"{icon} {agent_name}", expanded=(agent_name == "Chief Justice AI")):
                st.markdown(
                    f"""<div style="border-left:5px solid {color};padding-left:15px;">
                    <h4 style="color:{color};">{icon} {agent_name}</h4>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if opinion.startswith("[") and "API error" in opinion:
                    st.error(opinion)
                else:
                    st.markdown(opinion)

    # ------------------------------------------------------------------
    # Tab 5 – Precedents
    # ------------------------------------------------------------------

    def _render_precedents(self, precedents: List[Dict]):
        st.markdown("### 📚 Legal Precedents & Statutes Retrieved")

        if not precedents:
            st.info("No precedents retrieved. Generate judgment to search the legal database.")
            return

        preds = [p for p in precedents if p.get("type") == "precedent"]
        stats = [p for p in precedents if p.get("type") == "statute"]
        maxims = [p for p in precedents if p.get("type") == "maxim"]

        if preds:
            st.markdown(f"#### ⚖️ Case Precedents ({len(preds)} found)")
            for p in preds:
                verified = p.get("verified", False)
                badge_color = "#28a745" if verified else "#ffc107"
                badge_text  = "✔ VERIFIED" if verified else "⚠ UNVERIFIED"

                st.markdown(
                    f"""<div style="background:#fff8e7;border:1px solid #C6922A;
                    border-radius:8px;padding:14px;margin:8px 0;">
                    <b style="color:#1a2744;">{p.get('title','')}</b>
                    <span style="background:{badge_color};color:white;padding:2px 8px;
                    border-radius:10px;font-size:0.75em;margin-left:10px;">{badge_text}</span>
                    <br/>
                    <small style="color:#C6922A;"><b>Citation:</b> {p.get('citation','N/A')}</small>
                    &nbsp;|&nbsp;
                    <small><b>Court:</b> {p.get('court','N/A')} ({p.get('year','')})</small>
                    <br/>
                    <small><b>Judge:</b> {p.get('judge','N/A')}</small>
                    <br/><br/>
                    <b>Legal Principle:</b> {p.get('legal_principle','N/A')}
                    </div>""",
                    unsafe_allow_html=True,
                )

        if stats:
            st.divider()
            st.markdown(f"#### 📖 Relevant Statutes ({len(stats)} found)")
            for s in stats:
                with st.expander(s.get("title", "Statute")):
                    st.markdown(s.get("content", "N/A"))
                    st.caption(f"Source: {s.get('court_source','All Courts')}")

        if maxims:
            st.divider()
            st.markdown("#### 🏛️ Applicable Legal Maxims")
            for m in maxims:
                st.markdown(
                    f"""<div style="background:#f0f4ff;border-left:4px solid #1a2744;
                    padding:10px;border-radius:5px;margin:5px 0;font-style:italic;">
                    {m.get('content','')}
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------
    # Tab 6 – Final Judgment
    # ------------------------------------------------------------------

    def _render_final_judgment(self, final_judgment: Optional[Dict]):
        st.markdown("### 📜 Final AI Judgment")

        if not final_judgment:
            st.info("Click **Generate Judgment** in the sidebar to produce the final judgment.")
            return

        decision = final_judgment.get("decision", "RESERVED")
        confidence = final_judgment.get("confidence_score", 0)

        # Decision banner
        decision_colors = {
            "ALLOWED": ("#d4edda", "#155724", "#28a745"),
            "DISMISSED": ("#f8d7da", "#721c24", "#dc3545"),
            "PARTIALLY ALLOWED": ("#fff3cd", "#856404", "#ffc107"),
        }
        bg, fg, border = decision_colors.get(
            decision, ("#e8f0fe", "#1a2744", "#1a2744")
        )

        st.markdown(
            f"""<div style="background:{bg};border:2px solid {border};border-radius:10px;
            padding:20px;text-align:center;margin:10px 0;">
            <h2 style="color:{fg};margin:0;">⚖️ DECISION: {decision}</h2>
            <p style="color:{fg};margin:5px 0 0 0;">
            AI Confidence Score: <b>{confidence}%</b> &nbsp;|&nbsp;
            Court: <b>{final_judgment.get('primary_court', 'N/A')}</b>
            </p></div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        # Operative order
        operative = final_judgment.get("operative_order", "")
        if operative:
            st.markdown("#### 📌 Operative Order")
            st.markdown(
                f"""<div style="background:#fffde7;border:2px solid #C6922A;
                border-radius:10px;padding:20px;font-family:Georgia,serif;line-height:1.7;">
                {operative}
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # Three-column layout for details
        c1, c2 = st.columns(2)

        with c1:
            # Issues framed
            issues = final_judgment.get("issues_framed", [])
            if issues:
                st.markdown("#### ❓ Issues Framed")
                for i, issue in enumerate(issues, 1):
                    st.markdown(f"**{i}.** {issue}")

            # Findings
            findings = final_judgment.get("findings", [])
            if findings:
                st.markdown("#### ✅ Findings")
                for finding in findings:
                    st.markdown(
                        f"""<div style="background:#e8f5e9;border-left:3px solid #28a745;
                        padding:8px 12px;border-radius:5px;margin:4px 0;">
                        {finding}
                        </div>""",
                        unsafe_allow_html=True,
                    )

        with c2:
            # Relief granted
            relief = final_judgment.get("relief_granted", [])
            if relief:
                st.markdown("#### 🎯 Relief Granted")
                for r in relief:
                    st.markdown(f"• {r}")

            # Directives
            directives = final_judgment.get("directives", [])
            if directives:
                st.markdown("#### 📋 Court Directives")
                for d in directives:
                    st.markdown(
                        f"""<div style="background:#fff3cd;border-left:3px solid #C6922A;
                        padding:8px 12px;border-radius:5px;margin:4px 0;">
                        {d}
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # Costs
            costs = final_judgment.get("costs", "")
            if costs:
                st.markdown("#### 💼 Order as to Costs")
                st.markdown(costs)

        # Legal reasoning expander
        reasoning = final_judgment.get("legal_reasoning", "")
        if reasoning:
            with st.expander("📖 Full Legal Reasoning"):
                st.markdown(reasoning)

        # Precedents applied
        preds_applied = final_judgment.get("precedents_applied", [])
        if preds_applied:
            with st.expander(f"📚 Precedents Applied ({len(preds_applied)})"):
                for p in preds_applied:
                    verified = p.get("verified", False)
                    badge = "✔ Verified" if verified else "⚠ Requires Verification"
                    badge_color = "green" if verified else "orange"
                    st.markdown(
                        f"""**{p.get('citation','N/A')}** — :{badge_color}[{badge}]
                        \n*{p.get('principle','')}*"""
                    )

        # Shariah aspects
        shariah = final_judgment.get("shariah_aspects", "")
        if shariah and shariah != "N/A":
            with st.expander("☪️ Shariah / Islamic Law Aspects"):
                st.markdown(shariah)

        # Confidence progress
        st.divider()
        st.markdown("#### 📊 Judgment Confidence Breakdown")
        conf_cols = st.columns(4)
        labels = ["Evidence", "Legal Basis", "Precedents", "Overall"]
        values = [
            min(100, confidence + 5),
            min(100, confidence - 5),
            min(100, confidence),
            confidence,
        ]
        for col, label, val in zip(conf_cols, labels, values):
            with col:
                color = "green" if val >= 80 else "orange" if val >= 60 else "red"
                st.metric(label, f"{val}%")
                st.progress(val / 100)

        # Notes
        notes = final_judgment.get("notes", "")
        if notes:
            st.caption(f"⚠️ Note: {notes}")

        # Disclaimer
        st.markdown(
            """<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
            padding:15px;margin-top:20px;font-size:0.85em;color:#666;">
            <b>⚠️ DISCLAIMER:</b> This AI-generated judgment is for research and reference purposes only.
            It is NOT a legally binding court order. Consult qualified Pakistani legal counsel for
            actual legal advice. AI-generated citations marked UNVERIFIED must be independently
            confirmed before reliance.
            </div>""",
            unsafe_allow_html=True,
        )
