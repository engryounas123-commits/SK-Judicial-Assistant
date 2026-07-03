"""
AI Judicial Bench Assistant – Pakistan
Main Streamlit Application

Run with:  streamlit run app.py
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Env ─────────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Module imports ───────────────────────────────────────────────────────────
try:
    from modules.ocr             import DocumentProcessor
    from modules.parser          import CaseParser
    from modules.classifier      import CaseClassifier
    from modules.agents          import JudgePanel
    from modules.rag             import LegalRAG
    from modules.judgment_writer import JudgmentWriter
    from modules.dashboard       import DashboardRenderer
    MODULES_OK = True
except ImportError as exc:
    MODULES_OK = False
    logger.critical(f"Module import failed: {exc}")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Judicial Bench Assistant – Pakistan",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI Judicial Bench Assistant | Pakistan Legal AI System v1.0"},
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
def _inject_css():
    st.markdown(
        """
<style>
/* ── Global ─────────────────────────────────────── */
.stApp { background: #F5F0E8; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a2744 0%,#2d4070 100%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color:#BFD3F5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#F5D483 !important; }

/* ── Header ─────────────────────────────────────── */
.jb-header {
    background: linear-gradient(135deg,#1a2744,#2d4070,#1a2744);
    color:#F5D483;
    padding:22px 30px;
    border-radius:12px;
    text-align:center;
    border:2px solid #C6922A;
    margin-bottom:18px;
}
.jb-header h1 { font-size:2.1em; margin:0; color:#F5D483; }
.jb-header p  { color:#BFD3F5;  margin:4px 0 0 0; font-size:.95em; }

/* ── Cards ──────────────────────────────────────── */
.metric-card {
    background:#fff;
    border-radius:10px;
    padding:14px 18px;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
    border-left:4px solid #C6922A;
    margin-bottom:8px;
}

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:#1a2744;
    border-radius:8px 8px 0 0;
    gap:4px;
    padding:4px 6px;
}
.stTabs [data-baseweb="tab"] { color:#BFD3F5; font-weight:500; }
.stTabs [aria-selected="true"] {
    background:#C6922A !important;
    color:#fff !important;
    border-radius:6px;
}

/* ── Buttons ────────────────────────────────────── */
[data-testid="stSidebar"] .stButton button {
    background:#C6922A;
    color:#fff;
    border:none;
    font-weight:600;
    border-radius:6px;
}
[data-testid="stSidebar"] .stButton button:hover { background:#a5791e; }

/* ── Progress ───────────────────────────────────── */
.stProgress > div > div { background:#C6922A !important; }

/* ── Footer ─────────────────────────────────────── */
.jb-footer {
    text-align:center;
    color:#888;
    font-size:.8em;
    padding:16px;
    border-top:1px solid #ddd;
    margin-top:26px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = dict(
        case_files=[],
        extracted_text={},
        merged_text="",
        case_data=None,
        case_type=None,
        agent_opinions={},
        final_judgment=None,
        precedents=[],
        processing_done=False,
        judgment_docx=None,
        confidence_score=0,
        rag_ok=False,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_processor():
    return DocumentProcessor()

@st.cache_resource(show_spinner=False)
def _get_parser():
    return CaseParser()

@st.cache_resource(show_spinner=False)
def _get_classifier():
    return CaseClassifier()

@st.cache_resource(show_spinner=False)
def _get_rag():
    rag = LegalRAG()
    rag.initialize()
    return rag


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
def _render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ Bench Control Panel")
        st.divider()

        # API key
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            label_visibility="collapsed",
            placeholder="sk-…",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.divider()

        # File upload
        st.markdown("### 📁 Case Documents")
        uploaded = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "docx", "jpg", "jpeg", "png", "tiff", "bmp"],
            label_visibility="collapsed",
            help="PDF, DOCX, JPG, PNG – single or multiple files",
        )

        st.divider()

        # Court
        st.markdown("### 🏛️ Court")
        court = st.selectbox(
            "Court",
            [
                "Supreme Court Pakistan",
                "Lahore High Court",
                "Islamabad High Court",
                "Peshawar High Court",
                "Balochistan High Court",
                "Sindh High Court",
                "Federal Shariat Court",
            ],
            label_visibility="collapsed",
        )

        # Style & language
        st.markdown("### 📋 Style / Language")
        style = st.selectbox(
            "Judgment Style",
            ["Full Bench Judgment", "Single Judge Order", "Division Bench",
             "Full Court Reference", "Shariat Bench"],
            label_visibility="collapsed",
        )
        lang = st.selectbox(
            "Language",
            ["English", "Urdu", "Bilingual (En/Ur)"],
            label_visibility="collapsed",
        )

        st.divider()

        # Judge panel
        st.markdown("### 👨‍⚖️ Judge Panel")
        judges = st.multiselect(
            "Active Judges",
            options=[
                "Pakistani Law Expert",
                "Shariah Judge",
                "Domain Specialist Judge",
                "Precedent Research Judge",
                "Chief Justice AI",
            ],
            default=[
                "Pakistani Law Expert",
                "Shariah Judge",
                "Domain Specialist Judge",
                "Precedent Research Judge",
                "Chief Justice AI",
            ],
            label_visibility="collapsed",
        )

        # Model
        st.markdown("### 🤖 AI Model")
        model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            label_visibility="collapsed",
        )
        os.environ["AI_MODEL"] = model

        st.divider()

        # Generate
        can_generate = bool(uploaded and api_key)
        gen_btn = st.button(
            "🔨 Generate Judgment",
            use_container_width=True,
            type="primary",
            disabled=not can_generate,
        )
        if not api_key:
            st.caption("⚠️ Enter API key above")
        if not uploaded:
            st.caption("📁 Upload case files above")

        st.divider()

        # Export
        st.markdown("### 📤 Export")
        dl_btn = st.button("📄 Download DOCX", use_container_width=True)

        # Stats
        if st.session_state.processing_done:
            st.divider()
            st.markdown("### 📊 Session")
            c1, c2 = st.columns(2)
            c1.metric("Files", len(st.session_state.case_files))
            c2.metric("Confidence", f"{st.session_state.confidence_score}%")

    return uploaded, gen_btn, dl_btn, court, style, lang, judges, model


# ═══════════════════════════════════════════════════════════════════════════════
# Welcome screen
# ═══════════════════════════════════════════════════════════════════════════════
def _welcome():
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown(
            """
<div style="text-align:center;padding:30px 10px;">
  <div style="font-size:70px;">⚖️</div>
  <h2 style="color:#1a2744;">Welcome to AI Judicial Bench Assistant</h2>
  <p style="color:#555;font-size:1.05em;">
    Pakistan's AI-Powered Court Judgment Generation System
  </p>
</div>""",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """**🏛️ Supported Courts**
- Supreme Court of Pakistan
- Lahore High Court  
- Islamabad High Court
- Peshawar High Court
- Balochistan High Court
- Sindh High Court
- Federal Shariat Court"""
            )
        with c2:
            st.markdown(
                """**⚖️ AI Judge Panel**
- Pakistani Law Expert Agent
- Shariah Judge Agent
- Domain Specialist Agent
- Precedent Research Agent
- Chief Justice AI (synthesis)"""
            )

        st.markdown("---")
        st.markdown(
            """**📋 How to Use**  
1. Paste your **OpenAI API Key** in the sidebar  
2. **Upload** case documents (PDF / DOCX / images)  
3. Choose **court** and judgment style  
4. Click **Generate Judgment**  
5. Review the judgment and **Download DOCX**"""
        )
        st.markdown("---")

        feature_data = [
            ("📄", "Multi-Format Input", "PDF, DOCX, JPG, PNG, scanned images"),
            ("🔍", "OCR Processing",     "English + Urdu text extraction"),
            ("🤖", "5 AI Agents",        "Internal debate before final ruling"),
            ("📚", "Legal RAG",          "Pakistani precedents & statute database"),
            ("📝", "DOCX Output",        "Formatted court judgment document"),
        ]
        for icon, title, desc in feature_data:
            st.markdown(
                f"""<div class="metric-card">
                    <h4>{icon} {title}</h4>
                    <p style="color:#555;font-size:.9em;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Document processing
# ═══════════════════════════════════════════════════════════════════════════════
def _process_files(uploaded_files) -> Dict:
    processor = _get_processor()
    all_text: Dict = {}
    progress = st.progress(0, text="Processing documents…")

    for i, uf in enumerate(uploaded_files):
        pct = (i + 1) / len(uploaded_files)
        progress.progress(pct, text=f"Reading: {uf.name}")
        try:
            suffix = Path(uf.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            text, meta = processor.process_file(tmp_path, uf.name)
            os.unlink(tmp_path)

            all_text[uf.name] = {"text": text, "metadata": meta, "file_type": suffix,
                                  "word_count": meta.get("word_count", 0),
                                  "ocr_used": meta.get("ocr_used", False)}

        except Exception as exc:
            logger.error(f"File processing error ({uf.name}): {exc}", exc_info=True)
            st.warning(f"⚠️ Could not process {uf.name}: {str(exc)[:120]}")

    progress.empty()
    return all_text


# ═══════════════════════════════════════════════════════════════════════════════
# Judgment generation pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_judgment(court, style, lang, judges, model):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("❌ No OpenAI API key – enter it in the sidebar.")
        return

    merged = st.session_state.merged_text
    case_data = st.session_state.case_data
    case_type = st.session_state.case_type

    # Step 1 – retrieve precedents
    with st.status("📚 Retrieving legal precedents…", expanded=False):
        rag = _get_rag()
        ct_str = case_type.get("primary_type", "civil") if isinstance(case_type, dict) else str(case_type)
        precedents = rag.retrieve_precedents(merged, ct_str, top_k=8)
        st.session_state.precedents = precedents
        st.write(f"✅ {len(precedents)} documents retrieved")

    # Step 2 – judge panel deliberation
    with st.status("⚖️ Convening AI Judge Panel…", expanded=True) as status:
        try:
            panel = JudgePanel(api_key=api_key, model=model)
            opinions, final_judgment = panel.deliberate(
                case_data=case_data,
                case_text=merged,
                case_type=case_type,
                court=court,
                precedents=precedents,
                judgment_style=style,
                selected_judges=judges,
                rag_instance=rag,
            )
            st.session_state.agent_opinions  = opinions
            st.session_state.final_judgment  = final_judgment
            st.session_state.confidence_score = final_judgment.get("confidence_score", 0)
            status.update(label="⚖️ Judge Panel – complete", state="complete")
        except Exception as exc:
            logger.error(f"Judge panel error: {exc}", exc_info=True)
            st.error(f"❌ Judge panel error: {exc}")
            return

    # Step 3 – generate DOCX
    with st.status("📝 Drafting judgment document…", expanded=False):
        try:
            writer = JudgmentWriter()
            docx_bytes = writer.generate_docx(
                case_data=case_data,
                judgment=final_judgment,
                opinions=opinions,
                precedents=precedents,
                court=court,
                judgment_style=style,
                language=lang,
            )
            st.session_state.judgment_docx  = docx_bytes
            st.session_state.processing_done = True
            st.write(f"✅ DOCX generated ({len(docx_bytes):,} bytes)")
        except Exception as exc:
            logger.error(f"DOCX generation error: {exc}", exc_info=True)
            st.error(f"❌ DOCX error: {exc}")

    st.success("✅ Judgment generated successfully!")
    st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not MODULES_OK:
        st.error(
            "❌ One or more modules failed to import. "
            "Run `pip install -r requirements.txt` and restart."
        )
        return

    _inject_css()
    _init_state()

    # Header
    st.markdown(
        """<div class="jb-header">
            <h1>⚖️ AI Judicial Bench Assistant</h1>
            <p>Pakistan Legal AI System | Multi-Court Judgment Generator</p>
            <p style="font-size:.8em;color:#9ab0d0;">
              Multi-Agent AI · Pakistani Law · Shariah Principles · Legal RAG · DOCX Output
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Sidebar
    uploaded, gen_btn, dl_btn, court, style, lang, judges, model = _render_sidebar()

    # Show welcome if nothing uploaded yet
    if not uploaded and not st.session_state.processing_done:
        _welcome()
        return

    # ── Process uploaded files ───────────────────────────────────────────────
    if uploaded:
        file_names = [f.name for f in uploaded]
        new_files  = file_names != st.session_state.case_files

        if new_files:
            st.session_state.case_files      = file_names
            st.session_state.processing_done = False
            st.session_state.final_judgment  = None
            st.session_state.agent_opinions  = {}

            with st.spinner("📖 Extracting text from documents…"):
                all_text = _process_files(uploaded)
                st.session_state.extracted_text = all_text

            if not all_text:
                st.error("❌ No text could be extracted from the uploaded files.")
                return

            merged = "\n\n".join(v["text"] for v in all_text.values())
            st.session_state.merged_text = merged

            with st.spinner("🔍 Parsing case entities…"):
                parser = _get_parser()
                st.session_state.case_data = parser.parse_case(merged)

            with st.spinner("📂 Classifying case type…"):
                clf = _get_classifier()
                st.session_state.case_type = clf.classify(merged, st.session_state.case_data)

            st.success(f"✅ {len(all_text)} document(s) ready. Click **Generate Judgment**.")

    # ── Generate judgment ────────────────────────────────────────────────────
    if gen_btn:
        if not st.session_state.case_data:
            st.warning("Upload and process files first.")
        else:
            _generate_judgment(court, style, lang, judges, model)

    # ── Dashboard ────────────────────────────────────────────────────────────
    if st.session_state.case_data:
        renderer = DashboardRenderer()
        renderer.render_full_dashboard(
            case_data        = st.session_state.case_data,
            case_type        = st.session_state.case_type,
            agent_opinions   = st.session_state.agent_opinions,
            final_judgment   = st.session_state.final_judgment,
            precedents       = st.session_state.precedents,
            court            = court,
            confidence_score = st.session_state.confidence_score,
            extracted_text   = st.session_state.extracted_text,
        )

    # ── Download buttons ─────────────────────────────────────────────────────
    if dl_btn or (st.session_state.judgment_docx and st.session_state.processing_done):
        if st.session_state.judgment_docx:
            fname = (
                f"judgment_{court.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            st.download_button(
                label     = "📄 Download Judgment (DOCX)",
                data      = st.session_state.judgment_docx,
                file_name = fname,
                mime      = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            st.caption(
                "💡 For PDF: open the DOCX in Microsoft Word or LibreOffice → Save As PDF"
            )
        else:
            st.info("Generate a judgment first before downloading.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        """<div class="jb-footer">
        ⚖️ AI Judicial Bench Assistant &nbsp;|&nbsp; Pakistan Legal AI &nbsp;|&nbsp;
        For Research & Reference Only &nbsp;|&nbsp; Not a Substitute for Qualified Legal Counsel
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
