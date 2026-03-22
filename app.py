import gradio as gr
import yaml
import re
from pypdf import PdfReader
from src.pipeline import ReviewerPipeline

# ── Load pipeline once ───────────────────────────────────────────────────────
pipeline = ReviewerPipeline(config_path="config/config.yaml")

# ── PDF text extractor ───────────────────────────────────────────────────────
def extract_from_pdf(pdf_path: str) -> tuple[str, str]:
    """Extracts title and abstract from an uploaded PDF."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages[:4]:
        full_text += page.extract_text() or ""

    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    # Title: first non-trivial line
    title = ""
    for line in lines:
        if len(line) > 10 and not line.lower().startswith("abstract"):
            title = line
            break

    # Abstract: text after "Abstract" keyword
    abstract = ""
    joined = " ".join(lines)
    match = re.search(
        r'[Aa]bstract[\s\.\-:]*(.+?)(?=\n\n|\d+\s+[A-Z]|Introduction|INTRODUCTION)',
        joined
    )
    if match:
        abstract = match.group(1).strip()
    else:
        idx = joined.find(title)
        if idx != -1:
            abstract = joined[idx + len(title):idx + len(title) + 1500].strip()

    return title[:200], abstract[:2000]

# ── Helpers ──────────────────────────────────────────────────────────────────
def grade_color(grade: str) -> str:
    return {"Strong": "#22c55e", "Moderate": "#f59e0b",
            "Weak": "#ef4444", "Insufficient": "#dc2626"}.get(grade, "#888")

def _bar(label: str, val: int, c1: str, c2: str) -> str:
    return f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="color:#9090bb;font-size:12px;">{label}</span>
            <span style="color:#e8e8f4;font-size:12px;font-weight:600;">{val}%</span>
        </div>
        <div style="background:#1a1a2e;border-radius:5px;height:7px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,{c1},{c2});width:{val}%;height:100%;border-radius:5px;"></div>
        </div>
    </div>"""

def _citation_card(cite: dict, i: int) -> str:
    score     = cite.get("similarity_score", 0)
    score_pct = int(score * 100)
    color     = "#22c55e" if score >= 0.85 else "#f59e0b" if score >= 0.75 else "#ef4444"
    authors   = cite.get("authors", [])
    author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
    return f"""
    <div style="background:#0e0e1c;border:1px solid #1e1e30;border-radius:12px;
                padding:16px 18px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;min-width:0;">
                <div style="color:#6c63ff;font-size:10px;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:5px;">#{i+1}</div>
                <div style="color:#ddddf4;font-size:13px;font-weight:600;line-height:1.4;
                            margin-bottom:5px;word-break:break-word;">
                    {cite.get('title','Untitled')}
                </div>
                <div style="color:#55558a;font-size:11px;margin-bottom:8px;">
                    {author_str} · {cite.get('published','N/A')}
                </div>
                <a href="{cite.get('url','#')}" target="_blank"
                   style="color:#06b6d4;font-size:11px;text-decoration:none;
                          border-bottom:1px solid #06b6d433;">View on arXiv →</a>
            </div>
            <div style="text-align:center;min-width:52px;flex-shrink:0;">
                <div style="color:{color};font-size:20px;font-weight:800;line-height:1;">
                    {score_pct}<span style="font-size:11px;color:#55558a;">%</span>
                </div>
                <div style="color:#55558a;font-size:10px;margin-top:2px;">match</div>
            </div>
        </div>
    </div>"""

def _error(msg: str) -> str:
    return f"""
    <div style="font-family:'DM Sans',sans-serif;background:#1a0808;border:1px solid #4a1010;
                border-radius:12px;padding:16px 20px;color:#f87171;font-size:14px;">
        ⚠ {msg}
    </div>"""

# ── Main function ────────────────────────────────────────────────────────────
def run_review(pdf_file):
    if pdf_file is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value=_error("Please upload a PDF file first.")),
        )
    try:
        title, abstract = extract_from_pdf(pdf_file)
        if not abstract.strip():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True, value=_error(
                    "Could not extract text from this PDF. Make sure it's not a scanned image PDF."
                )),
            )

        result    = pipeline.review(title=title, abstract=abstract)
        q         = result["quality"]
        citations = result["missing_citations"]
        grade     = q.get("grade", "N/A")
        color     = grade_color(grade)
        c_val     = int(q.get("coherence",    0) * 100)
        cp_val    = int(q.get("completeness", 0) * 100)
        n_val     = int(q.get("novelty",      0) * 100)
        f_val     = int(q.get("final_score",  0) * 100)

        # ── Quality panel ─────────────────────────────────────────────────
        quality_html = f"""
        <div style="font-family:'DM Sans',sans-serif;display:flex;flex-direction:column;gap:14px;">

            <div style="background:#111120;border:1px solid #1e1e35;border-radius:14px;padding:18px 20px;">
                <div style="color:#55558a;font-size:10px;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:8px;font-family:'DM Mono',monospace;">
                    Paper
                </div>
                <div style="color:#e8e8f4;font-size:14px;font-weight:600;line-height:1.4;margin-bottom:8px;">
                    {title[:100]}{"..." if len(title)>100 else ""}
                </div>
                <div style="color:#6060aa;font-size:12px;line-height:1.6;">
                    {abstract[:250]}{"..." if len(abstract)>250 else ""}
                </div>
            </div>

            <div style="background:#111120;border:1px solid #1e1e35;border-radius:14px;padding:20px 22px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
                    <div style="color:#e8e8f4;font-size:14px;font-weight:700;">Quality Assessment</div>
                    <span style="background:{color}22;color:{color};border:1px solid {color}55;
                                 padding:4px 12px;border-radius:20px;font-size:10px;font-weight:800;
                                 letter-spacing:0.1em;text-transform:uppercase;">{grade}</span>
                </div>
                {_bar("Coherence",    c_val,  "#8b5cf6", "#a78bfa")}
                {_bar("Completeness", cp_val, "#06b6d4", "#38bdf8")}
                {_bar("Novelty",      n_val,  "#f59e0b", "#fbbf24")}
                <div style="background:#0d0d1c;border:1px solid #1e1e35;border-radius:10px;
                            padding:14px 18px;display:flex;align-items:center;
                            justify-content:space-between;margin-top:4px;">
                    <span style="color:#7070aa;font-size:13px;font-weight:500;">Overall Score</span>
                    <span style="color:{color};font-size:26px;font-weight:800;">
                        {f_val}<span style="font-size:13px;font-weight:400;color:#7070aa;">%</span>
                    </span>
                </div>
            </div>
        </div>
        """

        # ── Citations panel ───────────────────────────────────────────────
        if not citations:
            citations_html = """
            <div style="font-family:'DM Sans',sans-serif;background:#111120;border:1px solid #1e1e35;
                        border-radius:14px;padding:48px;text-align:center;color:#55558a;">
                <div style="font-size:28px;margin-bottom:10px;">✓</div>
                <div style="font-size:13px;">No missing citations found.</div>
            </div>"""
        else:
            cards = "".join(_citation_card(c, i) for i, c in enumerate(citations))
            citations_html = f"""
            <div style="font-family:'DM Sans',sans-serif;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
                    <div style="color:#e8e8f4;font-size:14px;font-weight:700;">Missing Citations</div>
                    <span style="background:#6c63ff22;color:#a78bfa;border:1px solid #6c63ff44;
                                 padding:3px 12px;border-radius:20px;font-size:10px;font-weight:700;">
                        {len(citations)} found
                    </span>
                </div>
                <div style="max-height:560px;overflow-y:auto;padding-right:2px;">
                    {cards}
                </div>
            </div>"""

        return (
            gr.update(visible=True, value=quality_html),
            gr.update(visible=True, value=citations_html),
            gr.update(visible=False),
        )

    except Exception as e:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value=_error(str(e))),
        )

def clear_all():
    return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# ── CSS ──────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #08080f !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
}
#sl-header {
    text-align: center;
    padding: 50px 32px 42px;
    background: #0a0a14;
    border: 1px solid #18182a;
    border-radius: 20px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#sl-header::after {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 240px;
    background: radial-gradient(ellipse, #6c63ff12 0%, transparent 65%);
    pointer-events: none;
}
#sl-header h1 {
    font-size: 46px !important;
    font-weight: 800 !important;
    color: #e8e8f4 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 10px !important;
    line-height: 1 !important;
}
#sl-header p {
    color: #44446a !important;
    font-size: 12px !important;
    letter-spacing: 0.12em !important;
    font-weight: 500 !important;
    margin: 0 !important;
}
.lens { color: #6c63ff; }

/* Left panel card */
#left-panel {
    background: #0a0a14;
    border: 1px solid #18182a;
    border-radius: 16px;
    padding: 22px;
}

/* Upload component */
.upload-box {
    background: #0d0d1a !important;
    border: 2px dashed #1e1e35 !important;
    border-radius: 12px !important;
}
.upload-box:hover { border-color: #6c63ff66 !important; }

/* Buttons */
#analyse-btn {
    background: linear-gradient(135deg,#6c63ff,#8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-size: 14px !important;
    font-weight: 700 !important; padding: 13px !important;
    width: 100% !important; cursor: pointer !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
#analyse-btn:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

#clear-btn {
    background: transparent !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 10px !important; color: #44446a !important;
    font-size: 13px !important; padding: 11px !important;
    width: 100% !important; cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s !important;
}
#clear-btn:hover { border-color: #6c63ff !important; color: #a78bfa !important; }

.output-panel { background: transparent !important; border: none !important; padding: 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a12; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6c63ff; }
"""

# ── Build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="ScholarLens") as demo:

    gr.HTML("""
    <div id="sl-header">
        <h1>Scholar<span class="lens">Lens</span></h1>
        <p>AUTOMATED SCIENTIFIC QUALITY ASSESSMENT &amp; MISSING CITATION DETECTION</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # Left: upload + controls
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div id="left-panel">')

            pdf_input = gr.File(
                label="Research Paper (PDF)",
                file_types=[".pdf"],
                elem_classes=["upload-box"],
            )

            analyse_btn = gr.Button("Analyse Paper", elem_id="analyse-btn", variant="primary")
            clear_btn   = gr.Button("Clear", elem_id="clear-btn")

            gr.HTML("""
            <div style="margin-top:20px;border-top:1px solid #18182a;padding-top:18px;">
                <div style="color:#33335a;font-size:10px;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;margin-bottom:12px;">How it works</div>
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div style="display:flex;gap:10px;">
                        <span style="color:#6c63ff;font-size:11px;font-weight:800;min-width:16px;padding-top:1px;">1</span>
                        <span style="color:#55558a;font-size:12px;line-height:1.5;">Upload your paper as PDF</span>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <span style="color:#6c63ff;font-size:11px;font-weight:800;min-width:16px;padding-top:1px;">2</span>
                        <span style="color:#55558a;font-size:12px;line-height:1.5;">SciBERT encodes contributions</span>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <span style="color:#6c63ff;font-size:11px;font-weight:800;min-width:16px;padding-top:1px;">3</span>
                        <span style="color:#55558a;font-size:12px;line-height:1.5;">arXiv searched for recent work</span>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <span style="color:#6c63ff;font-size:11px;font-weight:800;min-width:16px;padding-top:1px;">4</span>
                        <span style="color:#55558a;font-size:12px;line-height:1.5;">Missing citations ranked by similarity</span>
                    </div>
                </div>
            </div>
            """)

            gr.HTML('</div>')

        # Middle: quality + paper info
        with gr.Column(scale=2, min_width=320):
            error_out   = gr.HTML(visible=False)
            quality_out = gr.HTML(visible=False, elem_classes=["output-panel"])

        # Right: citations
        with gr.Column(scale=2, min_width=320):
            citations_out = gr.HTML(visible=False, elem_classes=["output-panel"])

    # Events
    analyse_btn.click(
        fn=run_review,
        inputs=[pdf_input],
        outputs=[quality_out, citations_out, error_out],
    )
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[pdf_input, quality_out, citations_out, error_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)