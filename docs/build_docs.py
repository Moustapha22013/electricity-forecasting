"""Build the deliverables from their sources:  python docs/build_docs.py

* docs/article.md  -> docs/article.pdf  (markdown -> HTML -> PDF with weasyprint)
* docs/slides.py   -> docs/soutenance.pptx

Documentation-only dependencies: requirements-docs.txt (not needed to run the experiments).
"""
from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 1.7cm 1.6cm 1.7cm 1.6cm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 9.6pt; line-height: 1.36; color: #111; }
h1 { font-size: 15.5pt; text-align: center; margin: 0 0 4pt 0; line-height: 1.2; }
h1 + p, h1 + p + p { text-align: center; margin: 2pt 0; }
h2 { font-size: 11pt; margin: 10pt 0 3pt 0; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
p { margin: 3pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 2pt 0; padding-left: 15pt; } li { margin: 1pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 5pt 0; }
th, td { border: 0.5pt solid #aaa; padding: 2pt 4pt; } th { background: #eee; }
img { display: block; margin: 5pt auto 1pt auto; max-width: 82%; }
img + em, p > em:only-child { font-size: 8pt; }
code { font-size: 8.4pt; font-family: 'DejaVu Sans Mono', monospace; }
hr { border: 0; border-top: 0.6pt solid #999; margin: 5pt 0; }
.caption { text-align: center; font-size: 8pt; color: #333; margin-bottom: 6pt; }
"""

LATEX = [
    (r"\\hat\s*y", "ŷ"), (r"\\overline\{P\}", "P̄"), (r"\\sqrt\{([^}]*)\}", r"√(\1)"), (r"\\ge", "≥"), (r"\\le", "≤"),
    (r"\\times", "×"), (r"\\text\{([^}]*)\}", r"\1"), (r"\\,", " "), (r"\\in", "∈"), (r"\\dots", "…"),
]


def _math(m: re.Match) -> str:
    s = m.group(1)
    for a, b in LATEX:
        s = re.sub(a, b, s)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", s)
    return f"<i>{s}</i>"


def build_article() -> Path:
    import markdown
    from weasyprint import HTML
    src = (DOCS / "article.md").read_text(encoding="utf-8")
    src = re.sub(r"(?<!\\)\$([^$]+)\$", _math, src)
    html = markdown.markdown(src, extensions=["tables"])
    # figure captions: ![caption](path) -> image + numbered caption
    n = [0]

    def cap(m: re.Match) -> str:
        n[0] += 1
        return f'<img src="{m.group(2)}" alt=""><p class="caption"><b>Figure {n[0]}.</b> {m.group(1)}</p>'
    html = re.sub(r'<img alt="([^"]*)" src="([^"]*)" ?/?>', cap, html)
    doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    out = DOCS / "article.pdf"
    pdf = HTML(string=doc, base_url=str(DOCS)).render()
    pdf.write_pdf(out)
    print(f"article.pdf: {len(pdf.pages)} pages")
    return out


if __name__ == "__main__":
    build_article()
    from slides import build_slides  # noqa: E402  (docs/slides.py)
    build_slides()
