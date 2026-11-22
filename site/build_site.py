"""
Build the static multi-page project site from the content fragments in
site/content/. This presents what has actually been built so far (data
sourcing, neuron alignment, split design, modeling infrastructure) and
explicitly states that no propagation data or scientific result exists
yet - it does not run, re-derive, or alter any analysis.

index.html is a special case: a standalone welcome/hero page with no
navigation, banner, or other content, built from site/landing_layout.html
(a bare skeleton, not the shared layout.html every other page uses). All
real pages - starting with overview.html, which holds the content that
used to live below the hero on index.html - use the shared layout and
nav, and a visitor returns to the welcome page via the site logo link in
that shared header, or an explicit "Back to welcome" link on Overview.

Usage:
    python site/build_site.py
"""
from __future__ import annotations

import pathlib

SITE_DIR = pathlib.Path(__file__).resolve().parent
CONTENT_DIR = SITE_DIR / "content"
LAYOUT_PATH = SITE_DIR / "layout.html"
LANDING_LAYOUT_PATH = SITE_DIR / "landing_layout.html"

# (filename, nav label, h1, dek, meta description, theme_slug)
PAGES = [
    (
        "overview.html",
        "Overview",
        "Project overview",
        "The research question, why it matters, scope, and how this site is organized.",
        "Project overview: whole-brain activity and behavior forecasting in C. elegans.",
        "overview",
    ),
    (
        "data.html",
        "Data",
        "Data sources, provenance, and license status",
        "80 real recordings, CC BY 3.0 US licensed, verified against the Zenodo record itself.",
        "Verified data sources, provenance, and license status for this project.",
        "data",
    ),
    (
        "methods.html",
        "Methods",
        "Methods",
        "Time-aware segmentation, forecast targets, population-level neural features, and the "
        "purge-gapped blocked / forward-chaining evaluation protocol.",
        "Methods: segmentation, targets, features, models, leakage-safe evaluation.",
        "methods",
    ),
    (
        "results.html",
        "Results",
        "Results",
        "Verified, exact results for velocity and reversal-onset forecasting, both evaluation schemes.",
        "Results: velocity and reversal-onset forecasting, blocked and forward-chaining CV.",
        "results",
    ),
    (
        "interpretation.html",
        "Interpretation",
        "Interpretation",
        "What these results do and do not support, and how they relate to prior published work.",
        "Interpretation: what the results support, caveats, and relation to prior work.",
        "interpretation",
    ),
    (
        "status.html",
        "Status",
        "Current status",
        "What is built, tested, and run on real data; what remains; and the paused earlier direction.",
        "Current project status: completed behavior-forecasting analysis and the paused connectome direction.",
        "status",
    ),
    (
        "reproducibility.html",
        "Reproducibility",
        "Reproducibility",
        "Setup, running the tests, and reproducing the behavior-forecasting analysis.",
        "Setup and reproduction instructions for this project.",
        "repro",
    ),
]


def build_nav(active_filename: str) -> str:
    links = []
    for filename, nav_label, *_ in PAGES:
        if filename == active_filename:
            links.append(f'<a href="{filename}" class="active" aria-current="page">{nav_label}</a>')
        else:
            links.append(f'<a href="{filename}">{nav_label}</a>')
    return "\n    ".join(links)


def build_footer_nav(index: int) -> str:
    prev_link = ""
    next_link = ""
    if index > 0:
        pfile, plabel = PAGES[index - 1][0], PAGES[index - 1][1]
        prev_link = f'<a href="{pfile}"><span class="dir">&larr; Previous</span>{plabel}</a>'
    if index < len(PAGES) - 1:
        nfile, nlabel = PAGES[index + 1][0], PAGES[index + 1][1]
        next_link = f'<a href="{nfile}" style="text-align:right;"><span class="dir">Next &rarr;</span>{nlabel}</a>'
    if not prev_link and not next_link:
        return ""
    return f'<div class="page-nav-footer">{prev_link or "<span></span>"}{next_link}</div>'


def build_page_head(h1: str, dek: str) -> str:
    return (
        '<div class="page-head">\n'
        '  <div class="eyebrow">C. elegans &middot; neural activity &amp; behavior forecasting</div>\n'
        f"  <h1>{h1}</h1>\n"
        f'  <p class="dek">{dek}</p>\n'
        "</div>"
    )


def build_landing_page() -> None:
    layout = LANDING_LAYOUT_PATH.read_text()
    content = (CONTENT_DIR / "index.html").read_text()
    page = layout.replace("__CONTENT__", content)
    out_path = SITE_DIR / "index.html"
    out_path.write_text(page)
    print(f"Built {out_path}")


def build_regular_pages() -> None:
    layout = LAYOUT_PATH.read_text()
    for i, (filename, nav_label, h1, dek, description, theme_slug) in enumerate(PAGES):
        content = (CONTENT_DIR / filename).read_text()
        page = layout
        page = page.replace("__PAGE_TITLE__", nav_label)
        page = page.replace("__PAGE_DESCRIPTION__", description)
        page = page.replace("__BODY_CLASS__", f"theme-dark theme-{theme_slug}")
        page = page.replace("__NAV__", build_nav(filename))
        page = page.replace("__PAGE_HEAD__", build_page_head(h1, dek))
        page = page.replace("__CONTENT__", content)
        page = page.replace("__PAGE_FOOTER_NAV__", build_footer_nav(i))
        out_path = SITE_DIR / filename
        out_path.write_text(page)
        print(f"Built {out_path}")


def main() -> None:
    build_landing_page()
    build_regular_pages()


if __name__ == "__main__":
    main()
