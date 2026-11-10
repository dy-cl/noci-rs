# docs/conf.py
"""Sphinx configuration for the noci-rs documentation."""

project = "noci-rs"
author = "Dylan Counsell"
copyright = "2026, Dylan Counsell"
release = "0.1.1"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

# MyST: dollar maths, colon-fenced directives and heading anchors for cross-page links.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "attrs_inline",
]
myst_heading_anchors = 3

# One numbered bibliography for the whole site, on the references page.
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

exclude_patterns = ["_build"]
source_suffix = {".md": "markdown"}

html_theme = "furo"
html_title = "noci-rs"
html_logo = "../assets/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/dy-cl/noci-rs/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#23761a",
        "color-brand-content": "#23761a",
        "color-admonition-title--note": "#23761a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6fcf5f",
        "color-brand-content": "#6fcf5f",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/dy-cl/noci-rs",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07'
                ".55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
                "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51"
                "-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21"
                " 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92"
                ".08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01"
                ' 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
