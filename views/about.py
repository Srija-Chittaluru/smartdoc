"""Page 4 - About. Introduces the product and what it can do.

Three sections, nothing else. The wording lives in ui/about_ui.py.
"""

from ui import about_ui, components

components.page_header(
    "ℹ️", "About", "What SmartDoc is and how it helps you find answers."
)

about_ui.intro()
about_ui.capabilities()
about_ui.benefits()
