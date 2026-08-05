"""Page 4 - About. What SmartDoc is, how it is built, and how a question flows.

The diagrams are built in ui/about_ui.py. Their numbers come from the live
/config response, so the page cannot describe settings the app is not using.
"""

import streamlit as st

from ui import about_ui, api, components

components.page_header(
    "ℹ️", "About", "How SmartDoc works, and what it is built from."
)

about_ui.overview()

settings, error = api.get_config()

if error:
    st.warning(
        "The pipeline settings could not be read, so the diagram below is "
        "hidden rather than shown with invented numbers."
    )
else:
    about_ui.architecture_diagram()
    about_ui.pipeline_flow(settings)

about_ui.tech_stack()

st.divider()
st.caption(
    "Interactive API documentation is available at "
    "[127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the backend is running."
)
