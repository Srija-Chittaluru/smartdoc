"""Page 3 - Analytics. A dashboard over the index and the questions asked.

Statistics are computed by the backend (backend/metrics.py); this page only
lays them out. Each piece is built in ui/analytics_ui.py.
"""

import streamlit as st

from ui import analytics_ui, api, components

components.page_header(
    "📊",
    "Analytics",
    "What is indexed, what people ask, and how well retrieval is doing.",
)

data, error = api.get_analytics()

if error:
    st.error(error)
    st.stop()

summary = data["summary"]

analytics_ui.kpi_tiles(summary, data["totals"])

if summary["questions_asked"] == 0:
    st.info(
        "No questions have been asked yet. Ask something on the "
        "**Ask Documents** page and the charts below will fill in."
    )
else:
    # Worth saying, because it explains an average that otherwise looks wrong:
    # the embedding model loads on the first question after a restart.
    st.caption(
        "Response time covers retrieval and answer generation together. The "
        "first question after a restart also loads the embedding model, which "
        "pulls the average up."
    )

analytics_ui.chunks_per_document(data["chunks_per_document"])

left, right = st.columns(2, gap="medium")
with left:
    analytics_ui.questions_over_time(summary["questions_per_day"])
with right:
    analytics_ui.similarity_distribution(summary["similarity_distribution"])

analytics_ui.document_usage(summary["document_usage"])
analytics_ui.recent_queries(summary["recent"])

st.divider()

refused = summary["unanswered"]
st.caption(
    f"{summary['answered']} of {summary['questions_asked']} questions were answered "
    f"from the documents; {refused} {'was' if refused == 1 else 'were'} refused as "
    "out of scope."
)
