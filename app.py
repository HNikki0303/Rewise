import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.summarizer import summarize, generate_title
from core.rag import build_rag_chain, load_rag_chain, ask_question
from core.session_manager import create_session, list_sessions

load_dotenv()

st.set_page_config(page_title="Rewise", page_icon="🎧", layout="wide")
st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#1D2128;
}

/* Main block */
.main .block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Headings */
h1,h2,h3,h4{
    color:#F4F2F2 !important;
}

/* Paragraphs */
p,label,span{
    color:#F4F2F2 !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:#215E61;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* Buttons */
.stButton>button{
    background:#FF9E20;
    color:white;
    border:none;
    border-radius:12px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover{
    background:#e38b12;
}

/* Tabs */
.stTabs [role="tab"]{
    background:#2a3039;
    color:white;
    border-radius:8px;
}

.stTabs [aria-selected="true"]{
    background:#215E61 !important;
}

/* Chat messages */

[data-testid="stChatMessage"]{
    background:#2b3139;
    border-radius:15px;
    padding:15px;
    margin-bottom:12px;
    border-left:5px solid #FF9E20;
}

/* Chat input */

[data-testid="stChatInput"]{
    background:#2b3139;
}

/* Text input */

.stTextInput input{
    border-radius:10px;
    border:2px solid #215E61;
}

/* Transcript */

textarea{
    border-radius:10px !important;
}

/* Info boxes */

[data-testid="stAlert"]{
    border-radius:12px;
}

/* Divider */

hr{
    border-color:#215E61;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Core pipeline — registers a session so it can be revisited later
# ---------------------------------------------------------------------------

def run_pipeline(source: str) -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    session_id = create_session(title)
    rag_chain = build_rag_chain(transcript, session_id)

    return {
        "session_id": session_id,
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------------------------
# Session state setup — persists across reruns within one browser session
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text) tuples
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pending_source" not in st.session_state:
    st.session_state.pending_source = ""
if "source_input" not in st.session_state:
    st.session_state.source_input = ""


# ---------------------------------------------------------------------------
# Callbacks — these run BEFORE the script reruns, so it's safe to mutate
# widget-backed session_state keys (like source_input) inside them.
# ---------------------------------------------------------------------------

def start_processing():
    typed_source = st.session_state.source_input.strip()
    if not typed_source:
        st.session_state._source_warning = True
        return
    st.session_state._source_warning = False
    st.session_state.pending_source = typed_source
    st.session_state.processing = True
    st.session_state.source_input = ""  # clear the box immediately


def load_past_session(session_id: str):
    rag_chain = load_rag_chain(session_id)
    st.session_state.result = None  # summary/items aren't re-fetched, only chat
    st.session_state.rag_chain = rag_chain
    st.session_state.chat_history = []
    st.session_state.active_session_id = session_id
    st.session_state.source_input = ""  # clear any half-typed URL


def end_conversation():
    st.session_state.result = None
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    st.session_state.active_session_id = None
    st.session_state.source_input = ""  # clean slate for the URL field too


# ---------------------------------------------------------------------------
# Sidebar — process new content, or pick any past one to resume
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎧 Rewise")
    st.caption("Turn any recording — meetings, lectures, podcasts, YouTube videos — into a searchable, chattable summary.")

    st.divider()
    st.subheader("New content")

    st.text_input("YouTube URL or local file path", key="source_input", disabled=st.session_state.processing)

    if st.session_state.get("_source_warning"):
        st.warning("Please enter a URL or file path first.")

    st.button(
        "Processing..." if st.session_state.processing else "Process content",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing,
        on_click=start_processing,
    )

    st.divider()
    st.subheader("Past sessions")

    sessions = list_sessions()

    if not sessions:
        st.caption("Nothing processed yet — add a URL or file above.")
    else:
        for record in sessions:
            label = f"{record['title']}  ·  {record['created_at']}"
            is_active = record["session_id"] == st.session_state.active_session_id
            st.button(
                ("✅ " if is_active else "") + label,
                key=f"session_{record['session_id']}",
                use_container_width=True,
                disabled=st.session_state.processing,
                on_click=load_past_session,
                args=(record["session_id"],),
            )

    st.divider()
    


# ---------------------------------------------------------------------------
# Run the pipeline (outside the sidebar block, so the spinner/results render
# in the main area) whenever a click has flagged processing=True
# ---------------------------------------------------------------------------

if st.session_state.processing and st.session_state.pending_source:
    with st.spinner("Processing — this can take a few minutes for longer recordings..."):
        try:
            result = run_pipeline(st.session_state.pending_source)
            st.session_state.result = result
            st.session_state.rag_chain = result["rag_chain"]
            st.session_state.chat_history = []
            st.session_state.active_session_id = result["session_id"]
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            st.session_state.processing = False
            st.session_state.pending_source = ""
    st.rerun()


# ---------------------------------------------------------------------------
# Main area — summary, action items, decisions, questions
# ---------------------------------------------------------------------------

result = st.session_state.result

if result:
    st.markdown(f"""
    <h1 style="
    color:#FF9E20;
    margin-bottom:0px;">
    🎧 {result['title']}
    </h1>

    <p style="
    color:#BFC6C7;
    margin-top:0px;">
    AI Meeting & Lecture Assistant
    </p>
    """, unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
        ["Summary", "Action Items", "Key Decisions", "Open Questions", "Full Transcript"]
    )

    with tab_summary:
        st.markdown(f"""
    <div style="
    background:#2b3139;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #FF9E20;
    color:white;
    font-size:17px;
    line-height:1.8;">
    {result["summary"]}
    </div>
    """, unsafe_allow_html=True)

    with tab_actions:
        st.write(result["action_items"])

    with tab_decisions:
        st.write(result["key_decisions"])

    with tab_questions:
        st.write(result["open_questions"])

    with tab_transcript:
        st.text_area("Transcript", result["transcript"], height=400)

    st.divider()

elif st.session_state.rag_chain is not None:
    st.info("Resumed a past session — summary/action items aren't reloaded, but you can chat below.")


# ---------------------------------------------------------------------------
# Chat section — only shown once a rag_chain exists
# ---------------------------------------------------------------------------

if st.session_state.rag_chain is not None:
    st.subheader("💬 Chat about this content")

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask something about this recording...")

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(st.session_state.rag_chain, question)
                st.write(answer)

        st.session_state.chat_history.append(("assistant", answer))

else:
    st.markdown("""
# 🎧 Rewise

### Your AI Meeting & Lecture Assistant

Upload a recording or paste a YouTube link to:

✅ Generate concise summaries

✅ Extract action items

✅ Identify key decisions

✅ List open questions

✅ Chat with your recording using AI

---

Start by entering a YouTube URL in the sidebar.
""")