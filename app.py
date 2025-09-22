# app.py
import os, re, string
from typing import List
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Silence HF tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Project imports ---
from src.guardrails import (
    detect_edge_case,
    needs_calculation,
    simple_calculator,
)
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# --- LangChain / Groq / Pinecone ---
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Robustness helpers ---
from rapidfuzz import fuzz, process as rf_process

# --------------------------------------------------------
load_dotenv()
app = Flask(__name__)

# Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Embeddings + Vector store
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# LLM
LLM_MODEL = "llama-3.1-8b-instant"  # fast & cheap; swap to 70B if you want more power
chatModel = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name=LLM_MODEL, temperature=0.3, max_tokens=1024)

# RAG chain
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
qa_chain = create_stuff_documents_chain(chatModel, rag_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ----------------- Robust input handling -----------------
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
PUNCT_KEEP = set(["?", "%", "+", "/", ".", ",", "-", "’", "'"])

COMMON_GREETINGS = [
    "hi", "hello", "hey", "hiya", "yo", "sup", "wassup", "whats up", "what’s up",
    "good morning", "good afternoon", "good evening", "how are you", "how r u",
    "hola", "namaste", "heyy", "heyyy", "hii", "hiii"
]

def normalize_text(t: str) -> str:
    if not t: return ""
    t = t.strip().lower()
    t = EMOJI_RE.sub(" ", t)
    # remove weird whitespace and keep light punctuation
    keep = set(string.ascii_letters + string.digits + " " + "".join(PUNCT_KEEP))
    t = "".join(ch if ch in keep else " " for ch in t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_greeting(t: str) -> bool:
    t_norm = normalize_text(t)
    # exact or fuzzy
    choices = COMMON_GREETINGS + [g + "!" for g in COMMON_GREETINGS]
    match, score, _ = rf_process.extractOne(t_norm, choices, scorer=fuzz.token_sort_ratio)
    return bool(match) and score >= 70

def rewrite_query(user_q: str) -> str:
    """
    Let the LLM clean typos + turn anything into a concise medical question.
    """
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the user's message into a single, well-formed medical question.\n"
        "Fix spelling. If it is chit-chat, return a short friendly reply instead.\n"
        "User: {u}\n"
        "Rewritten:"
    )
    return chatModel.invoke(prompt.format_messages(u=user_q)).content.strip()

def weak_retrieval(docs: List[Document]) -> bool:
    # If nothing came back, or all chunks tiny/irrelevant -> consider weak
    if not docs: return True
    joined = " ".join(d.page_content for d in docs)
    return len(joined) < 200  # tiny context ⇒ treat as weak

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_q_raw = request.form["msg"]
    user_q_norm = normalize_text(user_q_raw)

    # 0) small talk (broad)
    if is_greeting(user_q_raw):
        return "Hey! I’m doing well 😊 How can I help with your health question today?"

    # 1) safety / emergency guardrail FIRST
    is_edge, note = detect_edge_case(user_q_norm)
    if is_edge:
        return f"⚠️ {note} If you’re in danger or having severe symptoms now, please seek urgent care or call your local emergency number."

    # 2) quick calc path
    if needs_calculation(user_q_norm):
        try:
            return simple_calculator(user_q_norm)
        except Exception:
            pass  # fall through to normal flow

    # 3) query rewriting to handle typos/missing punctuation
    user_q_clean = rewrite_query(user_q_raw)[:500]

    # 4) try retrieval
    docs = retriever.invoke(user_q_clean)
    if not docs:
        # Try again using the raw text if rewrite was odd
        docs = retriever.invoke(user_q_norm)

    # 5) Decide: RAG or direct LLM answer
    if weak_retrieval(docs):
        # Direct answer with a SAFE medical system prompt
        safe_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a careful, friendly medical assistant. "
             "Answer clearly in 5–8 sentences. "
             "Use layperson language, add brief tips, include red-flag guidance when relevant, "
             "and end with: 'This is general information, not a diagnosis.'"),
            ("human", "{q}")
        ])
        msg = safe_prompt.format_messages(q=user_q_clean)
        ans = chatModel.invoke(msg).content.strip()
        return ans

    # 6) RAG answer with short refs
    response = rag_chain.invoke({"input": user_q_clean})
    answer = (response.get("answer") or "").strip()
    if not answer:
        # fallback to direct if RAG produced nothing
        return chatModel.invoke(
            ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{q}")]).format_messages(q=user_q_clean)
        ).content.strip()

    # build short refs
    def _meta_title(m):
        if not isinstance(m, dict): return "Source"
        title = m.get("title") or m.get("source") or m.get("file") or m.get("path") or "Source"
        return str(title).split("/")[-1]
    refs = []
    for d in (response.get("context", []) or docs)[:2]:
        m = getattr(d, "metadata", {}) or {}
        refs.append(_meta_title(m))
    ref_text = "References: " + " • ".join(refs) if refs else ""

    # friendly finishing
    return f"{answer}\n\n{ref_text}\n\nThis is general information, not a diagnosis."
# ----------------- Main -----------------
if __name__ == "__main__":
    # Run via gunicorn in Docker; dev server here for local runs
    app.run(host="0.0.0.0", port=8080, debug=True)
