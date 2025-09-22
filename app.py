# app.py  — LLM-routed + self-critique answering (no hand-coded vocab)
import os, json, re
from typing import List, Tuple
from dotenv import load_dotenv
from flask import Flask, render_template, request

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# your modules
from src.guardrails import detect_edge_case, needs_calculation, simple_calculator
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# LLM + RAG
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
app = Flask(__name__)

# ---------- keys ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if PINECONE_API_KEY: os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY: os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------- vector store ----------
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ---------- models ----------
primary_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.4, max_tokens=1100)
strong_llm  = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile", temperature=0.5, max_tokens=1400)

# ---------- tiny utils ----------
def jdump(d): return json.dumps(d, ensure_ascii=False)
def normalize(t:str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def format_sources(meta_list: List[dict]) -> str:
    if not meta_list: return "—"
    out=[]
    for m in meta_list[:2]:
        t = (m or {}).get("title") or (m or {}).get("source") or (m or {}).get("file") or (m or {}).get("path") or "Source"
        t = os.path.basename(t)
        p = (m or {}).get("page") or (m or {}).get("page_number") or (m or {}).get("loc")
        out.append(f"- {t}" + (f", p. {p}" if p is not None else ""))
    return "\n".join(out)

# ============================================================
# 1) ROUTER — ask the LLM to: (a) normalize/repair text, (b) classify intent
# ============================================================
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a robust input router for a medical assistant. "
     "Fix typos and missing punctuation WITHOUT changing the user's meaning. "
     "Return STRICT JSON with keys: "
     "`normalized`, `intent` in {greeting, medical, calculation, other}."),
    ("human", "{user_text}")
])

def llm_route(text: str) -> Tuple[str, str]:
    msgs = ROUTER_PROMPT.format_messages(user_text=text)
    out = primary_llm.invoke(msgs).content.strip()
    # be tolerant if model wrapped JSON in text
    m = re.search(r"\{.*\}", out, re.S)
    data = json.loads(m.group(0)) if m else json.loads(out)
    return data.get("normalized", text), data.get("intent", "medical")

# ============================================================
# 2) RAG answer (draft) with friendly style
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     system_prompt +
     "\n\nWrite like a warm, concise clinician. Be robust to typos. "
     "Use short paragraphs and bullet points when helpful. "
     "If the retrieved context is weak or irrelevant, still answer using general medical knowledge, "
     "but avoid speculation and include simple safety notes if appropriate."),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(primary_llm, RAG_PROMPT)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ============================================================
# 3) SELF-CHECK → possible revision with a stronger model
# ============================================================
CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict medical reviewer. Given the USER QUESTION, CONTEXT (may be empty), and DRAFT ANSWER: "
     "Return STRICT JSON with keys: `okay` (true/false) and `reason` (short). "
     "`okay` must be false if the draft is vague, repetitive, off-topic, or misses obvious info."),
    ("human", "USER QUESTION:\n{q}\n\nCONTEXT SNIPPETS:\n{ctx}\n\nDRAFT ANSWER:\n{a}")
])

REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Revise the answer to directly satisfy the user. Keep the tone friendly and precise. "
     "Mention safety/red-flags briefly when relevant. "
     "Prefer concrete steps and ranges (e.g., fever ranges)."),
    ("human", "USER QUESTION:\n{q}\n\nREVISION REQUEST:\n{why}\n\nPREVIOUS ANSWER:\n{a}")
])

def self_improve(question: str, ctx_chunks: List[str], draft: str) -> str:
    ctx = "\n\n".join(ctx_chunks[:4]) if ctx_chunks else "(no strong context)"
    msgs = CRITIC_PROMPT.format_messages(q=question, ctx=ctx, a=draft)
    critic = primary_llm.invoke(msgs).content.strip()
    try:
        m = re.search(r"\{.*\}", critic, re.S)
        data = json.loads(m.group(0)) if m else json.loads(critic)
    except Exception:
        data = {"okay": True, "reason": "parse-fail"}
    if bool(data.get("okay", True)):
        return draft
    msgs2 = REVISION_PROMPT.format_messages(q=question, why=data.get("reason","Improve clarity and completeness."), a=draft)
    revised = strong_llm.invoke(msgs2).content.strip()
    return revised or draft

# ============================================================
# Flask routes
# ============================================================
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat_route():
    user_raw = normalize(request.form["msg"])

    # LLM router: normalize + intent
    norm, intent = llm_route(user_raw)

    # Safety first (still server-side)
    is_edge, note = detect_edge_case(norm)
    if is_edge:
        return f"⚠️ {note} I can’t provide at-home advice for emergencies—please seek urgent care."

    # Simple calculator lane (dose conversions, etc.)
    if intent == "calculation" or needs_calculation(norm):
        try:
            return simple_calculator(norm)
        except Exception:
            pass  # fall through to normal answering

    # Greeting?
    if intent == "greeting":
        return "Hey! I’m your medical assistant. Ask me anything—e.g., “What’s a high fever?” or “Safe dose of ibuprofen for a 10-year-old?”"

    # RAG attempt (always try; it can choose to rely less on it)
    try:
        docs = retriever.invoke(norm)
    except Exception:
        docs = []
    ctx_chunks = [d.page_content for d in docs]
    sources = format_sources([d.metadata for d in docs]) if docs else "—"

    # First draft
    try:
        resp = rag_chain.invoke({"input": norm})
        draft = (resp.get("answer") or "").strip()
    except Exception:
        draft = ""

    # If the draft is empty/weak, ask the strong model directly (no RAG)
    if not draft or len(draft) < 40:
        draft = strong_llm.invoke([("human", norm)]).content.strip()

    # Self-critique → revise if needed
    final = self_improve(norm, ctx_chunks, draft)

    tail = "" if sources.strip()=="—" else f"\n\nSources:\n{sources}"
    follow = "\n\nIf you want, I can list red flags to watch for or safe home-care steps."
    return final + tail + follow

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
