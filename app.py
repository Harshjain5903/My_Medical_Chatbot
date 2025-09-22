# app.py
import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Silence HF tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Project imports you already have ---
from src.guardrails import (
    detect_edge_case,
    needs_calculation,
    simple_calculator,
    confidence_from_context,
    agree_two_sources,
)
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# LangChain / Groq / Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

# --- Keys from environment (in Docker these come from GitHub Secrets) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Embeddings + Vector store ---
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- LLM (Groq) ---
chatModel = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ----------------- Helpers -----------------
SMALLTALK_PAT = re.compile(
    r"^(hi|hello|hey|yo|good\s*(morning|afternoon|evening)|how\s*are\s*you|thanks|thank\s*you|bye|goodbye)[.!?]*$",
    re.I,
)

def detect_intent(text: str) -> str:
    t = text.strip()
    if SMALLTALK_PAT.match(t):
        return "smalltalk"
    if re.search(r"(chest pain|shortness of breath|severe|bleeding|unconscious|overdose|poison|suicid|kill myself)", t, re.I):
        return "emergency"
    return "medical"

def _meta_title(m):
    if not isinstance(m, dict):
        return "Source"
    title = m.get("title")
    if title:
        return str(title)
    src = m.get("source") or m.get("file") or m.get("path")
    if src:
        return os.path.basename(str(src))
    return "Source"

def format_sources(meta_list):
    if not meta_list:
        return "—"
    lines = []
    for m in meta_list[:2]:
        title = _meta_title(m)
        page = m.get("page") or m.get("page_number") or m.get("loc")
        if page is not None:
            lines.append(f"- {title}, p. {page}")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_q = request.form["msg"].strip()
    print("[User]", user_q)

    # 0) Small talk path — no RAG
    intent = detect_intent(user_q)
    if intent == "smalltalk":
        return "Hi! I’m your medical assistant. Ask a health question (e.g., “What helps with migraine?”) and I’ll cite the book."

    # 1) Safety / emergency guardrail FIRST
    is_edge, note = detect_edge_case(user_q)
    if is_edge:
        return (
            "⚠️ " + note +
            " Based on medical safety guidance, please seek urgent care or contact a licensed clinician. "
            "I cannot provide at-home advice for this."
        )

    # 2) Simple calculation path
    if needs_calculation(user_q):
        return simple_calculator(user_q)

    # 3) Retrieve context
    docs = retriever.invoke(user_q)
    ctx_chunks = [d.page_content for d in docs]
    sources_text = format_sources([d.metadata for d in docs]) if docs else "—"

    # 4) Ask the model via RAG
    response = rag_chain.invoke({"input": user_q})
    answer = response.get("answer", "").strip()

    # 5) Confidence & agreement checks
    conf = confidence_from_context(ctx_chunks)  # float in [0,1]
    try:
        agree = agree_two_sources(ctx_chunks, answer)
    except Exception:
        agree = True

    if (conf is not None and conf < 0.25) or not agree:
        return (
            "I’m not fully confident based on the available sources. Please consult a clinician.\n\n"
            "Closest references:\n" + sources_text
        )

    conf_pct = int((conf or 0.6) * 100)
    final = f"{answer}\n\n—\nConfidence: ~{conf_pct}%\nReferences:\n{sources_text}"
    print("[Answer]", final[:250], "...")
    return final

# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
