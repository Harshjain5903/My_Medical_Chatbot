from typing import List, Tuple
import re, math
import textdistance

HIGH_RISK_PATTERNS = [
    r"drink\s+bleach", r"kerosene", r"raw\s+mercury",
    r"chest pain.*pregnan", r"shortness of breath", r"blood.*vomit",
]
ALWAYS_ESCALATE = [
    r"chest pain", r"trouble breathing", r"one\s+side weakness", r"slurred speech",
]

def detect_edge_case(q: str) -> Tuple[bool, str]:
    lower = q.lower()
    for pat in HIGH_RISK_PATTERNS:
        if re.search(pat, lower):
            return True, "Dangerous request detected."
    for pat in ALWAYS_ESCALATE:
        if re.search(pat, lower):
            return True, "Potential emergency—recommend urgent care."
    return False, ""

def needs_calculation(q: str) -> bool:
    return any(k in q.lower() for k in ["bmi", "dose", "dosage", "mg/kg", "calculate"])

def simple_calculator(q: str) -> str:
    m = re.search(r"bmi.*?(\d+(?:\.\d+)?)\s*kg.*?(\d+(?:\.\d+)?)\s*m", q.lower())
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        bmi = w/(h*h)
        cat = ("underweight" if bmi<18.5 else
               "normal" if bmi<25 else
               "overweight" if bmi<30 else "obese")
        return f"Your BMI is {bmi:.1f} ({cat}). General advice only—confirm with a clinician."
    return "I can’t compute that number. Please give weight in kg and height in meters for BMI."

def confidence_from_context(chunks: List[str]) -> float:
    if not chunks: return 0.0
    uniq = set(chunks)
    length = sum(len(c) for c in uniq)
    return max(0.0, min(1.0, math.log10(1+length/1000.0)))

def agree_two_sources(chunks: List[str], draft_answer: str) -> bool:
    sims = []
    for c in chunks:
        sims.append(textdistance.cosine.normalized_similarity(
            draft_answer.lower()[:1000], c.lower()[:1000]))
    sims.sort(reverse=True)
    return len([s for s in sims[:3] if s > 0.75]) >= 2
