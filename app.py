import streamlit as st
import pdfplumber
import re
import json
import os
from groq import Groq
import pytesseract
from PIL import Image
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import datetime

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(page_title="MedFamily", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════
# GLOBAL CSS  — dark theme, zero HTML comments
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1321 50%,#111827 100%);min-height:100vh;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#111827;}
::-webkit-scrollbar-thumb{background:#3b82f6;border-radius:3px;}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea{
    background:rgba(15,23,42,0.85)!important;
    border:1px solid rgba(59,130,246,0.25)!important;
    border-radius:10px!important;color:#e2e8f0!important;
    font-family:'Inter',sans-serif!important;font-size:0.88rem!important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
    border-color:rgba(59,130,246,0.6)!important;
    box-shadow:0 0 0 3px rgba(59,130,246,0.1)!important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label{
    color:#64748b!important;font-size:0.74rem!important;
    font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.6px!important;
}
[data-baseweb="select"]>div{
    background:rgba(15,23,42,0.85)!important;
    border:1px solid rgba(59,130,246,0.25)!important;
    border-radius:10px!important;color:#e2e8f0!important;
}
[data-testid="stFileUploader"]{background:transparent!important;border:none!important;}
[data-testid="stFileUploader"]>div{
    background:rgba(59,130,246,0.05)!important;
    border:2px dashed rgba(59,130,246,0.3)!important;
    border-radius:16px!important;padding:1.8rem!important;transition:all 0.3s!important;
}
[data-testid="stFileUploader"]>div:hover{
    border-color:rgba(59,130,246,0.65)!important;
    background:rgba(59,130,246,0.09)!important;
}
[data-testid="stButton"]>button{
    background:linear-gradient(135deg,#3b82f6,#6366f1)!important;
    color:white!important;border:none!important;border-radius:12px!important;
    font-weight:700!important;font-size:0.88rem!important;padding:0.6rem 1.4rem!important;
    transition:all 0.2s!important;box-shadow:0 4px 15px rgba(59,130,246,0.3)!important;
    font-family:'Inter',sans-serif!important;
}
[data-testid="stButton"]>button:hover{
    transform:translateY(-1px)!important;
    box-shadow:0 8px 25px rgba(59,130,246,0.45)!important;
}
.stSpinner>div{border-top-color:#3b82f6!important;}
div[data-testid="stForm"]{background:transparent!important;border:none!important;padding:0!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
for k, v in {
    "page": "hub",
    "members": [],
    "active_idx": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page, **kw):
    st.session_state.page = page
    for k, v in kw.items():
        st.session_state[k] = v
    st.rerun()

# ══════════════════════════════════════════════════
# BACKEND — API / ML
# ══════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.markdown(
        '<div style="display:flex;justify-content:center;align-items:center;min-height:80vh;">'
        '<div style="background:#1e1e2e;border:1px solid #ef4444;border-radius:18px;'
        'padding:2.5rem 3rem;text-align:center;">'
        '<div style="font-size:3rem;margin-bottom:1rem;">&#x1F510;</div>'
        '<h3 style="color:#ef4444;margin:0 0 0.75rem;">Groq API Key Missing</h3>'
        '<p style="color:#94a3b8;margin:0;">Set: <code style="background:#0d1321;padding:4px 10px;'
        'border-radius:6px;color:#60a5fa;">setx GROQ_API_KEY "your_key"</code></p>'
        '</div></div>', unsafe_allow_html=True)
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_data
def load_benchmarks():
    with open("benchmarks.json", "r", encoding="utf-8") as f:
        return json.load(f)

benchmarks = load_benchmarks()
pytesseract.pytesseract.tesseract_cmd = r".\tesseract.exe"

def extract_text_from_pdf(f):
    text = ""
    try:
        with pdfplumber.open(f) as pdf:
            for pg in pdf.pages:
                c = pg.extract_text()
                if c: text += c + "\n"
    except: pass
    if not text.strip():
        f.seek(0)
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for pg in doc:
            pix = pg.get_pixmap(matrix=fitz.Matrix(3, 3))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img, config="--psm 6")
    return text

def extract_values(text):
    out = {}
    for t in benchmarks:
        m = re.search(rf"{t}\s+([\d]+(?:\.[\d]+)?)", text, re.IGNORECASE)
        if m:
            try: out[t] = float(m.group(1))
            except: pass
    return out

def compare_values(res):
    out = {}
    for t, v in res.items():
        r = benchmarks[t]
        s = "Low" if v < r["min"] else "High" if v > r["max"] else "Normal"
        out[t] = {"value": v, "unit": r["unit"], "status": s}
    return out

@st.cache_resource
def setup_rag():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("medical_knowledge.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")
    embs = model.encode(chunks)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(np.array(embs))
    return model, idx, chunks

emb_model, f_idx, k_chunks = setup_rag()

def rag_context(q):
    e = emb_model.encode([q])
    _, idxs = f_idx.search(np.array(e), 2)
    return "\n".join([k_chunks[i] for i in idxs[0]])

def ai_explain(data, m):
    summary = "\n".join(f"{t}: {i['value']} {i['unit']} ({i['status']})" for t, i in data.items())
    ctx = rag_context(summary)
    al = ", ".join(m.get("allergies_list", [])) or "None"
    co = ", ".join(m.get("conditions_list", [])) or "None"
    prompt = (f"You are a medical lab report explanation assistant.\n"
              f"Patient: {m.get('name','Patient')}, Age {m.get('age','?')}, "
              f"Blood Group {m.get('blood_group','?')}\n"
              f"Allergies: {al} | Conditions: {co}\n"
              f"Medical Knowledge:\n{ctx}\n\nLab Results:\n{summary}\n\n"
              f"Explain each result simply. Do NOT diagnose. Do NOT prescribe. Advise consulting a doctor.")
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2)
    return r.choices[0].message.content.strip()

# ══════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════
COLORS = ["#0d9488","#3b82f6","#8b5cf6","#f59e0b","#ef4444","#10b981","#6366f1","#ec4899"]

def avatar_color(i): return COLORS[i % len(COLORS)]

def initials(name):
    p = name.strip().split()
    return (p[0][0] + p[-1][0]).upper() if len(p) >= 2 else name[:2].upper()

def topbar(back_label=None, back_page=None, back_kw=None):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:1.1rem 0 0.9rem 2rem;">'
            '<div style="width:38px;height:38px;border-radius:10px;'
            'background:linear-gradient(135deg,#0d9488,#3b82f6);'
            'display:flex;align-items:center;justify-content:center;font-size:1.05rem;'
            'box-shadow:0 4px 12px rgba(13,148,136,0.4);">&#x1F4CA;</div>'
            '<div>'
            '<div style="color:#f1f5f9;font-weight:800;font-size:1.05rem;letter-spacing:-0.3px;">MedFamily</div>'
            '<div style="color:#475569;font-size:0.62rem;letter-spacing:1px;text-transform:uppercase;">Lab Report Intelligence</div>'
            '</div></div>', unsafe_allow_html=True)
    if back_label:
        with c2:
            st.markdown('<div style="padding:0.9rem 2rem 0 0;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
            if st.button(f"← {back_label}", key=f"back_{back_page}"):
                go(back_page, **(back_kw or {}))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 2rem 0;"></div>', unsafe_allow_html=True)

def section_header(icon, title, color="#3b82f6"):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:1.6rem 0 0.9rem;">'
        f'<div style="width:28px;height:28px;border-radius:8px;background:{color};'
        f'display:flex;align-items:center;justify-content:center;font-size:0.85rem;">{icon}</div>'
        f'<span style="color:#e2e8f0;font-weight:700;font-size:0.92rem;letter-spacing:-0.2px;">{title}</span>'
        f'</div>', unsafe_allow_html=True)

def card_wrap(border_color="rgba(255,255,255,0.08)", bg="rgba(15,23,42,0.85)"):
    return (f'<div style="background:{bg};border:1px solid {border_color};'
            f'border-radius:16px;padding:1.5rem 1.8rem;margin-bottom:1rem;">')

# ══════════════════════════════════════════════════
# PAGE: HUB
# ══════════════════════════════════════════════════
def page_hub():
    members = st.session_state.members

    # Topbar with Add Member on right
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:1.1rem 0 0.9rem 2rem;">'
            '<div style="width:38px;height:38px;border-radius:10px;'
            'background:linear-gradient(135deg,#0d9488,#3b82f6);'
            'display:flex;align-items:center;justify-content:center;font-size:1.05rem;'
            'box-shadow:0 4px 12px rgba(13,148,136,0.4);">&#x1F4CA;</div>'
            '<div>'
            '<div style="color:#f1f5f9;font-weight:800;font-size:1.05rem;">MedFamily</div>'
            '<div style="color:#475569;font-size:0.62rem;letter-spacing:1px;text-transform:uppercase;">Lab Report Intelligence</div>'
            '</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="padding:0.85rem 2rem 0 0;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
        if st.button("＋ Add Member", key="hub_add"):
            go("add_member")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 2rem;"></div>', unsafe_allow_html=True)

    # Hero text
    total_reports = sum(len(m.get("reports", [])) for m in members)
    total_doctors = sum(len(m.get("doctors", [])) for m in members)

    st.markdown(
        '<div style="max-width:1100px;margin:2.5rem auto 0;padding:0 2rem;">'
        '<h1 style="color:#f8fafc;font-size:2.1rem;font-weight:800;margin:0 0 0.5rem;letter-spacing:-1px;">'
        'Your Family\'s <span style="background:linear-gradient(90deg,#0d9488,#3b82f6);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Health Hub</span>'
        '</h1>'
        '<p style="color:#64748b;font-size:0.88rem;margin:0 0 1.8rem;max-width:540px;line-height:1.65;">'
        'Manage health profiles, track doctors, and upload lab reports — all in one place. '
        'AI-powered insights to simplify complex medical data.</p>'
        '</div>', unsafe_allow_html=True)

    # Stats
    st.markdown('<div style="max-width:1100px;margin:0 auto;padding:0 2rem;">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for col, label, val, color in [
        (s1, "Family Members", len(members), "#3b82f6"),
        (s2, "Total Reports", total_reports, "#8b5cf6"),
        (s3, "Doctors", total_doctors, "#0d9488"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:14px;padding:1.4rem 1.8rem;text-align:center;margin-bottom:1.5rem;">'
                f'<div style="color:{color};font-size:2rem;font-weight:800;">{val}</div>'
                f'<div style="color:#475569;font-size:0.76rem;margin-top:3px;">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # Family Profiles header
    st.markdown(
        '<div style="color:#e2e8f0;font-size:0.95rem;font-weight:700;'
        'margin-bottom:0.9rem;letter-spacing:-0.2px;">Family Profiles</div>',
        unsafe_allow_html=True)

    if not members:
        st.markdown(
            '<div style="background:rgba(15,23,42,0.6);border:1px dashed rgba(59,130,246,0.2);'
            'border-radius:16px;padding:3rem;text-align:center;margin-bottom:2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;">&#x1F46A;</div>'
            '<div style="color:#475569;font-size:0.88rem;">No family members yet. '
            'Click <b style="color:#60a5fa;">＋ Add Member</b> to get started.</div>'
            '</div>', unsafe_allow_html=True)
    else:
        for row in range(0, len(members), 2):
            cols = st.columns(2)
            for ci, mi in enumerate([row, row + 1]):
                if mi >= len(members): break
                m = members[mi]
                with cols[ci]:
                    ini = initials(m.get("name", "?"))
                    ac = avatar_color(mi)
                    blood = m.get("blood_group", "")
                    conds = m.get("conditions_list", [])
                    rel = m.get("relation", "")
                    age = m.get("age", "")
                    gen = m.get("gender", "")
                    meta = " · ".join(filter(None, [rel, f"{age} yrs" if age else "", gen]))
                    n_d = len(m.get("doctors", []))
                    n_r = len(m.get("reports", []))

                    blood_html = (f'<span style="background:rgba(239,68,68,0.12);color:#fca5a5;'
                                  f'border:1px solid rgba(239,68,68,0.2);border-radius:20px;'
                                  f'padding:2px 10px;font-size:0.7rem;font-weight:600;margin-right:5px;">'
                                  f'&#x1F525; {blood}</span>') if blood else ""
                    cond_html = "".join(
                        f'<span style="background:rgba(239,68,68,0.09);color:#fca5a5;'
                        f'border:1px solid rgba(239,68,68,0.15);border-radius:20px;'
                        f'padding:2px 9px;font-size:0.68rem;font-weight:600;margin-right:4px;">'
                        f'&#x2665; {c}</span>' for c in conds[:2])

                    st.markdown(
                        f'<div style="background:rgba(15,23,42,0.85);'
                        f'border:1px solid rgba(255,255,255,0.07);border-radius:14px;'
                        f'padding:1.2rem 1.4rem;margin-bottom:0.6rem;">'
                        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;">'
                        f'<div style="display:flex;align-items:center;gap:13px;">'
                        f'<div style="width:46px;height:46px;border-radius:11px;background:{ac};'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'color:white;font-size:0.95rem;font-weight:800;flex-shrink:0;">{ini}</div>'
                        f'<div>'
                        f'<div style="color:#f1f5f9;font-weight:700;font-size:0.92rem;">{m.get("name","")}</div>'
                        f'<div style="color:#475569;font-size:0.73rem;margin-top:2px;">{meta}</div>'
                        f'<div style="margin-top:7px;">{blood_html}{cond_html}</div>'
                        f'</div></div>'
                        f'<span style="color:#334155;font-size:1.1rem;padding-top:4px;">&#x203A;</span>'
                        f'</div>'
                        f'<div style="display:flex;gap:1rem;margin-top:0.75rem;padding-top:0.65rem;'
                        f'border-top:1px solid rgba(255,255,255,0.05);">'
                        f'<span style="color:#475569;font-size:0.72rem;">{n_d} doctor{"s" if n_d!=1 else ""}</span>'
                        f'<span style="color:#334155;">&middot;</span>'
                        f'<span style="color:#475569;font-size:0.72rem;">{n_r} report{"s" if n_r!=1 else ""}</span>'
                        f'</div></div>', unsafe_allow_html=True)

                    # Invisible styled button overlaid
                    if st.button(f"View {m['name']}", key=f"view_m_{mi}", use_container_width=True):
                        go("profile_detail", active_idx=mi)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE: ADD MEMBER  (full page — no modal)
# ══════════════════════════════════════════════════
def page_add_member():
    topbar("Family Hub", "hub")

    st.markdown(
        '<div style="max-width:600px;margin:2rem auto 0;padding:0 2rem;">'
        '<div style="background:rgba(15,23,42,0.9);border:1px solid rgba(59,130,246,0.2);'
        'border-radius:20px;padding:2.2rem 2.5rem;box-shadow:0 20px 60px rgba(0,0,0,0.5);">',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#f1f5f9;font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;">Add Family Member</div>'
        '<div style="color:#475569;font-size:0.78rem;margin-bottom:1.5rem;">Fill in the details below to create a health profile.</div>',
        unsafe_allow_html=True)

    with st.form("add_member_form", clear_on_submit=True):
        nm = st.text_input("Full Name", placeholder="e.g. Priya Sharma")

        fc1, fc2 = st.columns(2)
        with fc1:
            ag = st.text_input("Age", placeholder="30")
        with fc2:
            gen = st.selectbox("Gender", ["", "Male", "Female", "Other"])

        fc3, fc4 = st.columns(2)
        with fc3:
            bg = st.selectbox("Blood Group", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
        with fc4:
            rel = st.text_input("Relation", placeholder="e.g. Mother, Self, Son")

        hc = st.text_input("Health Conditions (comma separated)", placeholder="e.g. Diabetes, Asthma")
        al = st.text_input("Allergies (comma separated)", placeholder="e.g. Peanuts, Dust")

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            submitted = st.form_submit_button("Add Member", use_container_width=True)
        with b2:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    if submitted:
        if nm.strip():
            conds = [x.strip() for x in hc.split(",") if x.strip()]
            algs = [x.strip() for x in al.split(",") if x.strip()]
            st.session_state.members.append({
                "name": nm.strip(), "age": ag.strip(), "gender": gen,
                "blood_group": bg, "relation": rel.strip(),
                "conditions_list": conds, "allergies_list": algs,
                "doctors": [], "reports": [],
            })
            go("hub")
        else:
            st.error("Please enter a name.")

    if cancelled:
        go("hub")


# ══════════════════════════════════════════════════
# PAGE: ADD DOCTOR  (full page)
# ══════════════════════════════════════════════════
def page_add_doctor():
    idx = st.session_state.active_idx
    if idx is None or idx >= len(st.session_state.members):
        go("hub")
        return
    m = st.session_state.members[idx]

    topbar(f"{m['name']}", "profile_detail", {"active_idx": idx})

    st.markdown(
        '<div style="max-width:580px;margin:2rem auto 0;padding:0 2rem;">'
        '<div style="background:rgba(15,23,42,0.9);border:1px solid rgba(16,185,129,0.2);'
        'border-radius:20px;padding:2.2rem 2.5rem;box-shadow:0 20px 60px rgba(0,0,0,0.5);">',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#f1f5f9;font-size:1.15rem;font-weight:700;margin-bottom:0.3rem;">Add Doctor / Specialist</div>'
        '<div style="color:#475569;font-size:0.78rem;margin-bottom:1.5rem;">Add a doctor to this profile for easy reference.</div>',
        unsafe_allow_html=True)

    with st.form("add_doctor_form", clear_on_submit=True):
        doc_name = st.text_input("Doctor's Name", placeholder="Dr. Priya Mehta")
        spec_opts = ["General Physician", "Cardiologist", "Endocrinologist", "Nephrologist",
                     "Neurologist", "Orthopedic", "Gastroenterologist", "Pulmonologist",
                     "Oncologist", "Dermatologist", "Gynaecologist", "Diabetologist",
                     "Rheumatologist", "Psychiatrist", "Other"]
        doc_spec = st.selectbox("Specialization", spec_opts)
        dc1, dc2 = st.columns(2)
        with dc1:
            doc_phone = st.text_input("Phone", placeholder="+91-98765-43210")
        with dc2:
            doc_hosp = st.text_input("Hospital / Clinic", placeholder="Apollo Hospitals")

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        db1, db2 = st.columns(2)
        with db1:
            doc_submit = st.form_submit_button("Add Doctor", use_container_width=True)
        with db2:
            doc_cancel = st.form_submit_button("Cancel", use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    if doc_submit:
        if doc_name.strip():
            st.session_state.members[idx]["doctors"].append({
                "name": doc_name.strip(), "specialization": doc_spec,
                "phone": doc_phone.strip(), "hospital": doc_hosp.strip(),
            })
            go("profile_detail", active_idx=idx)
        else:
            st.error("Please enter a doctor name.")

    if doc_cancel:
        go("profile_detail", active_idx=idx)


# ══════════════════════════════════════════════════
# PAGE: PROFILE DETAIL
# ══════════════════════════════════════════════════
def page_profile_detail():
    idx = st.session_state.active_idx
    if idx is None or idx >= len(st.session_state.members):
        go("hub"); return
    m = st.session_state.members[idx]

    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:1.1rem 0 0.9rem 2rem;">'
            '<div style="width:38px;height:38px;border-radius:10px;'
            'background:linear-gradient(135deg,#0d9488,#3b82f6);'
            'display:flex;align-items:center;justify-content:center;font-size:1.05rem;">&#x1F4CA;</div>'
            '<div>'
            f'<div style="color:#f1f5f9;font-weight:800;font-size:1.05rem;">{m.get("name","")}</div>'
            '<div style="color:#475569;font-size:0.62rem;letter-spacing:1px;text-transform:uppercase;">Health Profile</div>'
            '</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="padding:0.85rem 2rem 0 0;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
        if st.button("← Family Hub", key="back_hub_det"):
            go("hub")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 2rem;"></div>', unsafe_allow_html=True)

    st.markdown('<div style="max-width:860px;margin:0 auto;padding:1.8rem 2rem 0;">', unsafe_allow_html=True)

    # Profile card
    ini = initials(m.get("name", "?"))
    ac = avatar_color(idx)
    blood = m.get("blood_group", "")
    rel = m.get("relation", "")
    age = m.get("age", "")
    gen = m.get("gender", "")
    meta = " · ".join(filter(None, [rel, f"{age} yrs" if age else "", gen]))
    blood_html = (f'<span style="background:rgba(239,68,68,0.12);color:#fca5a5;'
                  f'border:1px solid rgba(239,68,68,0.25);border-radius:20px;'
                  f'padding:3px 12px;font-size:0.74rem;font-weight:700;display:inline-block;margin-top:8px;">'
                  f'&#x1F525; {blood}</span>') if blood else ""

    st.markdown(
        f'<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:16px;padding:1.6rem 1.8rem;margin-bottom:1rem;">'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'<div style="width:60px;height:60px;border-radius:13px;background:{ac};'
        f'display:flex;align-items:center;justify-content:center;'
        f'color:white;font-size:1.3rem;font-weight:800;flex-shrink:0;">{ini}</div>'
        f'<div>'
        f'<div style="color:#f1f5f9;font-weight:800;font-size:1.2rem;letter-spacing:-0.4px;">{m.get("name","")}</div>'
        f'<div style="color:#64748b;font-size:0.8rem;margin-top:2px;">{meta}</div>'
        f'{blood_html}'
        f'</div></div></div>', unsafe_allow_html=True)

    # Health Overview
    conds = m.get("conditions_list", [])
    algs = m.get("allergies_list", [])
    if conds or algs:
        st.markdown(
            '<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.07);'
            'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;">'
            '<div style="color:#e2e8f0;font-weight:700;font-size:0.9rem;margin-bottom:1rem;">Health Overview</div>',
            unsafe_allow_html=True)
        hc1, hc2 = st.columns(2)
        with hc1:
            cond_items = "".join(
                f'<div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">&#x2022; {c}</div>'
                for c in conds) or '<div style="color:#334155;font-size:0.78rem;">None recorded</div>'
            st.markdown(
                '<div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.12);'
                'border-radius:11px;padding:1rem 1.1rem;">'
                '<div style="color:#fca5a5;font-size:0.78rem;font-weight:600;margin-bottom:0.5rem;">&#x2665; Conditions</div>'
                f'{cond_items}</div>', unsafe_allow_html=True)
        with hc2:
            allergy_tags = "".join(
                f'<span style="background:rgba(245,158,11,0.12);color:#fbbf24;'
                f'border:1px solid rgba(245,158,11,0.2);border-radius:6px;'
                f'padding:2px 9px;font-size:0.72rem;display:inline-block;margin:2px;">{a}</span>'
                for a in algs) or '<div style="color:#334155;font-size:0.78rem;">None recorded</div>'
            st.markdown(
                '<div style="background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.12);'
                'border-radius:11px;padding:1rem 1.1rem;">'
                '<div style="color:#fbbf24;font-size:0.78rem;font-weight:600;margin-bottom:0.5rem;">&#x26A0; Allergies</div>'
                f'{allergy_tags}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Doctors & Specialists
    st.markdown(
        '<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.07);'
        'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;">',
        unsafe_allow_html=True)
    dc1, dc2 = st.columns([4, 1])
    with dc1:
        st.markdown('<div style="color:#e2e8f0;font-weight:700;font-size:0.9rem;">Doctors &amp; Specialists</div>', unsafe_allow_html=True)
    with dc2:
        if st.button("＋ Add Doctor", key="add_doc_btn"):
            go("add_doctor", active_idx=idx)

    docs = m.get("doctors", [])
    if not docs:
        st.markdown('<div style="color:#334155;font-size:0.8rem;margin-top:0.7rem;">No doctors added yet.</div>', unsafe_allow_html=True)
    for doc in docs:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:11px;padding:0.9rem 1.1rem;margin-top:0.6rem;'
            f'display:flex;align-items:center;gap:13px;">'
            f'<div style="width:34px;height:34px;border-radius:9px;background:rgba(59,130,246,0.12);'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">&#x1FA7A;</div>'
            f'<div style="flex:1;">'
            f'<div style="color:#e2e8f0;font-weight:600;font-size:0.85rem;">Dr. {doc.get("name","")}</div>'
            f'<div style="color:#3b82f6;font-size:0.73rem;margin-top:1px;">{doc.get("specialization","")}</div>'
            f'<div style="display:flex;gap:1.2rem;margin-top:4px;flex-wrap:wrap;">'
            + (f'<span style="color:#475569;font-size:0.72rem;">&#x1F4DE; {doc.get("phone","")}</span>' if doc.get("phone") else "")
            + (f'<span style="color:#475569;font-size:0.72rem;">&#x1F3E5; {doc.get("hospital","")}</span>' if doc.get("hospital") else "")
            + f'</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lab Reports
    st.markdown(
        '<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(255,255,255,0.07);'
        'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.5rem;">',
        unsafe_allow_html=True)
    rc1, rc2 = st.columns([4, 1])
    with rc1:
        st.markdown('<div style="color:#e2e8f0;font-weight:700;font-size:0.9rem;">Lab Reports</div>', unsafe_allow_html=True)
    with rc2:
        if st.button("&#x2B06; Upload PDF", key="upload_btn"):
            go("analyzer", active_idx=idx)

    reports = m.get("reports", [])
    if not reports:
        st.markdown('<div style="color:#334155;font-size:0.8rem;margin-top:0.7rem;">No reports yet. Click Upload PDF to analyze a report.</div>', unsafe_allow_html=True)
    for rep in reports:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:11px;padding:0.9rem 1.1rem;margin-top:0.6rem;'
            f'display:flex;align-items:center;gap:13px;">'
            f'<div style="width:34px;height:34px;border-radius:9px;background:rgba(239,68,68,0.1);'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">&#x1F4C4;</div>'
            f'<div>'
            f'<div style="color:#e2e8f0;font-weight:600;font-size:0.83rem;">{rep.get("name","Lab Report")}</div>'
            f'<div style="color:#475569;font-size:0.72rem;margin-top:2px;">'
            f'&#x1F4C5; {rep.get("date","")} &nbsp;&middot;&nbsp; {rep.get("size","")}</div>'
            f'</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE: ANALYZER
# ══════════════════════════════════════════════════
def page_analyzer():
    idx = st.session_state.active_idx
    m = st.session_state.members[idx] if (idx is not None and idx < len(st.session_state.members)) else {}

    # Topbar
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding:1.1rem 0 0.9rem 2rem;">'
            '<div style="width:38px;height:38px;border-radius:10px;'
            'background:linear-gradient(135deg,#0d9488,#3b82f6);'
            'display:flex;align-items:center;justify-content:center;font-size:1.05rem;">&#x1F4CA;</div>'
            '<div>'
            '<div style="color:#f1f5f9;font-weight:800;font-size:1.05rem;">MedFamily</div>'
            '<div style="color:#475569;font-size:0.62rem;letter-spacing:1px;text-transform:uppercase;">Lab Report Intelligence</div>'
            '</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="padding:0.85rem 2rem 0 0;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
        back_lbl = f"← {m.get('name','Back')}" if m else "← Hub"
        back_pg = "profile_detail" if m else "hub"
        if st.button(back_lbl, key="back_analyzer"):
            go(back_pg, active_idx=idx)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 2rem;"></div>', unsafe_allow_html=True)

    # Patient strip
    if m:
        blood = m.get("blood_group","") or "?"
        al_disp = ", ".join((m.get("allergies_list") or [])[:3]) or "None"
        st.markdown(
            '<div style="background:rgba(15,23,42,0.8);border-bottom:1px solid rgba(59,130,246,0.1);padding:0.75rem 3rem;">'
            '<div style="max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">'
            f'<span style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;">Patient</span>'
            f'<span style="color:#e2e8f0;font-size:0.8rem;font-weight:600;">{m.get("name","")}'
            + (f' &middot; <span style="color:#64748b;">{m.get("age","")} yrs</span>' if m.get("age") else "")
            + f'</span><span style="color:#334155;">|</span>'
            f'<span style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);'
            f'border-radius:6px;padding:2px 9px;font-size:0.7rem;font-weight:800;">{blood}</span>'
            f'<span style="color:#334155;">|</span>'
            f'<span style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;">Allergies</span>'
            f'<span style="color:#fbbf24;font-size:0.76rem;">{al_disp}</span>'
            '</div></div>', unsafe_allow_html=True)

    # Hero
    st.markdown(
        '<div style="background:linear-gradient(160deg,#0f172a 0%,#1a2540 100%);'
        'border-bottom:1px solid rgba(59,130,246,0.1);padding:1.8rem 3rem 1.5rem;">'
        '<div style="max-width:1100px;margin:0 auto;text-align:center;">'
        '<h1 style="font-size:clamp(1.6rem,3.5vw,2.6rem);font-weight:800;color:#f8fafc;'
        'margin:0 0 0.6rem;line-height:1.15;letter-spacing:-1px;">'
        'Analyze Lab Report<br>'
        '<span style="background:linear-gradient(90deg,#3b82f6,#8b5cf6,#06b6d4);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI-Powered Insights</span>'
        '</h1>'
        '<p style="color:#64748b;font-size:0.86rem;max-width:460px;margin:0 auto 1.4rem;line-height:1.7;">'
        'Upload a PDF lab report. Our AI will extract values, flag abnormal results, and explain everything simply.</p>'
        '<div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">'
        '<div style="text-align:center;"><div style="color:#3b82f6;font-size:1.3rem;font-weight:800;">50+</div>'
        '<div style="color:#475569;font-size:0.66rem;">Tests</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.06);"></div>'
        '<div style="text-align:center;"><div style="color:#8b5cf6;font-size:1.3rem;font-weight:800;">RAG</div>'
        '<div style="color:#475569;font-size:0.66rem;">Knowledge</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.06);"></div>'
        '<div style="text-align:center;"><div style="color:#06b6d4;font-size:1.3rem;font-weight:800;">OCR</div>'
        '<div style="color:#475569;font-size:0.66rem;">Scanned PDFs</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.06);"></div>'
        '<div style="text-align:center;"><div style="color:#0d9488;font-size:1.3rem;font-weight:800;">&#x2764;</div>'
        '<div style="color:#475569;font-size:0.66rem;">Personalized</div></div>'
        '</div></div></div>', unsafe_allow_html=True)

    # Upload
    st.markdown(
        '<div style="max-width:720px;margin:2rem auto 0;padding:0 2rem;">'
        '<div style="background:rgba(15,23,42,0.9);border:1px solid rgba(59,130,246,0.18);'
        'border-radius:18px;padding:1.6rem 2rem 1.4rem;">'
        '<div style="display:flex;align-items:center;gap:11px;margin-bottom:0.9rem;">'
        '<div style="width:30px;height:30px;border-radius:8px;background:rgba(59,130,246,0.12);'
        'border:1px solid rgba(59,130,246,0.3);display:flex;align-items:center;justify-content:center;font-size:0.9rem;">&#x1F4C4;</div>'
        '<div>'
        '<div style="color:#e2e8f0;font-weight:600;font-size:0.88rem;">Upload Lab Report PDF</div>'
        '<div style="color:#475569;font-size:0.68rem;">Digital or Scanned &middot; Max 20 MB</div>'
        '</div></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    st.markdown('</div></div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🔬 Extracting and analyzing..."):
            text = extract_text_from_pdf(uploaded_file)
            extracted = extract_values(text)
            compared = compare_values(extracted)

        # Save to member
        if m and idx is not None:
            rname = uploaded_file.name.replace(".pdf","").replace("_"," ").title()
            sz = uploaded_file.size // 1024
            sz_str = f"{sz/1024:.1f} MB" if sz > 1024 else f"{sz} KB"
            today = datetime.date.today().isoformat()
            if rname not in [r.get("name") for r in st.session_state.members[idx].get("reports",[])]:
                st.session_state.members[idx]["reports"].append({"name": rname, "date": today, "size": sz_str})

        if compared:
            st.markdown(
                '<div style="max-width:1100px;margin:2.2rem auto 1rem;padding:0 2rem;">'
                '<div style="display:flex;align-items:center;gap:10px;">'
                '<div style="width:3px;height:22px;background:linear-gradient(180deg,#3b82f6,#6366f1);border-radius:2px;"></div>'
                '<h2 style="color:#f1f5f9;font-size:1.15rem;font-weight:700;margin:0;">Detected Lab Results</h2>'
                '</div></div>', unsafe_allow_html=True)

            st.markdown(
                '<div style="max-width:1100px;margin:0 auto;padding:0 2rem;">'
                '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));gap:11px;">',
                unsafe_allow_html=True)

            cfg = {
                "High":   ("#ef4444","rgba(239,68,68,0.07)","rgba(239,68,68,0.15)","#f87171","&#x2191;","rgba(239,68,68,0.17)"),
                "Low":    ("#f59e0b","rgba(245,158,11,0.07)","rgba(245,158,11,0.15)","#fbbf24","&#x2193;","rgba(245,158,11,0.17)"),
                "Normal": ("#10b981","rgba(16,185,129,0.05)","rgba(16,185,129,0.12)","#34d399","&#x2713;","rgba(16,185,129,0.14)"),
            }
            for test, info in compared.items():
                v, u, s = info["value"], info["unit"], info["status"]
                border, bg, badge_bg, bc, icon, glow = cfg[s]
                ref = benchmarks.get(test, {})
                ref_r = f"{ref.get('min','?')}–{ref.get('max','?')} {u}" if ref else "N/A"
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {border}40;border-radius:13px;'
                    f'padding:1rem 1.1rem;position:relative;overflow:hidden;box-shadow:0 4px 18px {glow};">'
                    f'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:{border};opacity:0.55;border-radius:13px 13px 0 0;"></div>'
                    f'<div style="color:#64748b;font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem;">{test}</div>'
                    f'<div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:0.55rem;">'
                    f'<div><span style="color:#f1f5f9;font-size:1.7rem;font-weight:800;letter-spacing:-1.5px;">{v}</span>'
                    f'<span style="color:#475569;font-size:0.7rem;margin-left:3px;">{u}</span></div>'
                    f'<div style="background:{badge_bg};color:{bc};font-size:0.66rem;font-weight:700;'
                    f'padding:3px 8px;border-radius:20px;border:1px solid {border}45;">{icon} {s}</div>'
                    f'</div>'
                    f'<div style="background:rgba(0,0,0,0.2);border-radius:7px;padding:4px 7px;display:flex;align-items:center;gap:5px;">'
                    f'<span style="color:#334155;font-size:0.58rem;text-transform:uppercase;letter-spacing:0.4px;">REF</span>'
                    f'<span style="color:#64748b;font-size:0.68rem;">{ref_r}</span>'
                    f'</div></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            hi = sum(1 for i in compared.values() if i["status"]=="High")
            lo = sum(1 for i in compared.values() if i["status"]=="Low")
            nm = sum(1 for i in compared.values() if i["status"]=="Normal")

            st.markdown(
                f'<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);'
                f'border-radius:11px;padding:0.8rem 1.2rem;margin-top:0.8rem;'
                f'display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">'
                f'<span style="color:#475569;font-size:0.74rem;font-weight:600;">&#x1F4CB; Summary</span>'
                f'<span style="color:#34d399;font-size:0.76rem;">&#x25CF; {nm} Normal</span>'
                f'<span style="color:#fbbf24;font-size:0.76rem;">&#x25CF; {lo} Low</span>'
                f'<span style="color:#f87171;font-size:0.76rem;">&#x25CF; {hi} High</span>'
                f'<span style="color:#334155;font-size:0.74rem;margin-left:auto;">{len(compared)} tests</span>'
                f'</div></div>', unsafe_allow_html=True)

            # Doctor alert
            if (hi > 0 or lo > 0) and m and m.get("doctors"):
                d0 = m["doctors"][0]
                st.markdown(
                    f'<div style="max-width:1100px;margin:1rem auto 0;padding:0 2rem;">'
                    f'<div style="background:rgba(13,148,136,0.06);border:1px solid rgba(13,148,136,0.2);'
                    f'border-radius:13px;padding:0.9rem 1.4rem;display:flex;align-items:center;gap:12px;">'
                    f'<div style="font-size:1.4rem;">&#x1FA7A;</div>'
                    f'<div>'
                    f'<div style="color:#2dd4bf;font-size:0.8rem;font-weight:700;margin-bottom:2px;">'
                    f'{hi+lo} abnormal result(s) — consider contacting your doctor</div>'
                    f'<div style="color:#64748b;font-size:0.75rem;">Dr. {d0.get("name","")} · {d0.get("specialization","")}'
                    + (f' · {d0.get("phone","")}' if d0.get("phone") else "")
                    + f'</div></div></div></div>', unsafe_allow_html=True)

            # AI Explanation
            st.markdown(
                '<div style="max-width:1100px;margin:2rem auto 0;padding:0 2rem;">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:0.9rem;">'
                '<div style="width:3px;height:22px;background:linear-gradient(180deg,#8b5cf6,#06b6d4);border-radius:2px;"></div>'
                '<h2 style="color:#f1f5f9;font-size:1.15rem;font-weight:700;margin:0;">AI Explanation</h2>'
                '<div style="background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.28);'
                'border-radius:20px;padding:3px 10px;color:#a78bfa;font-size:0.66rem;font-weight:600;margin-left:4px;">'
                '&#10022; Profile-Aware + RAG</div>'
                '</div>', unsafe_allow_html=True)

            with st.spinner("🤖 Generating personalized explanation..."):
                explanation = ai_explain(compared, m)

            pt = m.get("name","you") or "you"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(139,92,246,0.07),rgba(6,182,212,0.04));'
                f'border:1px solid rgba(139,92,246,0.18);border-radius:16px;'
                f'padding:1.6rem 2rem;box-shadow:0 10px 35px rgba(139,92,246,0.09);">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">'
                f'<div style="width:34px;height:34px;border-radius:9px;'
                f'background:linear-gradient(135deg,#8b5cf6,#06b6d4);'
                f'display:flex;align-items:center;justify-content:center;font-size:0.95rem;">&#x1F916;</div>'
                f'<div>'
                f'<div style="color:#e2e8f0;font-weight:600;font-size:0.84rem;">LLaMA 3.1 · Personalized for {pt}</div>'
                f'<div style="color:#475569;font-size:0.67rem;">Patient-friendly · Non-diagnostic</div>'
                f'</div></div>'
                f'<div style="color:#cbd5e1;font-size:0.89rem;line-height:1.9;'
                f'white-space:pre-wrap;border-top:1px solid rgba(255,255,255,0.05);padding-top:1rem;">'
                f'{explanation}</div>'
                f'</div></div>', unsafe_allow_html=True)

        else:
            st.markdown(
                '<div style="max-width:500px;margin:3rem auto;padding:0 2rem;text-align:center;">'
                '<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);'
                'border-radius:16px;padding:2.5rem;">'
                '<div style="font-size:2.5rem;margin-bottom:0.8rem;">&#x1F50D;</div>'
                '<h3 style="color:#fbbf24;margin:0 0 0.5rem;font-size:1.05rem;font-weight:700;">No Values Detected</h3>'
                '<p style="color:#64748b;margin:0;font-size:0.82rem;line-height:1.6;">'
                'Could not extract recognized lab values. Try a clearer scan.</p>'
                '</div></div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div style="max-width:1100px;margin:2.5rem auto;padding:0 2rem 3rem;">'
        '<div style="background:rgba(15,23,42,0.5);border:1px solid rgba(255,255,255,0.06);'
        'border-radius:13px;padding:0.9rem 1.5rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        '<div style="font-size:0.85rem;">&#x26A0;&#xFE0F;</div>'
        '<div style="flex:1;">'
        '<div style="color:#fbbf24;font-size:0.73rem;font-weight:700;">For Educational Purposes Only</div>'
        '<div style="color:#475569;font-size:0.7rem;">Always consult a qualified healthcare provider.</div>'
        '</div>'
        '<div style="color:#1e293b;font-size:0.67rem;background:rgba(255,255,255,0.04);'
        'padding:3px 10px;border-radius:20px;">MedFamily &copy; 2025</div>'
        '</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════
pg = st.session_state.page

if pg == "hub":          page_hub()
elif pg == "add_member": page_add_member()
elif pg == "add_doctor": page_add_doctor()
elif pg == "profile_detail": page_profile_detail()
elif pg == "analyzer":   page_analyzer()
else:                    page_hub()