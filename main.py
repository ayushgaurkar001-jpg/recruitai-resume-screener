from dotenv import load_dotenv
load_dotenv()
import os
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pdfplumber
import shutil, re, json
import spacy
from sentence_transformers import SentenceTransformer, util
import yake
from google import genai as google_genai

# =============================================
# CONFIGURATION
# =============================================
client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# =============================================
# LOAD NLP MODELS
# =============================================
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=20)
print("All models loaded!")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_entities(text):
    doc = nlp(text[:5000])
    entities = {
        "organizations": [],
        "dates": [],
        "locations": [],
        "persons": []
    }
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["locations"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
    return entities

def extract_keywords_yake(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0].lower() for kw in keywords]

def compute_semantic_similarity(resume_text, jd_text):
    resume_embedding = embedder.encode(resume_text[:1000], convert_to_tensor=True)
    jd_embedding = embedder.encode(jd_text[:1000], convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding)
    return round(float(similarity[0][0]) * 100, 2)

def analyze_with_gemini(resume_text, job_description):
    prompt = f"""
You are an expert HR recruiter and resume screening AI.

Analyze the following resume against the job description and return ONLY a JSON object with no extra text, no markdown, no code blocks.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text[:3000]}

Return exactly this JSON structure:
{{
    "skill_score": <number 0-100>,
    "experience_score": <number 0-100>,
    "education_score": <number 0-100>,
    "experience_years": <number>,
    "education_level": "<highest degree>",
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],
    "strengths": "<2 sentence summary of candidate strengths>",
    "weaknesses": "<1 sentence summary of gaps>",
    "recommendation": "<Highly Recommended / Recommended / Maybe / Not Recommended>"
}}
"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return {
            "skill_score": 0,
            "experience_score": 0,
            "education_score": 0,
            "experience_years": 0,
            "education_level": "Unknown",
            "matched_skills": [],
            "missing_skills": [],
            "strengths": "Could not analyze.",
            "weaknesses": "Could not analyze.",
            "recommendation": "Maybe"
        }

def extract_name(text, filename):
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            return ent.text.title()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in lines[:5]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            return line.title()
    return filename.replace(".pdf", "").replace("_", " ").title()

def compute_final_score(gemini_data, semantic_score):
    skill_score = gemini_data.get("skill_score", 0)
    experience_score = gemini_data.get("experience_score", 0)
    education_score = gemini_data.get("education_score", 0)
    total = round(
        semantic_score   * 0.25 +
        skill_score      * 0.40 +
        experience_score * 0.25 +
        education_score  * 0.10,
        2
    )
    return min(total, 100)

candidates_store = []
job_desc_store = {"text": "", "keywords": []}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "results": candidates_store,
        "job_description": job_desc_store["text"],
        "job_keywords": job_desc_store["keywords"]
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...)
):
    jd_keywords = extract_keywords_yake(job_description)
    job_desc_store["text"] = job_description
    job_desc_store["keywords"] = jd_keywords

    for resume in resumes:
        path = f"uploads/{resume.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        text = extract_text(path)
        entities = extract_entities(text)
        name = extract_name(text, resume.filename)
        semantic_score = compute_semantic_similarity(text, job_description)
        gemini_data = analyze_with_gemini(text, job_description)
        total_score = compute_final_score(gemini_data, semantic_score)

        existing = next((c for c in candidates_store if c["filename"] == resume.filename), None)
        if existing:
            candidates_store.remove(existing)

        candidate = {
            "name": name,
            "filename": resume.filename,
            "status": "Applied",
            "total_score": total_score,
            "semantic_score": semantic_score,
            "skill_score": gemini_data.get("skill_score", 0),
            "experience_score": gemini_data.get("experience_score", 0),
            "education_score": gemini_data.get("education_score", 0),
            "experience_years": gemini_data.get("experience_years", 0),
            "education_level": gemini_data.get("education_level", "Unknown"),
            "matched_skills": gemini_data.get("matched_skills", []),
            "missing_skills": gemini_data.get("missing_skills", []),
            "strengths": gemini_data.get("strengths", ""),
            "weaknesses": gemini_data.get("weaknesses", ""),
            "recommendation": gemini_data.get("recommendation", "Maybe"),
            "entities": entities,
            "jd_keywords": jd_keywords
        }
        candidates_store.append(candidate)

    candidates_store.sort(key=lambda x: x["total_score"], reverse=True)

    return templates.TemplateResponse(request, "index.html", {
        "results": candidates_store,
        "job_description": job_description,
        "job_keywords": jd_keywords
    })

@app.post("/update-status", response_class=HTMLResponse)
async def update_status(
    request: Request,
    filename: str = Form(...),
    status: str = Form(...)
):
    for candidate in candidates_store:
        if candidate["filename"] == filename:
            candidate["status"] = status
            break
    return templates.TemplateResponse(request, "index.html", {
        "results": candidates_store,
        "job_description": job_desc_store["text"],
        "job_keywords": job_desc_store["keywords"]
    })

@app.post("/clear", response_class=HTMLResponse)
async def clear(request: Request):
    candidates_store.clear()
    job_desc_store["text"] = ""
    job_desc_store["keywords"] = []
    return templates.TemplateResponse(request, "index.html", {
        "results": [],
        "job_description": "",
        "job_keywords": []
    })