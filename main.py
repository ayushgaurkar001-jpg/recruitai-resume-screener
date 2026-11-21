from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pdfplumber
import shutil, re, json, io, uuid
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import spacy
from sentence_transformers import SentenceTransformer, util
import yake
from groq import Groq
import os
from dotenv import load_dotenv
from docx import Document
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

# =============================================
# GROQ AI CONFIGURATION
# =============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not set — will use basic scoring fallback")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# =============================================
# FIREBASE CONFIGURATION
# =============================================
if not firebase_admin._apps:
    firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_creds:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_candidate_to_firebase(candidate):
    try:
        safe_candidate = json.loads(json.dumps(candidate, default=str))
        db.collection("candidates").document(candidate["id"]).set(safe_candidate)
        print(f"✅ Saved {candidate['name']} to Firebase")
    except Exception as e:
        print(f"Firebase save error: {e}")

def save_job_to_firebase(job):
    try:
        db.collection("jobs").document(job["id"]).set(job)
        print(f"✅ Saved job {job['title']} to Firebase")
    except Exception as e:
        print(f"Firebase save error: {e}")

def load_from_firebase():
    try:
        candidates = []
        docs = db.collection("candidates").stream()
        for doc in docs:
            candidates.append(doc.to_dict())
        candidates.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        jobs = []
        docs = db.collection("jobs").stream()
        for doc in docs:
            jobs.append(doc.to_dict())
        print(f"✅ Loaded {len(candidates)} candidates and {len(jobs)} jobs from Firebase")
        return candidates, jobs
    except Exception as e:
        print(f"Firebase load error: {e}")
        return [], []

def update_candidate_status_firebase(filename, status):
    try:
        docs = db.collection("candidates").where("filename", "==", filename).stream()
        for doc in docs:
            doc.reference.update({"status": status})
    except Exception as e:
        print(f"Firebase update error: {e}")

def delete_job_from_firebase(job_id):
    try:
        db.collection("jobs").document(job_id).delete()
    except Exception as e:
        print(f"Firebase delete error: {e}")

def clear_all_candidates_firebase():
    try:
        docs = db.collection("candidates").stream()
        for doc in docs:
            doc.reference.delete()
        print("✅ Cleared all candidates from Firebase")
    except Exception as e:
        print(f"Firebase clear error: {e}")

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

# =============================================
# LOAD DATA FROM FIREBASE ON STARTUP
# =============================================
candidates_store, jobs_store = load_from_firebase()

# =============================================
# TRAINING SUGGESTIONS DATABASE
# =============================================
TRAINING_SUGGESTIONS = {
    "python": {"course": "Python for Everybody", "platform": "Coursera", "url": "https://www.coursera.org/specializations/python"},
    "machine learning": {"course": "Machine Learning Specialization", "platform": "Coursera", "url": "https://www.coursera.org/specializations/machine-learning-introduction"},
    "deep learning": {"course": "Deep Learning Specialization", "platform": "Coursera", "url": "https://www.coursera.org/specializations/deep-learning"},
    "nlp": {"course": "Natural Language Processing", "platform": "Coursera", "url": "https://www.coursera.org/specializations/natural-language-processing"},
    "sql": {"course": "SQL for Data Science", "platform": "Coursera", "url": "https://www.coursera.org/learn/sql-for-data-science"},
    "data science": {"course": "IBM Data Science", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/ibm-data-science"},
    "react": {"course": "React - The Complete Guide", "platform": "Udemy", "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux/"},
    "javascript": {"course": "The Complete JavaScript Course", "platform": "Udemy", "url": "https://www.udemy.com/course/the-complete-javascript-course/"},
    "docker": {"course": "Docker and Kubernetes", "platform": "Udemy", "url": "https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"},
    "aws": {"course": "AWS Certified Solutions Architect", "platform": "Udemy", "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03/"},
    "tensorflow": {"course": "TensorFlow Developer Certificate", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/tensorflow-in-practice"},
    "tableau": {"course": "Tableau Training", "platform": "Udemy", "url": "https://www.udemy.com/course/tableau10/"},
    "power bi": {"course": "Microsoft Power BI", "platform": "Udemy", "url": "https://www.udemy.com/course/powerbi-complete-introduction/"},
    "java": {"course": "Java Masterclass", "platform": "Udemy", "url": "https://www.udemy.com/course/java-the-complete-java-developer-course/"},
    "flutter": {"course": "Flutter & Dart", "platform": "Udemy", "url": "https://www.udemy.com/course/learn-flutter-dart-to-build-ios-android-apps/"},
    "cybersecurity": {"course": "Google Cybersecurity Certificate", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/google-cybersecurity"},
    "project management": {"course": "Google Project Management", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/google-project-management"},
    "digital marketing": {"course": "Google Digital Marketing", "platform": "Coursera", "url": "https://www.coursera.org/professional-certificates/google-digital-marketing-ecommerce"},
    "communication": {"course": "Business Communication Skills", "platform": "Coursera", "url": "https://www.coursera.org/learn/communication-skills"},
    "leadership": {"course": "Leadership and Management", "platform": "Coursera", "url": "https://www.coursera.org/specializations/leadership-development-for-engineers"},
}

# =============================================
# BIAS DETECTION PATTERNS
# =============================================
BIAS_PATTERNS = {
    "gender": ["he ", "she ", "him ", "her ", "his ", "man ", "woman ", "male ", "female ", "guys ", "manpower"],
    "age": ["young ", "fresh graduate only", "older", "age limit", "under 30", "under 25", "recent graduate only"],
    "appearance": ["attractive", "good looking", "presentable", "fair complexion", "height"],
    "religion": ["christian only", "hindu only", "muslim only", "specific religion"],
    "caste": ["specific caste", "upper caste", "brahmin only"],
}

# =============================================
# TEXT EXTRACTION — PDF + DOCX
# =============================================
def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"Text extraction error: {e}")
    return text

def extract_entities(text):
    doc = nlp(text[:5000])
    entities = {"organizations": [], "dates": [], "locations": [], "persons": []}
    for ent in doc.ents:
        if ent.label_ == "ORG": entities["organizations"].append(ent.text)
        elif ent.label_ == "DATE": entities["dates"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]: entities["locations"].append(ent.text)
        elif ent.label_ == "PERSON": entities["persons"].append(ent.text)
    return entities

def extract_keywords_yake(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0].lower() for kw in keywords]

def compute_semantic_similarity(resume_text, jd_text):
    resume_embedding = embedder.encode(resume_text[:1000], convert_to_tensor=True)
    jd_embedding = embedder.encode(jd_text[:1000], convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding)
    return round(float(similarity[0][0]) * 100, 2)

# =============================================
# BIAS DETECTION
# =============================================
def detect_bias(jd_text):
    jd_lower = jd_text.lower()
    biases_found = []
    for bias_type, patterns in BIAS_PATTERNS.items():
        for pattern in patterns:
            if pattern in jd_lower:
                biases_found.append({
                    "type": bias_type,
                    "pattern": pattern.strip(),
                    "suggestion": f"Remove '{pattern.strip()}' to make JD more inclusive"
                })
    return biases_found

# =============================================
# TRAINING SUGGESTIONS
# =============================================
def get_training_suggestions(missing_skills):
    suggestions = []
    for skill in missing_skills:
        skill_lower = skill.lower()
        for key, value in TRAINING_SUGGESTIONS.items():
            if key in skill_lower or skill_lower in key:
                suggestions.append({
                    "skill": skill,
                    "course": value["course"],
                    "platform": value["platform"],
                    "url": value["url"]
                })
                break
    return suggestions[:5]

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
    return filename.replace(".pdf", "").replace(".docx", "").replace("_", " ").title()

def extract_email(text):
    match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
    return match.group(0) if match else "N/A"

def extract_phone(text):
    match = re.search(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)
    return match.group(0) if match else "N/A"

def compute_final_score(ai_data, semantic_score):
    skill_score = ai_data.get("skill_score", 0)
    experience_score = ai_data.get("experience_score", 0)
    education_score = ai_data.get("education_score", 0)
    ats_score = ai_data.get("ats_score", 0)
    total = round(
        skill_score      * 0.35 +
        semantic_score   * 0.25 +
        experience_score * 0.20 +
        education_score  * 0.10 +
        ats_score        * 0.10,
        2
    )
    return min(total, 100)

# =============================================
# GROQ AI MODEL FALLBACK PIPELINE
# =============================================
GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
]

def analyze_with_ai(resume_text, job_description):
    prompt = f"""You are an expert HR recruiter and ATS system with 20 years experience.

Carefully analyze the resume against the job description.
Be accurate and thorough. Extract ALL skills mentioned.
Return ONLY a valid JSON object — no markdown, no explanation, no code blocks.

JOB DESCRIPTION:
{job_description}

RESUME TEXT:
{resume_text[:4000]}

Scoring instructions:
- skill_score: Percentage of required JD skills found in resume (0-100)
- experience_score: Score based on years of experience match (0-100)
- education_score: Score based on education requirement match (0-100)
- ats_score: ATS compatibility score based on keyword density (0-100)
- matched_skills: List ALL skills from JD found in resume
- missing_skills: List ALL skills from JD NOT in resume
- experience_years: Exact years of experience (number only)
- education_level: Highest degree (B.Tech/B.E/B.Sc/MBA/M.Tech/PhD/Diploma)
- strengths: 2 specific sentences about candidate strengths
- weaknesses: 1 specific sentence about biggest gap
- recommendation: Highly Recommended/Recommended/Maybe/Not Recommended
- summary: 3 professional sentences about candidate

Return exactly this JSON:
{{
    "skill_score": <0-100>,
    "experience_score": <0-100>,
    "education_score": <0-100>,
    "experience_years": <number>,
    "education_level": "<degree>",
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],
    "strengths": "<2 specific sentences>",
    "weaknesses": "<1 specific sentence>",
    "ats_score": <0-100>,
    "keyword_match_percent": <0-100>,
    "recommendation": "<Highly Recommended/Recommended/Maybe/Not Recommended>",
    "summary": "<3 sentence professional summary>"
}}"""

    # If no Groq client, use basic scoring directly
    if client is None:
        print("⚠️ No Groq client — using basic scoring")
        return basic_scoring(resume_text, job_description)

    # Try each model in sequence
    for model_name in GROQ_MODELS:
        try:
            print(f"Trying model: {model_name}")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert HR recruiter. Always return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            result = json.loads(raw)
            print(f"✅ Success with model: {model_name}")
            result["model_used"] = model_name
            return result
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
            continue

    # All models failed — use basic scoring
    print("⚠️ All Groq models failed — using basic scoring")
    return basic_scoring(resume_text, job_description)


def basic_scoring(resume_text, job_description):
    """Fallback scoring when all AI models fail"""
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd_lower))
    resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_lower))

    SKILLS = [
        "python", "java", "javascript", "sql", "machine learning", "deep learning",
        "nlp", "data analysis", "react", "nodejs", "html", "css", "flask", "fastapi",
        "tensorflow", "keras", "scikit", "pandas", "numpy", "git", "docker",
        "aws", "azure", "communication", "leadership", "teamwork"
    ]

    matched = [s for s in SKILLS if s in resume_lower and s in jd_lower]
    missing = [s for s in SKILLS if s in jd_lower and s not in resume_lower]

    exp_match = re.search(r'(\d+)\+?\s*years?', resume_lower)
    experience_years = int(exp_match.group(1)) if exp_match else 0

    edu_map = {
        "phd": ("PhD", 100), "m.tech": ("M.Tech", 85), "mba": ("MBA", 85),
        "b.tech": ("B.Tech", 70), "b.e": ("B.E", 70),
        "bca": ("BCA", 60), "diploma": ("Diploma", 40),
    }
    education_level = "Unknown"
    education_score = 30
    for key, (level, score) in edu_map.items():
        if key in resume_lower:
            education_level = level
            education_score = score
            break

    skill_score = round((len(matched) / max(len([s for s in SKILLS if s in jd_lower]), 1)) * 100, 2)
    exp_score = min(experience_years * 15, 100)
    keyword_match = round((len(jd_words & resume_words) / max(len(jd_words), 1)) * 100, 2)

    return {
        "skill_score": skill_score,
        "experience_score": exp_score,
        "education_score": education_score,
        "experience_years": experience_years,
        "education_level": education_level,
        "matched_skills": matched,
        "missing_skills": missing[:10],
        "strengths": "Candidate has relevant skills for this position.",
        "weaknesses": "Some required skills may be missing.",
        "ats_score": keyword_match,
        "keyword_match_percent": keyword_match,
        "recommendation": "Recommended" if skill_score > 50 else "Maybe",
        "summary": f"Candidate has {experience_years} years of experience with {education_level} degree. Skill match score is {skill_score}%.",
        "model_used": "basic_scoring"
    }

# =============================================
# ROUTES — DASHBOARD
# =============================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    total = len(candidates_store)
    selected = len([c for c in candidates_store if c["status"] == "Selected"])
    recommended = len([c for c in candidates_store if c["recommendation"] == "Highly Recommended"])
    avg_score = round(sum(c["total_score"] for c in candidates_store) / total, 1) if total else 0
    recent = sorted(candidates_store, key=lambda x: x["created_at"], reverse=True)[:5]
    return templates.TemplateResponse(request, "index.html", {
        "active": "dashboard",
        "total": total, "selected": selected,
        "recommended": recommended, "avg_score": avg_score,
        "recent": recent, "jobs": jobs_store,
        "candidates": candidates_store
    })

# =============================================
# ROUTES — JOBS
# =============================================
@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html", {
        "active": "jobs", "jobs": jobs_store
    })

@app.post("/jobs/create", response_class=HTMLResponse)
async def create_job(
    request: Request,
    title: str = Form(...),
    company: str = Form(...),
    description: str = Form(...),
    location: str = Form(""),
    job_type: str = Form("Full-time")
):
    bias_flags = detect_bias(description)
    job = {
        "id": str(uuid.uuid4())[:8],
        "title": title, "company": company,
        "description": description, "location": location,
        "job_type": job_type,
        "created_at": datetime.now().strftime("%d %b %Y"),
        "candidates_count": 0,
        "bias_flags": bias_flags
    }
    jobs_store.append(job)
    save_job_to_firebase(job)
    return templates.TemplateResponse(request, "jobs.html", {
        "active": "jobs", "jobs": jobs_store,
        "success": f"Job '{title}' created!",
        "bias_flags": bias_flags,
        "last_job": job
    })

@app.post("/jobs/delete", response_class=HTMLResponse)
async def delete_job(request: Request, job_id: str = Form(...)):
    global jobs_store
    jobs_store = [j for j in jobs_store if j["id"] != job_id]
    delete_job_from_firebase(job_id)
    return templates.TemplateResponse(request, "jobs.html", {
        "active": "jobs", "jobs": jobs_store
    })

# =============================================
# ROUTES — SCREENING
# =============================================
@app.get("/screen", response_class=HTMLResponse)
async def screen_page(request: Request):
    return templates.TemplateResponse(request, "screen.html", {
        "active": "screen", "jobs": jobs_store,
        "results": [], "job_keywords": []
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_description: str = Form(...),
    job_id: str = Form(""),
    resumes: list[UploadFile] = File(...)
):
    jd_keywords = extract_keywords_yake(job_description)
    bias_flags = detect_bias(job_description)

    if job_id:
        for job in jobs_store:
            if job["id"] == job_id:
                job["candidates_count"] += len(resumes)
                try:
                    db.collection("jobs").document(job_id).update({"candidates_count": job["candidates_count"]})
                except:
                    pass

    new_candidates = []
    for resume in resumes:
        filename = resume.filename
        path = f"uploads/{filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        text = extract_text(path)
        entities = extract_entities(text)
        name = extract_name(text, filename)
        email = extract_email(text)
        phone = extract_phone(text)
        semantic_score = compute_semantic_similarity(text, job_description)
        ai_data = analyze_with_ai(text, job_description)
        total_score = compute_final_score(ai_data, semantic_score)
        training = get_training_suggestions(ai_data.get("missing_skills", []))

        existing = next((c for c in candidates_store if c["filename"] == filename), None)
        if existing:
            candidates_store.remove(existing)
            try:
                docs = db.collection("candidates").where("filename", "==", filename).stream()
                for doc in docs:
                    doc.reference.delete()
            except:
                pass

        candidate = {
            "id": str(uuid.uuid4())[:8],
            "name": name, "email": email, "phone": phone,
            "filename": filename, "status": "Applied",
            "total_score": total_score,
            "semantic_score": semantic_score,
            "skill_score": ai_data.get("skill_score", 0),
            "experience_score": ai_data.get("experience_score", 0),
            "education_score": ai_data.get("education_score", 0),
            "ats_score": ai_data.get("ats_score", 0),
            "keyword_match_percent": ai_data.get("keyword_match_percent", 0),
            "experience_years": ai_data.get("experience_years", 0),
            "education_level": ai_data.get("education_level", "Unknown"),
            "matched_skills": ai_data.get("matched_skills", []),
            "missing_skills": ai_data.get("missing_skills", []),
            "strengths": ai_data.get("strengths", ""),
            "weaknesses": ai_data.get("weaknesses", ""),
            "recommendation": ai_data.get("recommendation", "Maybe"),
            "summary": ai_data.get("summary", ""),
            "entities": entities,
            "jd_keywords": jd_keywords,
            "job_id": job_id,
            "training_suggestions": training,
            "created_at": datetime.now().strftime("%d %b %Y %H:%M"),
        }
        candidates_store.append(candidate)
        new_candidates.append(candidate)
        save_candidate_to_firebase(candidate)

    new_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    candidates_store.sort(key=lambda x: x["total_score"], reverse=True)

    return templates.TemplateResponse(request, "screen.html", {
        "active": "screen",
        "jobs": jobs_store,
        "results": new_candidates,
        "job_keywords": jd_keywords,
        "job_description": job_description,
        "bias_flags": bias_flags
    })

# =============================================
# ROUTES — CANDIDATES
# =============================================
@app.get("/candidates", response_class=HTMLResponse)
async def candidates_page(request: Request, status: str = "", search: str = ""):
    filtered = candidates_store
    if status:
        filtered = [c for c in filtered if c["status"] == status]
    if search:
        filtered = [c for c in filtered if search.lower() in c["name"].lower()]
    return templates.TemplateResponse(request, "candidates.html", {
        "active": "candidates",
        "candidates": filtered,
        "status_filter": status,
        "search": search,
        "total": len(candidates_store)
    })

@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request, id1: str = "", id2: str = ""):
    c1 = next((c for c in candidates_store if c["id"] == id1), None)
    c2 = next((c for c in candidates_store if c["id"] == id2), None)
    return templates.TemplateResponse(request, "compare.html", {
        "active": "candidates",
        "candidates": candidates_store,
        "c1": c1, "c2": c2
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
    update_candidate_status_firebase(filename, status)
    return templates.TemplateResponse(request, "candidates.html", {
        "active": "candidates",
        "candidates": candidates_store,
        "status_filter": "", "search": "",
        "total": len(candidates_store)
    })

# =============================================
# ROUTES — ANALYTICS
# =============================================
@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    if not candidates_store:
        return templates.TemplateResponse(request, "analytics.html", {
            "active": "analytics", "has_data": False
        })
    scores = [c["total_score"] for c in candidates_store]
    avg_score = round(sum(scores) / len(scores), 1)
    top_score = max(scores)
    status_counts = {}
    for c in candidates_store:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    rec_counts = {}
    for c in candidates_store:
        rec_counts[c["recommendation"]] = rec_counts.get(c["recommendation"], 0) + 1
    all_matched = []
    for c in candidates_store:
        all_matched.extend(c["matched_skills"])
    skill_freq = {}
    for s in all_matched:
        skill_freq[s] = skill_freq.get(s, 0) + 1
    top_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    return templates.TemplateResponse(request, "analytics.html", {
        "active": "analytics", "has_data": True,
        "total": len(candidates_store),
        "avg_score": avg_score, "top_score": top_score,
        "status_counts": status_counts, "rec_counts": rec_counts,
        "top_skills": top_skills, "candidates": candidates_store
    })

# =============================================
# ROUTES — EXPORT EXCEL
# =============================================
@app.get("/export/excel")
async def export_excel():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Candidate Rankings"
    headers = ["Rank", "Name", "Email", "Phone", "Total Score", "ATS Score",
               "Skill Score", "Semantic Score", "Experience Score", "Education Score",
               "Experience (Years)", "Education", "Recommendation", "Status",
               "Matched Skills", "Missing Skills", "Strengths", "Weaknesses", "Summary"]
    header_fill = PatternFill(start_color="7C6FFF", end_color="7C6FFF", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    for rank, c in enumerate(candidates_store, 1):
        ws.append([
            rank, c["name"], c["email"], c["phone"],
            c["total_score"], c["ats_score"], c["skill_score"],
            c["semantic_score"], c["experience_score"], c["education_score"],
            c["experience_years"], c["education_level"], c["recommendation"],
            c["status"], ", ".join(c.get("matched_skills", [])),
            ", ".join(c.get("missing_skills", [])), c["strengths"],
            c["weaknesses"], c.get("summary", "")
        ])
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=candidate_rankings.xlsx"}
    )

@app.post("/clear")
async def clear(request: Request):
    candidates_store.clear()
    clear_all_candidates_firebase()
    return templates.TemplateResponse(request, "screen.html", {
        "active": "screen", "jobs": jobs_store,
        "results": [], "job_keywords": [],
        "job_description": ""
    })
