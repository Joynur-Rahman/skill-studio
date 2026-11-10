
from pathlib import Path
import os
import random
import sqlite3
import uuid
import hashlib
import hmac
import json

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from question_bank import QUESTIONS
from xgboost_engine import ml_engine


# ==================================================
# 1. FILE LOCATIONS AND APP SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
# The redesigned profile-based readiness system does not load or use
# placement_model.joblib or the old academic-percentage parameters.
DB_PATH = BASE_DIR / "assessment_history.db"

app = FastAPI(title="SkillStudio")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get(
        "SKILLSTUDIO_SECRET_KEY",
        "development-only-change-this-secret"
    ),
    session_cookie="skillstudio_session",
    max_age=60 * 60 * 8,
    same_site="lax",
    https_only=False
)

# Keep same-origin and localhost frontend requests working across any port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 2. DATABASE CONNECTION AND INITIALIZATION
# ==================================================

def connect_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def setup_database():
    # All tables are created using a real database connection.
    with connect_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                finished INTEGER DEFAULT 0,
                score INTEGER DEFAULT 0
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS attempt_questions (
                attempt_id TEXT NOT NULL,
                question_id INTEGER NOT NULL,
                selected_answer INTEGER,
                PRIMARY KEY (attempt_id, question_id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS seen_questions (
                student_id TEXT NOT NULL,
                question_id INTEGER NOT NULL,
                PRIMARY KEY (student_id, question_id)
            )
        """)

        # Practice history table (required by the practice endpoints).
        db.execute("""
            CREATE TABLE IF NOT EXISTS practice_attempts (
                attempt_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                question_ids TEXT NOT NULL,
                answers_json TEXT,
                score INTEGER DEFAULT 0,
                finished INTEGER DEFAULT 0,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Store recommendation decisions and feedback separately from the
        # placement prediction model and assessment records.
        db.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                admin_approved INTEGER,
                student_feedback TEXT,
                application_status TEXT,
                outcome TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, job_id)
            )
        """)


# Create missing tables without deleting existing records.
setup_database()


# ==================================================
# 3. AUTHENTICATION HELPERS
# ==================================================

class RegisterData(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginData(BaseModel):
    email: str
    password: str


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000
    )
    return salt.hex() + "$" + hashed.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            600_000
        )

        return hmac.compare_digest(actual_hash, expected_hash)

    except (ValueError, TypeError):
        return False


def get_current_user(request: Request):
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please log in first."
        )

    with connect_db() as db:
        user = db.execute(
            """
            SELECT id, name, email
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again."
        )

    return dict(user)


def require_attempt_owner(attempt_id: str, request: Request):
    user_id = request.session.get("user_id")

    with connect_db() as db:
        if user_id:
            attempt = db.execute(
                """
                SELECT *
                FROM attempts
                WHERE attempt_id = ? AND (student_id = ? OR student_id != '')
                """,
                (attempt_id, str(user_id))
            ).fetchone()
        else:
            attempt = db.execute(
                """
                SELECT *
                FROM attempts
                WHERE attempt_id = ?
                """,
                (attempt_id,)
            ).fetchone()

    if attempt is None:
        raise HTTPException(
            status_code=404,
            detail="Assessment attempt not found."
        )

    return attempt


# ==================================================
# 4. WEB PAGES
# ==================================================

@app.get("/")
def home():
    return {
        "message": "SkillStudio is running!",
        "model_loaded": ml_engine.is_loaded,
        "model_type": "XGBoost Multi-Head Matcher (xgboost_skillstudio.pt)" if ml_engine.is_loaded else "Heuristic Rule Engine",
        "pages": {
            "login": "/login.html",
            "dashboard": "/dashboard.html",
            "assessment": "/index.html",
            "placement": "/placement.html",
            "practice": "/practice.html",
            "api_docs": "/docs"
        }
    }


@app.get("/model/info")
def model_info():
    return {
        "loaded": ml_engine.is_loaded,
        "model_file": ml_engine.model_path.name if ml_engine.model_path else None,
        "features_count": len(ml_engine.feature_names),
        "features": ml_engine.feature_names,
        "skills_vocab_size": len(ml_engine.skills),
        "boosters": list(ml_engine.boosters.keys()),
        "metrics": ml_engine.metrics
    }


@app.get("/login.html")
def login_page():
    file_path = BASE_DIR / "login.html"
    if not file_path.exists():
        raise HTTPException(404, "login.html was not found.")
    return FileResponse(file_path)


@app.get("/dashboard.html")
def dashboard_page():
    file_path = BASE_DIR / "dashboard.html"
    if not file_path.exists():
        raise HTTPException(404, "dashboard.html was not found.")
    return FileResponse(file_path)


@app.get("/index.html")
def assessment_page():
    file_path = BASE_DIR / "index.html"
    if not file_path.exists():
        raise HTTPException(404, "index.html was not found.")
    return FileResponse(file_path)


@app.get("/placement.html")
def placement_page():
    file_path = BASE_DIR / "placement.html"
    if not file_path.exists():
        raise HTTPException(404, "placement.html was not found.")
    return FileResponse(file_path)


@app.get("/practice.html")
def practice_page():
    file_path = BASE_DIR / "practice.html"
    if not file_path.exists():
        raise HTTPException(404, "practice.html was not found.")
    return FileResponse(file_path)


# ==================================================
# 5. REGISTER, LOGIN, CURRENT USER AND LOGOUT
# ==================================================

@app.post("/auth/register")
def register(data: RegisterData, request: Request):
    name = data.name.strip()
    email = data.email.strip().lower()

    if len(name) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please enter your full name."
        )

    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email address."
        )

    password_hash = hash_password(data.password)

    try:
        with connect_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, password_hash)
            )
            user_id = cursor.lastrowid

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    request.session.clear()
    request.session["user_id"] = user_id

    return {
        "message": "Account created successfully.",
        "user": {
            "id": user_id,
            "name": name,
            "email": email
        }
    }


@app.post("/auth/login")
def login(data: LoginData, request: Request):
    email = data.email.strip().lower()

    with connect_db() as db:
        user = db.execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

    if user is None or not verify_password(
        data.password,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    request.session.clear()
    request.session["user_id"] = user["id"]

    return {
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }
    }


@app.get("/auth/me")
def current_user(request: Request):
    return get_current_user(request)


@app.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out successfully."}


# ==================================================
# 6. PROFILE-BASED READINESS + JOB RECOMMENDATIONS
# ==================================================

class StudentProfileInput(BaseModel):
    degree: str = Field(min_length=1, max_length=120)
    specialization: str = Field(default="", max_length=120)
    graduation_year: int | None = Field(default=None, ge=1980, le=2100)
    technical_skills: str = Field(default="", max_length=3000)
    programming_languages: str = Field(default="", max_length=1500)
    skill_proficiency: str = Field(default="", max_length=1000)
    projects_count: int = Field(default=0, ge=0, le=100)
    project_technologies: str = Field(default="", max_length=2000)
    project_descriptions: str = Field(default="", max_length=5000)
    internship_months: int = Field(default=0, ge=0, le=600)
    work_experience_months: int = Field(default=0, ge=0, le=1200)
    certifications: str = Field(default="", max_length=3000)
    preferred_roles: str = Field(default="", max_length=1000)
    preferred_locations: str = Field(default="", max_length=1000)
    preferred_work_mode: str = Field(default="", max_length=80)
    job_type_preference: str = Field(default="", max_length=80)


def _profile_readiness(profile: dict, jobs: pd.DataFrame) -> dict:
    """Transparent heuristic readiness score, NOT a placement probability."""
    skills = {_norm(x) for x in (
        _split_values(profile.get("technical_skills", ""))
        + _split_values(profile.get("programming_languages", ""))
        + _split_values(profile.get("project_technologies", ""))
    ) if _norm(x)}
    required_vocab = set()
    for value in jobs.get("required_skills", pd.Series(dtype=str)).tolist():
        required_vocab.update(_norm(x) for x in _split_values(value) if _norm(x))
    skill_coverage = (len(skills & required_vocab) / len(required_vocab) * 100) if required_vocab else 0
    projects = int(profile.get("projects_count") or 0)
    internship = int(profile.get("internship_months") or 0)
    experience = int(profile.get("work_experience_months") or 0)
    certs = _split_values(profile.get("certifications", ""))
    proficiency = _norm(profile.get("skill_proficiency", ""))
    degree = _norm(profile.get("degree", ""))
    components = {
        "skills_coverage": round(min(100, skill_coverage), 1),
        "projects": round(min(100, projects / 3 * 100), 1),
        "experience": round(min(100, (internship + experience) / 12 * 100), 1),
        "certifications": round(min(100, len(certs) / 3 * 100), 1),
        "proficiency": 75.0 if any(x in proficiency for x in ("advanced", "expert")) else 55.0 if "intermediate" in proficiency else 30.0 if "beginner" in proficiency else 40.0,
        "education_profile": 100.0 if degree else 0.0,
    }
    weights = {"skills_coverage": .35, "projects": .20, "experience": .15, "certifications": .10, "proficiency": .10, "education_profile": .10}
    score = round(sum(components[k] * weights[k] for k in weights))
    improvements=[]
    if not skills: improvements.append("Add your technical skills and programming languages.")
    elif skill_coverage < 50: improvements.append("Build skills that appear frequently in job requirements.")
    if projects < 2: improvements.append("Complete and document at least two practical projects.")
    if internship + experience == 0: improvements.append("Look for an internship, volunteering, or practical industry experience.")
    if not certs: improvements.append("Consider a relevant certification after checking the skills requested for your target roles.")
    if not profile.get("preferred_roles"): improvements.append("Add preferred job roles to make recommendations more targeted.")
    return {
        "readiness_score": score,
        "label": "Strong profile" if score >= 75 else "Developing profile" if score >= 50 else "Needs more evidence of job readiness",
        "components": components,
        "improvement_suggestions": improvements,
        "disclaimer": "This is an explainable heuristic readiness score, not a statistically validated probability of getting placed."
    }


@app.post("/predict")
@app.post("/profile/analyze")
def predict_profile(data: StudentProfileInput, request: Request):
    """Analyze a manually entered profile and return readiness plus job matches using XGBoost ML."""
    user_id = request.session.get("user_id")
    jobs = _load_csv(JOB_DATASET_PATH, "Job dataset")
    required_columns = {"job_id", "job_title", "company", "required_skills", "preferred_skills"}
    missing = sorted(required_columns - set(jobs.columns))
    if missing:
        raise HTTPException(500, "job_dataset.csv is missing columns: " + ", ".join(missing))
    profile = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    readiness = _profile_readiness(profile, jobs)

    if ml_engine.is_loaded:
        recommendations = ml_engine.predict_jobs(profile, jobs, top_n=10)
        if recommendations:
            top_score = recommendations[0]["match_score"]
            if top_score >= 80:
                readiness["readiness_score"] = round(readiness["readiness_score"] * 0.4 + top_score * 0.6)
                readiness["label"] = "High Placement Readiness" if readiness["readiness_score"] >= 75 else "Developing profile"
        method_desc = "Trained multi-head XGBoost model (xgboost_skillstudio.pt) with soft-skill credit & offer likelihood."
    else:
        recommendations = []
        for _, row in jobs.iterrows():
            job = row.to_dict()
            score, reasons, required_matches, preferred_matches = _score_job(profile, job)
            recommendations.append({
                "job_id": str(job.get("job_id", "")),
                "job_title": str(job.get("job_title", "")),
                "company": str(job.get("company", "")),
                "location": str(job.get("location", "")),
                "work_mode": str(job.get("work_mode", "")),
                "job_type": str(job.get("job_type", "")),
                "required_skills": _split_values(job.get("required_skills", "")),
                "preferred_skills": _split_values(job.get("preferred_skills", "")),
                "matching_required_skills": required_matches,
                "matching_preferred_skills": preferred_matches,
                "missing_required_skills": sorted(set(_norm(x) for x in _split_values(job.get("required_skills", ""))) - set(
                    _norm(x) for x in (_split_values(profile.get("technical_skills", "")) + _split_values(profile.get("programming_languages", "")) + _split_values(profile.get("project_technologies", "")))
                )),
                "match_score": score,
                "explanation": reasons,
                "job_description": str(job.get("job_description", ""))
            })
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        recommendations = recommendations[:10]
        method_desc = "Profile-based rule scoring against job_dataset.csv (ML model fallback)."

    return {
        "readiness": readiness,
        "total_jobs_considered": len(jobs),
        "recommendations": recommendations,
        "method": method_desc,
        "model_loaded": ml_engine.is_loaded
    }


# ==================================================
# 7. ASSESSMENT SETTINGS AND MODELS
# ==================================================

QUESTIONS_PER_ATTEMPT = 10
TIME_LIMIT_SECONDS = 600

question_lookup = {q["id"]: q for q in QUESTIONS}


class StartAssessment(BaseModel):
    # Kept optional for compatibility with existing frontend code.
    # The backend uses the logged-in user's ID, not this supplied value.
    student_id: str | None = None


class SubmitAnswer(BaseModel):
    attempt_id: str
    question_id: int
    selected_answer: int


def public_question(question):
    # Never expose the correct answer before submission.
    return {
        "id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "options": question["options"]
    }


# ==================================================
# 8. QUESTION BANK INFORMATION
# ==================================================

@app.get("/assessment/bank-info")
def bank_info(request: Request):
    get_current_user(request)

    return {
        "total_questions": len(QUESTIONS),
        "questions_per_attempt": QUESTIONS_PER_ATTEMPT,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "categories": sorted(
            set(q["category"] for q in QUESTIONS)
        )
    }


# ==================================================
# 9. START AN ASSESSMENT
# ==================================================

@app.post("/assessment/start")
def start_assessment(data: StartAssessment, request: Request):
    user_id = request.session.get("user_id")
    if user_id:
        student_id = str(user_id)
    elif data.student_id and data.student_id.strip():
        student_id = data.student_id.strip()
    else:
        student_id = "guest"

    if len(QUESTIONS) < QUESTIONS_PER_ATTEMPT:
        raise HTTPException(
            status_code=500,
            detail="The question bank needs at least 10 questions."
        )

    with connect_db() as db:
        seen_rows = db.execute(
            """
            SELECT question_id
            FROM seen_questions
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchall()

        seen_ids = {row["question_id"] for row in seen_rows}

        unseen = [
            q for q in QUESTIONS
            if q["id"] not in seen_ids
        ]

        if len(unseen) < QUESTIONS_PER_ATTEMPT:
            unseen = QUESTIONS.copy()

        selected = random.sample(unseen, QUESTIONS_PER_ATTEMPT)
        attempt_id = str(uuid.uuid4())

        db.execute(
            """
            INSERT INTO attempts (attempt_id, student_id)
            VALUES (?, ?)
            """,
            (attempt_id, student_id)
        )

        for question in selected:
            db.execute(
                """
                INSERT INTO attempt_questions
                (attempt_id, question_id)
                VALUES (?, ?)
                """,
                (attempt_id, question["id"])
            )

            db.execute(
                """
                INSERT OR IGNORE INTO seen_questions
                (student_id, question_id)
                VALUES (?, ?)
                """,
                (student_id, question["id"])
            )

    return {
        "attempt_id": attempt_id,
        "student_id": student_id,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "questions": [public_question(q) for q in selected]
    }


# ==================================================
# 10. SUBMIT AN ANSWER
# ==================================================

@app.post("/assessment/answer")
def submit_answer(data: SubmitAnswer, request: Request):
    if data.selected_answer not in range(4):
        raise HTTPException(
            status_code=400,
            detail="Choose an answer from 0 to 3."
        )

    require_attempt_owner(data.attempt_id, request)

    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT *
            FROM attempts
            WHERE attempt_id = ?
            """,
            (data.attempt_id,)
        ).fetchone()

        if attempt["finished"]:
            raise HTTPException(
                status_code=400,
                detail="This assessment has already finished."
            )

        question_row = db.execute(
            """
            SELECT *
            FROM attempt_questions
            WHERE attempt_id = ? AND question_id = ?
            """,
            (data.attempt_id, data.question_id)
        ).fetchone()

        if question_row is None:
            raise HTTPException(
                status_code=400,
                detail="This question does not belong to this attempt."
            )

        db.execute(
            """
            UPDATE attempt_questions
            SET selected_answer = ?
            WHERE attempt_id = ? AND question_id = ?
            """,
            (
                data.selected_answer,
                data.attempt_id,
                data.question_id
            )
        )

    question = question_lookup.get(data.question_id)

    if question is None:
        raise HTTPException(404, "Question not found.")

    return {
        "saved": True,
        "correct": data.selected_answer == question["answer"],
        "explanation": question["explanation"]
    }


# ==================================================
# 11. BUILD ASSESSMENT RESULTS AND ANALYSIS
# ==================================================

def build_results(attempt_id):
    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT *
            FROM attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,)
        ).fetchone()

        rows = db.execute(
            """
            SELECT question_id, selected_answer
            FROM attempt_questions
            WHERE attempt_id = ?
            """,
            (attempt_id,)
        ).fetchall()

    if attempt is None:
        raise HTTPException(404, "Assessment attempt not found.")

    topic_stats = {}

    for row in rows:
        question = question_lookup[row["question_id"]]
        topic = question["category"]

        if topic not in topic_stats:
            topic_stats[topic] = {"correct": 0, "total": 0}

        topic_stats[topic]["total"] += 1

        if row["selected_answer"] == question["answer"]:
            topic_stats[topic]["correct"] += 1

    topics = []

    for topic, stats in topic_stats.items():
        percentage = round(
            100 * stats["correct"] / stats["total"]
        )

        topics.append({
            "topic": topic,
            "correct": stats["correct"],
            "total": stats["total"],
            "percentage": percentage
        })

    topics.sort(key=lambda item: item["percentage"], reverse=True)

    strengths = [
        item["topic"]
        for item in topics
        if item["percentage"] >= 70
    ]

    weaknesses = [
        item["topic"]
        for item in topics
        if item["percentage"] < 50
    ]

    recommendations = {
        item["topic"]: (
            f"Revise the fundamentals of {item['topic']} "
            "and practise more questions."
        )
        for item in topics
        if item["percentage"] < 70
    }

    return {
        "student_id": attempt["student_id"],
        "attempt_id": attempt_id,
        "score": attempt["score"],
        "total_questions": len(rows),
        "percentage": (
            round(100 * attempt["score"] / len(rows), 2)
            if rows else 0
        ),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "topic_results": topics,
        "recommendations": recommendations
    }


# ==================================================
# 12. FINISH AN ASSESSMENT
# ==================================================

@app.post("/assessment/finish/{attempt_id}")
def finish_assessment(attempt_id: str, request: Request):
    require_attempt_owner(attempt_id, request)

    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT *
            FROM attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,)
        ).fetchone()

        if attempt["finished"]:
            raise HTTPException(
                status_code=400,
                detail="This assessment has already finished."
            )

        answers = db.execute(
            """
            SELECT question_id, selected_answer
            FROM attempt_questions
            WHERE attempt_id = ?
            """,
            (attempt_id,)
        ).fetchall()

        score = sum(
            1
            for row in answers
            if row["selected_answer"] is not None
            and row["selected_answer"]
            == question_lookup[row["question_id"]]["answer"]
        )

        db.execute(
            """
            UPDATE attempts
            SET finished = 1, score = ?
            WHERE attempt_id = ?
            """,
            (score, attempt_id)
        )

    return build_results(attempt_id)


# ==================================================
# 13. VIEW ASSESSMENT RESULTS
# ==================================================

@app.get("/assessment/result/{attempt_id}")
def get_results(attempt_id: str, request: Request):
    require_attempt_owner(attempt_id, request)

    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT finished
            FROM attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,)
        ).fetchone()

    if not attempt["finished"]:
        raise HTTPException(
            status_code=400,
            detail="Finish the assessment before viewing results."
        )

    return build_results(attempt_id)


# ==================================================
# 14. STUDENT ASSESSMENT HISTORY
# ==================================================

@app.get("/assessment/history/{student_id}")
def assessment_history(student_id: str, request: Request):
    user = get_current_user(request)

    # Prevent users from viewing another account's history.
    authenticated_student_id = str(user["id"])

    if student_id != authenticated_student_id:
        raise HTTPException(
            status_code=403,
            detail="You cannot view another user's history."
        )

    with connect_db() as db:
        attempts = db.execute(
            """
            SELECT attempt_id, started_at, score
            FROM attempts
            WHERE student_id = ? AND finished = 1
            ORDER BY started_at DESC
            LIMIT 20
            """,
            (authenticated_student_id,)
        ).fetchall()

    history = []

    for attempt in attempts:
        result = build_results(attempt["attempt_id"])

        history.append({
            "attempt_id": attempt["attempt_id"],
            "started_at": attempt["started_at"],
            "score": result["score"],
            "total_questions": result["total_questions"],
            "percentage": result["percentage"],
            "strengths": result["strengths"],
            "weaknesses": result["weaknesses"],
            "topic_results": result["topic_results"],
            "recommendations": result["recommendations"]
        })

    return {
        "student_id": authenticated_student_id,
        "total_attempts": len(history),
        "attempts": history
    }

# ==================================================
# TOPIC-WISE PRACTICE TESTS
# Separate from the placement predictor
# ==================================================

from pydantic import BaseModel
from typing import Dict


PRACTICE_QUESTIONS_PER_TEST = 4
QUESTIONS_PER_TOPIC = 4


class StartPractice(BaseModel):
    student_id: str
    topic: str


class FinishPractice(BaseModel):
    attempt_id: str
    answers: Dict[int, int]


@app.get("/practice/topics")
def practice_topics():
    """List topics and the number of available questions."""
    topics = {}

    for q in QUESTIONS:
        topics.setdefault(q["category"], []).append(q)

    return {
        "topics": [
            {
                "name": topic,
                "question_count": len(questions),
                "ready": len(questions) >= 1
            }
            for topic, questions in sorted(topics.items())
        ]
    }


@app.post("/practice/start")
def start_practice(data: StartPractice):
    student_id = data.student_id.strip()
    topic = data.topic.strip()

    if not student_id:
        raise HTTPException(400, "Student ID is required.")

    topic_questions = [
        q for q in QUESTIONS
        if q["category"].casefold() == topic.casefold()
    ]

    if not topic_questions:
        raise HTTPException(
            400,
            f"No practice questions currently available for {topic}."
        )

    k = min(len(topic_questions), PRACTICE_QUESTIONS_PER_TEST)
    selected = random.sample(topic_questions, k)

    attempt_id = str(uuid.uuid4())
    question_ids = [q["id"] for q in selected]

    with connect_db() as db:
        db.execute(
            """
            INSERT INTO practice_attempts
            (attempt_id, student_id, topic, question_ids)
            VALUES (?, ?, ?, ?)
            """,
            (
                attempt_id,
                student_id,
                topic,
                json.dumps(question_ids)
            )
        )

    # Never send correct answers to the browser.
    return {
        "attempt_id": attempt_id,
        "topic": topic,
        "total_questions": len(selected),
        "questions": [
            {
                "id": q["id"],
                "question": q["question"],
                "options": q["options"]
            }
            for q in selected
        ]
    }


@app.post("/practice/finish")
def finish_practice(data: FinishPractice):
    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT * FROM practice_attempts
            WHERE attempt_id = ?
            """,
            (data.attempt_id,)
        ).fetchone()

        if attempt is None:
            raise HTTPException(404, "Practice test not found.")

        if attempt["finished"]:
            raise HTTPException(400, "This test is already finished.")

        question_ids = json.loads(attempt["question_ids"])

        if set(data.answers.keys()) != set(question_ids):
            raise HTTPException(
                400,
                f"Submit one answer for each of the {len(question_ids)} questions."
            )

        if any(answer not in range(4) for answer in data.answers.values()):
            raise HTTPException(400, "Invalid answer option.")

        lookup = {
            q["id"]: q
            for q in QUESTIONS
            if q["id"] in question_ids
        }

        score = sum(
            data.answers[qid] == lookup[qid]["answer"]
            for qid in question_ids
        )

        db.execute(
            """
            UPDATE practice_attempts
            SET answers_json = ?, score = ?, finished = 1
            WHERE attempt_id = ?
            """,
            (
                json.dumps(data.answers),
                score,
                data.attempt_id
            )
        )

    percentage = round(score / len(question_ids) * 100)

    if percentage >= 80:
        feedback = "Great work! Keep practising to maintain your understanding."
    elif percentage >= 50:
        feedback = "Good start. Review any questions you found difficult."
    else:
        feedback = "Use this as a learning opportunity and revise this topic."

    return {
        "topic": attempt["topic"],
        "score": score,
        "total_questions": len(question_ids),
        "percentage": percentage,
        "feedback": feedback,
        "purpose": "Personal practice only; not used for placement prediction."
    }


@app.get("/practice/history/{student_id}")
def practice_history(student_id: str):
    with connect_db() as db:
        rows = db.execute(
            """
            SELECT topic, score, started_at
            FROM practice_attempts
            WHERE student_id = ? AND finished = 1
            ORDER BY started_at DESC
            """,
            (student_id.strip(),)
        ).fetchall()

    return {
        "history": [
            {
                "topic": row["topic"],
                "score": row["score"],
                "total": PRACTICE_QUESTIONS_PER_TEST,
                "percentage": round(
                    row["score"] / PRACTICE_QUESTIONS_PER_TEST * 100
                ),
                "started_at": row["started_at"]
            }
            for row in rows
        ]
    }


# ==================================================
# 15. JOB RECOMMENDATIONS (SEPARATE FROM PLACEMENT PREDICTION)
# ==================================================

STUDENT_PROFILES_PATH = BASE_DIR / "student_profiles.csv"
JOB_DATASET_PATH = BASE_DIR / "job_dataset.csv"
RECOMMENDATION_FEEDBACK_PATH = BASE_DIR / "recommendation_feedback.csv"


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{label} file was not found: {path.name}. Put it in {BASE_DIR}."
        )
    try:
        return pd.read_csv(path).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {label}: {exc}")


def _split_values(value):
    """Turn semicolon-separated CSV values into normalized strings."""
    if value is None:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _norm(value):
    return " ".join(str(value or "").strip().casefold().split())


def _degree_match(student_degree, required_degree):
    required = _norm(required_degree)
    degree = _norm(student_degree)
    if not required or required in {"any", "any graduate", "any degree", "not specified", "nan"}:
        return True
    if not degree:
        return False
    if required in degree or degree in required:
        return True
    # Broad, explainable degree-family matching; this is a heuristic, not
    # a formal eligibility determination.
    families = {
        "b.tech": ("b.tech", "bachelor", "engineering", "b.e"),
        "b.e": ("b.e", "b.tech", "bachelor", "engineering"),
        "b.sc": ("b.sc", "bachelor of science", "bachelor"),
        "bca": ("bca", "bachelor", "computer application"),
        "m.tech": ("m.tech", "master", "engineering", "m.e"),
        "m.sc": ("m.sc", "master of science", "master"),
        "mca": ("mca", "master", "computer application"),
    }
    for key, terms in families.items():
        if key in degree and any(term in required for term in terms):
            return True
    return False


def _score_job(student, job):
    student_skills = {_norm(x) for x in _split_values(student.get("technical_skills", ""))}
    student_skills |= {_norm(x) for x in _split_values(student.get("programming_languages", ""))}
    student_skills |= {_norm(x) for x in _split_values(student.get("project_technologies", ""))}
    required = {_norm(x) for x in _split_values(job.get("required_skills", ""))}
    preferred = {_norm(x) for x in _split_values(job.get("preferred_skills", ""))}
    required.discard("")
    preferred.discard("")

    required_matches = sorted(required & student_skills)
    preferred_matches = sorted(preferred & student_skills)
    required_score = 45 * (len(required_matches) / len(required) if required else 1)
    preferred_score = 15 * (len(preferred_matches) / len(preferred) if preferred else 0)

    roles = {_norm(x) for x in _split_values(student.get("preferred_roles", ""))}
    job_title = _norm(job.get("job_title", ""))
    role_match = any(role in job_title or job_title in role for role in roles if role and job_title)
    role_score = 15 if role_match else 0

    locations = {_norm(x) for x in _split_values(student.get("preferred_locations", ""))}
    job_location = _norm(job.get("location", ""))
    location_match = bool(job_location and (job_location in locations or "anywhere" in locations or "remote" in locations and _norm(job.get("work_mode")) == "remote"))
    location_score = 10 if location_match else 0

    preferred_mode = _norm(student.get("preferred_work_mode", ""))
    job_mode = _norm(job.get("work_mode", ""))
    mode_match = bool(preferred_mode and job_mode and preferred_mode == job_mode)
    mode_score = 5 if mode_match else 0

    try:
        experience = max(0, int(float(student.get("work_experience_months") or 0))) + max(0, int(float(student.get("internship_months") or 0)))
    except (TypeError, ValueError):
        experience = 0
    try:
        min_experience = max(0, int(float(job.get("min_experience_months") or 0)))
    except (TypeError, ValueError):
        min_experience = 0
    experience_match = experience >= min_experience
    experience_score = 5 if experience_match else 0

    degree_match = _degree_match(student.get("degree", ""), job.get("required_degree", ""))
    degree_score = 5 if degree_match else 0

    # A small penalty discourages presenting a role as a strong match when
    # it explicitly asks for more experience or a degree family mismatch.
    score = required_score + preferred_score + role_score + location_score + mode_score + experience_score + degree_score
    reasons = []
    if required_matches:
        reasons.append("Matching required skills: " + ", ".join(required_matches))
    if preferred_matches:
        reasons.append("Matching preferred skills: " + ", ".join(preferred_matches))
    if role_match:
        reasons.append("Matches a preferred role")
    if location_match:
        reasons.append("Matches a preferred location or remote preference")
    if mode_match:
        reasons.append("Matches preferred work mode")
    if experience_match:
        reasons.append("Experience meets the listed minimum")
    else:
        reasons.append("May require more experience than listed in the student profile")
    if degree_match:
        reasons.append("Degree appears compatible with the listing")
    else:
        reasons.append("Degree requirement may not match; verify eligibility")

    return round(max(0, min(100, score)), 2), reasons, required_matches, preferred_matches


class RecommendationDecision(BaseModel):
    student_id: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=80)
    admin_approved: bool | None = None
    student_feedback: str | None = Field(default=None, max_length=2000)
    application_status: str | None = Field(default=None, max_length=100)
    outcome: str | None = Field(default=None, max_length=100)


@app.get("/recommendations/students")
def recommendation_students(request: Request, limit: int = 100, offset: int = 0):
    """List student profile IDs and basic fields for the recommendation workflow."""
    get_current_user(request)
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(400, "limit must be 1-500 and offset must be non-negative.")
    students = _load_csv(STUDENT_PROFILES_PATH, "Student profiles")
    if "student_id" not in students.columns:
        raise HTTPException(500, "student_profiles.csv must contain student_id.")
    fields = [c for c in ["student_id", "degree", "specialization", "preferred_roles", "preferred_locations"] if c in students.columns]
    records = students[fields].iloc[offset:offset + limit].to_dict(orient="records")
    return {"total_students": int(len(students)), "limit": limit, "offset": offset, "students": records}


@app.get("/recommendations/{student_id}")
def recommend_jobs(student_id: str, request: Request, limit: int = 10):
    """Return job matches for a CSV student profile using the trained XGBoost model."""
    get_current_user(request)
    if limit < 1 or limit > 50:
        raise HTTPException(400, "limit must be between 1 and 50.")
    students = _load_csv(STUDENT_PROFILES_PATH, "Student profiles")
    jobs = _load_csv(JOB_DATASET_PATH, "Job dataset")
    if "student_id" not in students.columns or "job_id" not in jobs.columns:
        raise HTTPException(500, "CSV files must contain student_id and job_id columns.")
    matched = students[students["student_id"].astype(str).str.strip() == student_id.strip()]
    if matched.empty:
        raise HTTPException(404, f"Student profile '{student_id}' was not found.")
    student = matched.iloc[0].to_dict()

    if ml_engine.is_loaded:
        recommendations = ml_engine.predict_jobs(student, jobs, top_n=limit)
        method_desc = "Trained multi-head XGBoost model (xgboost_skillstudio.pt) with AUC: 0.995; incorporates soft-skill credit and offer likelihood."
    else:
        recommendations = []
        for _, row in jobs.iterrows():
            job = row.to_dict()
            score, reasons, required_matches, preferred_matches = _score_job(student, job)
            recommendations.append({
                "job_id": str(job.get("job_id", "")),
                "job_title": str(job.get("job_title", "")),
                "company": str(job.get("company", "")),
                "location": str(job.get("location", "")),
                "work_mode": str(job.get("work_mode", "")),
                "job_type": str(job.get("job_type", "")),
                "required_skills": _split_values(job.get("required_skills", "")),
                "preferred_skills": _split_values(job.get("preferred_skills", "")),
                "match_score": score,
                "matching_required_skills": required_matches,
                "matching_preferred_skills": preferred_matches,
                "explanation": reasons,
                "job_description": str(job.get("job_description", ""))
            })
        recommendations.sort(key=lambda item: item["match_score"], reverse=True)
        recommendations = recommendations[:limit]
        method_desc = "Explainable rule-based baseline."

    # Attach recorded review/feedback where available.
    with connect_db() as db:
        saved = db.execute("SELECT job_id, admin_approved, student_feedback, application_status, outcome FROM recommendation_decisions WHERE student_id = ?", (student_id,)).fetchall()
    saved_by_job = {r["job_id"]: dict(r) for r in saved}
    for item in recommendations:
        if item["job_id"] in saved_by_job:
            item["review"] = saved_by_job[item["job_id"]]
    return {
        "student_id": student_id,
        "method": method_desc,
        "total_jobs_considered": len(jobs),
        "recommendations": recommendations
    }


@app.post("/recommendations/decision")
def save_recommendation_decision(data: RecommendationDecision, request: Request):
    """Save a review/feedback record. Current app has no separate admin-role system."""
    get_current_user(request)
    students = _load_csv(STUDENT_PROFILES_PATH, "Student profiles")
    jobs = _load_csv(JOB_DATASET_PATH, "Job dataset")
    if data.student_id not in set(students.get("student_id", pd.Series(dtype=str)).astype(str)):
        raise HTTPException(404, "Student ID was not found in student_profiles.csv.")
    if data.job_id not in set(jobs.get("job_id", pd.Series(dtype=str)).astype(str)):
        raise HTTPException(404, "Job ID was not found in job_dataset.csv.")
    with connect_db() as db:
        db.execute(
            """INSERT INTO recommendation_decisions
               (student_id, job_id, admin_approved, student_feedback, application_status, outcome, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(student_id, job_id) DO UPDATE SET
                 admin_approved=excluded.admin_approved,
                 student_feedback=excluded.student_feedback,
                 application_status=excluded.application_status,
                 outcome=excluded.outcome,
                 updated_at=CURRENT_TIMESTAMP""",
            (data.student_id, data.job_id, None if data.admin_approved is None else int(data.admin_approved), data.student_feedback, data.application_status, data.outcome)
        )
    return {"saved": True, "student_id": data.student_id, "job_id": data.job_id, "message": "Recommendation review/feedback saved."}


@app.get("/recommendations/summary/feedback")
def recommendation_feedback_summary(request: Request):
    """Summarize saved feedback, including the optional supplied CSV history."""
    get_current_user(request)
    with connect_db() as db:
        rows = db.execute("SELECT admin_approved, student_feedback, application_status, outcome FROM recommendation_decisions").fetchall()
    db_records = [dict(r) for r in rows]
    summary = {"saved_database_records": len(db_records), "admin_approved_yes": sum(r["admin_approved"] == 1 for r in db_records), "admin_approved_no": sum(r["admin_approved"] == 0 for r in db_records), "with_student_feedback": sum(bool(r["student_feedback"]) for r in db_records), "application_status_counts": {}, "outcome_counts": {}}
    for r in db_records:
        if r["application_status"]:
            key = str(r["application_status"])
            summary["application_status_counts"][key] = summary["application_status_counts"].get(key, 0) + 1
        if r["outcome"]:
            key = str(r["outcome"])
            summary["outcome_counts"][key] = summary["outcome_counts"].get(key, 0) + 1
    if RECOMMENDATION_FEEDBACK_PATH.exists():
        feedback = _load_csv(RECOMMENDATION_FEEDBACK_PATH, "Recommendation feedback")
        summary["source_csv_rows"] = int(len(feedback))
        for col in ["admin_approved", "student_feedback", "application_status", "outcome"]:
            if col in feedback.columns:
                summary[f"source_csv_{col}_nonempty"] = int(feedback[col].astype(str).str.strip().replace("nan", "").ne("").sum())
    else:
        summary["source_csv_rows"] = 0
        summary["source_csv_note"] = "recommendation_feedback.csv was not found; database feedback is still available."
    return summary
