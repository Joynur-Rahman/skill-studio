
from pathlib import Path
import os
import random
import sqlite3
import uuid
import hashlib
import hmac
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from question_bank import QUESTIONS


# ==================================================
# 1. FILE LOCATIONS AND APP SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path.home() / "Downloads" / "placement_model.joblib"
DB_PATH = BASE_DIR / "assessment_history.db"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Placement model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model classes:", model.classes_)
print("Model features:", model.feature_names_in_)

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

# Keep same-origin frontend requests working.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
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

        # Separate storage for topic practice attempts. Existing tables/data stay intact.
        db.execute("""
            CREATE TABLE IF NOT EXISTS practice_attempts (
                attempt_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                question_ids TEXT NOT NULL,
                answers TEXT DEFAULT '{}',
                score INTEGER DEFAULT 0,
                finished INTEGER DEFAULT 0,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Decisions made while reviewing job recommendations.
        db.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                approved INTEGER NOT NULL,
                note TEXT DEFAULT '',
                decided_by INTEGER,
                decided_at TEXT DEFAULT CURRENT_TIMESTAMP,
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
    user = get_current_user(request)
    student_id = str(user["id"])

    with connect_db() as db:
        attempt = db.execute(
            """
            SELECT *
            FROM attempts
            WHERE attempt_id = ? AND student_id = ?
            """,
            (attempt_id, student_id)
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
        "pages": {
            "login": "/login.html",
            "dashboard": "/dashboard.html",
            "assessment": "/index.html",
            "placement": "/placement.html",
            "api_docs": "/docs"
        }
    }


@app.get("/login.html")
def login_page():
    file_path = BASE_DIR / "login.html"

    if not file_path.exists():
        raise HTTPException(404, "login.html was not found.")

    return FileResponse(file_path)


@app.get("/dashboard.html")
def dashboard_page(request: Request):
    get_current_user(request)
    file_path = BASE_DIR / "dashboard.html"

    if not file_path.exists():
        raise HTTPException(404, "dashboard.html was not found.")

    return FileResponse(file_path)


@app.get("/index.html")
def assessment_page(request: Request):
    get_current_user(request)
    file_path = BASE_DIR / "index.html"

    if not file_path.exists():
        raise HTTPException(404, "index.html was not found.")

    return FileResponse(file_path)


@app.get("/placement.html")
def placement_page(request: Request):
    get_current_user(request)
    file_path = BASE_DIR / "placement.html"

    if not file_path.exists():
        raise HTTPException(404, "placement.html was not found.")

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
# 6. PLACEMENT PREDICTION
# ==================================================

class StudentData(BaseModel):
    ssc_p: float
    hsc_p: float
    degree_p: float
    etest_p: float
    workex: int


@app.post("/predict")
def predict(data: StudentData, request: Request):
    # Only signed-in users may use the predictor.
    get_current_user(request)

    percentages = [
        data.ssc_p,
        data.hsc_p,
        data.degree_p,
        data.etest_p
    ]

    if not all(0 <= x <= 100 for x in percentages):
        raise HTTPException(
            status_code=400,
            detail="Percentages must be between 0 and 100."
        )

    if data.workex not in (0, 1):
        raise HTTPException(
            status_code=400,
            detail="workex must be 0 or 1."
        )

    student = pd.DataFrame([{
        "ssc_p": data.ssc_p,
        "hsc_p": data.hsc_p,
        "degree_p": data.degree_p,
        "etest_p": data.etest_p,
        "workex": data.workex
    }])

    prediction = int(model.predict(student)[0])
    probabilities = model.predict_proba(student)[0]

    # Find the probability corresponding to class 1,
    # rather than assuming it is always the second class.
    class_list = list(model.classes_)
    placed_index = class_list.index(1) if 1 in class_list else None

    placement_probability = (
        float(probabilities[placed_index]) * 100
        if placed_index is not None
        else 0.0
    )

    return {
        "prediction": "Placed" if prediction == 1 else "Not Placed",
        "placement_probability": round(placement_probability, 2)
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
    user = get_current_user(request)
    # Every attempt is associated with the authenticated account.
    student_id = str(user["id"])

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


PRACTICE_QUESTIONS_PER_TEST = 5
QUESTIONS_PER_TOPIC = 20


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
                "ready": len(questions) >= QUESTIONS_PER_TOPIC
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

    if len(topic_questions) < QUESTIONS_PER_TOPIC:
        raise HTTPException(
            400,
            f"{topic} has {len(topic_questions)} questions. "
            f"Add {QUESTIONS_PER_TOPIC - len(topic_questions)} more "
            f"to reach the required 20 questions per topic."
        )

    selected = random.sample(
        topic_questions,
        PRACTICE_QUESTIONS_PER_TEST
    )

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
                "Submit one answer for each of the five questions."
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
# JOB RECOMMENDATION SYSTEM
# Separate from the existing placement prediction model
# Put these CSV files in the same folder as backi2.py:
#   student_profiles.csv, job_dataset.csv, recommendation_feedback.csv
# ==================================================

STUDENT_PROFILES_PATH = BASE_DIR / "student_profiles.csv"
JOB_DATASET_PATH = BASE_DIR / "job_dataset.csv"
RECOMMENDATION_FEEDBACK_PATH = BASE_DIR / "recommendation_feedback.csv"


class RecommendationDecision(BaseModel):
    student_id: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=80)
    approved: bool
    note: str = Field(default="", max_length=1000)


def _load_recommendation_csv(path: Path, required_columns):
    """Load a recommendation CSV and validate its columns without modifying it."""
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Required dataset not found: {path.name}. Put it beside backi2.py."
        )
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {exc}"
        )
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"{path.name} is missing columns: {', '.join(missing)}"
        )
    return frame.fillna("")


def _normalise_text(value):
    """Normalize a semicolon-separated list or plain text for comparisons."""
    if value is None:
        return []
    return [part.strip().lower() for part in str(value).split(";") if part.strip()]


def _degree_matches(student_degree, required_degree):
    required = str(required_degree or "").strip().lower()
    degree = str(student_degree or "").strip().lower()
    if not required or required in {"any", "any degree", "any graduate", "not specified", "none", "nan"}:
        return True
    if not degree:
        return False
    # A simple transparent baseline; not a formal eligibility decision.
    if required in degree or degree in required:
        return True
    if "bachelor" in required and any(x in degree for x in ("b.tech", "btech", "b.e", "b.sc", "bca", "bachelor")):
        return True
    if "master" in required and any(x in degree for x in ("m.tech", "mtech", "m.e", "m.sc", "mca", "master")):
        return True
    return False


def _score_job(student, job):
    student_skills = set(_normalise_text(student.get("technical_skills", "")))
    student_skills.update(_normalise_text(student.get("programming_languages", "")))
    student_skills.update(_normalise_text(student.get("project_technologies", "")))
    job_required = set(_normalise_text(job.get("required_skills", "")))
    job_preferred = set(_normalise_text(job.get("preferred_skills", "")))

    required_hits = sorted(student_skills & job_required)
    missing_required = sorted(job_required - student_skills)
    preferred_hits = sorted(student_skills & job_preferred)

    required_score = 0.0 if not job_required else len(required_hits) / len(job_required) * 50.0
    preferred_score = 0.0 if not job_preferred else len(preferred_hits) / len(job_preferred) * 15.0

    preferred_roles = _normalise_text(student.get("preferred_roles", ""))
    title = str(job.get("job_title", "")).strip().lower()
    role_match = any(role in title or title in role for role in preferred_roles if role and title)
    role_score = 15.0 if role_match else 0.0

    try:
        experience = max(0, int(float(student.get("internship_months", 0) or 0))) + max(
            0, int(float(student.get("work_experience_months", 0) or 0))
        )
    except (TypeError, ValueError):
        experience = 0
    try:
        minimum = max(0, int(float(job.get("min_experience_months", 0) or 0)))
    except (TypeError, ValueError):
        minimum = 0
    experience_score = 10.0 if experience >= minimum else (10.0 * experience / minimum if minimum else 10.0)

    preferred_locations = _normalise_text(student.get("preferred_locations", ""))
    job_location = str(job.get("location", "")).strip().lower()
    location_match = (not preferred_locations or not job_location or
                      any(loc == job_location or loc in job_location or job_location in loc for loc in preferred_locations))
    preferred_mode = str(student.get("preferred_work_mode", "")).strip().lower()
    job_mode = str(job.get("work_mode", "")).strip().lower()
    mode_match = not preferred_mode or not job_mode or preferred_mode == job_mode or "hybrid" in {preferred_mode, job_mode}
    preferred_type = str(student.get("job_type_preference", "")).strip().lower()
    job_type = str(job.get("job_type", "")).strip().lower()
    type_match = not preferred_type or not job_type or preferred_type == job_type
    preference_score = 10.0 * sum([location_match, mode_match, type_match]) / 3.0

    degree_match = _degree_matches(student.get("degree", ""), job.get("required_degree", ""))
    score = required_score + preferred_score + role_score + experience_score + preference_score
    if not degree_match:
        # Degree mismatch is a penalty, not an automatic exclusion, since dataset values can be noisy.
        score *= 0.75
    score = round(max(0.0, min(100.0, score)), 1)

    reasons = []
    if required_hits:
        reasons.append("Required skills matched: " + ", ".join(required_hits))
    if preferred_hits:
        reasons.append("Preferred skills matched: " + ", ".join(preferred_hits))
    if role_match:
        reasons.append("Matches a preferred role")
    if experience >= minimum:
        reasons.append("Meets the listed minimum experience")
    elif minimum:
        reasons.append(f"Experience gap: {minimum - experience} month(s)")
    if location_match and preferred_locations:
        reasons.append("Location matches a stated preference")
    if mode_match and preferred_mode:
        reasons.append("Work mode matches a stated preference")
    if type_match and preferred_type:
        reasons.append("Job type matches a stated preference")
    if not degree_match:
        reasons.append("Degree requirement may not match; verify manually")
    if missing_required:
        reasons.append("Required skills not found in profile: " + ", ".join(missing_required))
    if not reasons:
        reasons.append("Limited profile information matched this posting")

    return {
        "match_score": score,
        "matched_required_skills": required_hits,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": preferred_hits,
        "degree_match": degree_match,
        "reasons": reasons
    }


@app.get("/recommendations/students")
def recommendation_students(request: Request, search: str = "", limit: int = 100):
    """List student profiles for the recommendation review page. Login required."""
    get_current_user(request)
    students = _load_recommendation_csv(
        STUDENT_PROFILES_PATH,
        ["student_id", "degree", "specialization", "technical_skills", "preferred_roles"]
    )
    query = search.strip().lower()
    if query:
        mask = students.apply(
            lambda row: query in str(row.get("student_id", "")).lower()
            or query in str(row.get("specialization", "")).lower()
            or query in str(row.get("preferred_roles", "")).lower(), axis=1
        )
        students = students[mask]
    limit = max(1, min(int(limit), 500))
    return {
        "total": int(len(students)),
        "students": students.head(limit)[[
            column for column in ["student_id", "degree", "specialization", "graduation_year", "preferred_roles"]
            if column in students.columns
        ]].to_dict(orient="records")
    }


@app.get("/recommendations/{student_id}")
def get_job_recommendations(student_id: str, request: Request, limit: int = 10):
    """Rank jobs for one student using explainable skill/profile matching."""
    get_current_user(request)
    student_id = student_id.strip()
    students = _load_recommendation_csv(STUDENT_PROFILES_PATH, ["student_id"])
    jobs = _load_recommendation_csv(JOB_DATASET_PATH, [
        "job_id", "job_title", "company", "required_skills", "preferred_skills",
        "min_experience_months", "max_experience_months", "required_degree",
        "location", "work_mode", "job_type", "job_description"
    ])
    found = students[students["student_id"].astype(str).str.strip() == student_id]
    if found.empty:
        raise HTTPException(status_code=404, detail=f"Student {student_id} was not found.")
    student = found.iloc[0].to_dict()
    limit = max(1, min(int(limit), 100))

    with connect_db() as db:
        decisions = db.execute(
            "SELECT job_id, approved, note, decided_at FROM recommendation_decisions WHERE student_id = ?",
            (student_id,)
        ).fetchall()
    decision_by_job = {row["job_id"]: row for row in decisions}

    results = []
    for _, row in jobs.iterrows():
        job = row.to_dict()
        scoring = _score_job(student, job)
        decision = decision_by_job.get(str(job.get("job_id", "")))
        results.append({
            "job_id": str(job.get("job_id", "")),
            "job_title": str(job.get("job_title", "")),
            "company": str(job.get("company", "")),
            "required_skills": str(job.get("required_skills", "")),
            "preferred_skills": str(job.get("preferred_skills", "")),
            "min_experience_months": job.get("min_experience_months", ""),
            "required_degree": str(job.get("required_degree", "")),
            "location": str(job.get("location", "")),
            "work_mode": str(job.get("work_mode", "")),
            "job_type": str(job.get("job_type", "")),
            "job_description": str(job.get("job_description", "")),
            **scoring,
            "admin_approved": (None if decision is None else bool(decision["approved"])),
            "admin_note": ("" if decision is None else decision["note"]),
            "decision_at": (None if decision is None else decision["decided_at"])
        })
    results.sort(key=lambda item: (item["match_score"], len(item["matched_required_skills"])), reverse=True)
    return {"student_id": student_id, "student": student, "total_jobs": len(results), "recommendations": results[:limit]}


@app.post("/recommendations/decision")
def save_recommendation_decision(data: RecommendationDecision, request: Request):
    """Save an approval/rejection decision. Login required; decisions are separate from feedback CSV."""
    user = get_current_user(request)
    student_id = data.student_id.strip()
    job_id = data.job_id.strip()
    students = _load_recommendation_csv(STUDENT_PROFILES_PATH, ["student_id"])
    jobs = _load_recommendation_csv(JOB_DATASET_PATH, ["job_id"])
    if not (students["student_id"].astype(str).str.strip() == student_id).any():
        raise HTTPException(status_code=404, detail=f"Student {student_id} was not found.")
    if not (jobs["job_id"].astype(str).str.strip() == job_id).any():
        raise HTTPException(status_code=404, detail=f"Job {job_id} was not found.")
    with connect_db() as db:
        db.execute("""
            INSERT INTO recommendation_decisions (student_id, job_id, approved, note, decided_by, decided_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(student_id, job_id) DO UPDATE SET
                approved = excluded.approved,
                note = excluded.note,
                decided_by = excluded.decided_by,
                decided_at = CURRENT_TIMESTAMP
        """, (student_id, job_id, int(data.approved), data.note.strip(), user["id"]))
    return {
        "message": "Recommendation approved." if data.approved else "Recommendation rejected.",
        "student_id": student_id,
        "job_id": job_id,
        "approved": data.approved
    }


@app.get("/recommendations/summary/feedback")
def recommendation_feedback_summary(request: Request):
    """Summarize available feedback; blank labels are counted as unknown, not negative."""
    get_current_user(request)
    feedback = _load_recommendation_csv(RECOMMENDATION_FEEDBACK_PATH, [
        "student_id", "job_id", "admin_approved", "student_feedback", "application_status", "outcome"
    ])
    def count_values(column):
        values = feedback[column].astype(str).str.strip()
        values = values[values != ""]
        return {str(k): int(v) for k, v in values.value_counts().items()}
    return {
        "total_feedback_rows": int(len(feedback)),
        "admin_approval_counts": count_values("admin_approved"),
        "student_feedback_counts": count_values("student_feedback"),
        "application_status_counts": count_values("application_status"),
        "outcome_counts": count_values("outcome"),
        "note": "Blank feedback fields are unknown and are not treated as negative labels."
    }

