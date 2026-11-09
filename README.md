# SkillStudio - Student Assessment & Placement Readiness Platform

SkillStudio is an end-to-end web platform designed to assess student technical competencies, analyze placement readiness, recommend relevant job roles, and provide targeted practice assessments.

---

## 🌟 Key Features

- **Placement Readiness Analysis**: Comprehensive readiness scoring based on student profile attributes, academic performance, project experience, and technical skill assessments.
- **Adaptive Question Bank & Assessments**: Timed interactive assessments with topic-specific questions and automatic score calculation.
- **Practice Arena**: Topic-wise practice modules allowing students to prepare across various core subjects.
- **Job Matching & Recommendations**: Intelligent matching between candidate profiles and job datasets (`job_dataset.csv`).
- **Interactive Dashboards**: Visual analytics, attempt histories, and performance tracking.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/), Starlette
- **Database & Data**: SQLite, [Pandas](https://pandas.pydata.org/)
- **Frontend**: HTML5, modern CSS, Vanilla JavaScript (REST API integration)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn backi2:app --reload --port 8000
```

Once running, access the web pages in your browser:
- **Login / Register**: `http://127.0.0.1:8000/login.html`
- **Dashboard**: `http://127.0.0.1:8000/dashboard.html`
- **Assessment**: `http://127.0.0.1:8000/index.html`
- **Placement Analysis**: `http://127.0.0.1:8000/placement.html`
- **Practice Arena**: `http://127.0.0.1:8000/practice.html`
- **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`

---

## 📁 Project Structure

```text
├── backi2.py                    # Main FastAPI backend application
├── question_bank.py             # Assessment question bank data
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore specifications
├── index.html                   # Assessment interface
├── dashboard.html               # Student analytics dashboard
├── login.html                   # Authentication page
├── placement.html               # Placement prediction & readiness page
├── practice.html                # Topic practice arena
├── job_dataset.csv              # Job requirements dataset
├── student_profiles.csv         # Candidate profiles dataset
└── recommendation_feedback.csv  # Recommendation feedback data
```

---

## 👤 Author

- **Snehanjali Sahu** ([sneha03anjali@gmail.com](mailto:sneha03anjali@gmail.com))
-
-
- **Amisha Kalita** ([amishakalita380@gmail.com](mailto:amishakalita380@gmail.com))
