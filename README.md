# SkillStudio - Student Assessment & Placement Readiness Platform

SkillStudio is an end-to-end web platform designed to assess student technical competencies, analyze placement readiness, recommend relevant job roles, and provide targeted practice assessments. Developed as part of an advanced academic project at IIIT Guwahati.

---

## 🌟 Key Features

- **Placement Readiness Analysis**: Comprehensive readiness scoring backed by trained machine learning models (`scikit-learn`).
- **Adaptive Question Bank & Assessments**: Timed interactive assessments with topic-specific questions and automatic score calculation.
- **Practice Arena**: Topic-wise practice modules allowing students to prepare across various core subjects.
- **Job Matching & Recommendations**: Intelligent semantic matching between candidate profiles and job datasets (`job_dataset.csv`).
- **Interactive Dashboards**: Visual analytics, attempt histories, and performance tracking.

---

## 🤖 Machine Learning & Intelligence

- **Core Algorithm**: Gradient Boosting Classifier optimized with hyperparameter tuning.
- **Performance**: Achieves a verified **89.08% classification accuracy** on student placement records.
- **Serialization**: Models are stored and loaded dynamically via `joblib` / `.pkl` serialization artifacts.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), NumPy
- **Database**: SQLite
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

## 👤 Authors & Contributors

- **Snehanjali Sahu** ([sneha03anjali@gmail.com](mailto:sneha03anjali@gmail.com))
- **Shramana Mukherjee** ([shrawmana@gmail.com](mailto:shrawmana@gmail.com))
- **Pratiksha Sarma** ([pratikshasarma40@gmail.com](mailto:pratikshasarma40@gmail.com))
- **Amisha Kalita** ([amishakalita380@gmail.com](mailto:amishakalita380@gmail.com))

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Acknowledgments & Institution

- **Department of Computer Science & Engineering**, Indian Institute of Information Technology Guwahati (IIIT Guwahati).
- Special thanks to our academic supervisor Mr. Nilotpal Chakraborty for their guidance during the design, development, and evaluation of **SkillStudio**.
