import urllib.request
import urllib.parse
import json
import sys
import http.cookiejar

BASE = "http://127.0.0.1:8080"
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with opener.open(req) as resp:
        return resp.getcode(), resp.read()

def post(path, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(req) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

results = []

print("Running comprehensive SkillStudio health check...\n")

# 1. API Root & ML Model Info
try:
    code, raw = get("/")
    data = json.loads(raw.decode("utf-8"))
    results.append(("GET /", code == 200, f"Status {code} - {data.get('message')}"))
except Exception as e:
    results.append(("GET /", False, str(e)))

try:
    code, raw = get("/model/info")
    info = json.loads(raw.decode("utf-8"))
    feat_len = len(info.get("features", []))
    boosters = info.get("boosters", [])
    ok = info.get("loaded") is True and feat_len == 28 and len(boosters) == 3
    results.append(("GET /model/info (XGBoost)", ok, f"Loaded: {info.get('loaded')}, Boosters: {boosters}, Features: {feat_len}"))
except Exception as e:
    results.append(("GET /model/info (XGBoost)", False, str(e)))

# 2. Static HTML Pages
for page in ["login.html", "dashboard.html", "placement.html", "practice.html", "index.html"]:
    try:
        code, raw = get(f"/{page}")
        ok = code == 200 and len(raw) > 500
        results.append((f"GET /{page}", ok, f"Status {code}, Size: {len(raw)} bytes"))
    except Exception as e:
        results.append((f"GET /{page}", False, str(e)))

# 3. Authentication
try:
    code, resp = post("/auth/login", {"email": "student@skillstudio.com", "password": "password123"})
    ok = code == 200 and "user" in resp
    results.append(("POST /auth/login", ok, f"Logged in as: {resp.get('user', {}).get('name')}"))
except Exception as e:
    results.append(("POST /auth/login", False, str(e)))

try:
    code, raw = get("/auth/me")
    me = json.loads(raw.decode("utf-8"))
    ok = code == 200 and me.get("email") == "student@skillstudio.com"
    results.append(("GET /auth/me", ok, f"Authenticated session: {me.get('email')} (ID: {me.get('id')})"))
except Exception as e:
    results.append(("GET /auth/me", False, str(e)))

# 4. Assessment Workflow
try:
    code, raw = get("/assessment/bank-info")
    bank = json.loads(raw.decode("utf-8"))
    q_count = bank.get("total_questions", 0)
    results.append(("GET /assessment/bank-info", q_count > 0, f"Available questions in bank: {q_count}"))
except Exception as e:
    results.append(("GET /assessment/bank-info", False, str(e)))

try:
    code, start_resp = post("/assessment/start", {})
    attempt_id = start_resp.get("attempt_id")
    questions = start_resp.get("questions", [])
    ok = code == 200 and len(questions) == 10
    results.append(("POST /assessment/start", ok, f"Attempt {attempt_id} generated {len(questions)} timed questions"))

    if questions:
        post("/assessment/answer", {
            "attempt_id": attempt_id,
            "question_id": questions[0]["id"],
            "selected_answer": 0
        })

    code, finish_resp = post(f"/assessment/finish/{attempt_id}", {})
    score_ok = code == 200 and "score" in finish_resp
    results.append(("POST /assessment/finish", score_ok, f"Score: {finish_resp.get('score')}/{finish_resp.get('total_questions')} ({finish_resp.get('percentage')}%)"))
except Exception as e:
    results.append(("Assessment Workflow", False, str(e)))

# 5. Practice Arena Workflow
try:
    code, raw = get("/practice/topics")
    topic_objs = json.loads(raw.decode("utf-8")).get("topics", [])
    topics = [t["name"] for t in topic_objs]
    results.append(("GET /practice/topics", len(topics) > 0, f"Practice topics ({len(topics)}): {', '.join(topics[:4])}..."))

    if topics:
        code, pstart = post("/practice/start", {"student_id": "1", "topic": topics[0]})
        p_id = pstart.get("attempt_id")
        p_qs = pstart.get("questions", [])
        p_answers = {q["id"]: 0 for q in p_qs}
        code, pfinish = post("/practice/finish", {"attempt_id": p_id, "answers": p_answers})
        ok = code == 200 and "score" in pfinish
        results.append(("Practice Quiz (/start & /finish)", ok, f"Topic '{topics[0]}': {pfinish.get('score')}/{pfinish.get('total_questions')} correct"))
except Exception as e:
    results.append(("Practice Arena Workflow", False, str(e)))

# 6. Career Readiness & XGBoost ML Model
try:
    profile = {
        "degree": "B.Tech",
        "specialization": "Computer Science",
        "graduation_year": 2026,
        "technical_skills": "Python; SQL; Docker; AWS; REST API; Machine Learning",
        "programming_languages": "Python; JavaScript",
        "skill_proficiency": "Python:Advanced; SQL:Intermediate; Docker:Intermediate",
        "projects_count": 3,
        "project_technologies": "Python; FastAPI; PostgreSQL; Docker",
        "internship_months": 6,
        "work_experience_months": 0,
        "certifications": "AWS Certified Cloud Practitioner",
        "preferred_roles": "Backend Developer; Data Scientist",
        "preferred_locations": "Remote; Bangalore",
        "preferred_work_mode": "Remote",
        "job_type_preference": "Full-time"
    }
    code, pred = post("/profile/analyze", profile)
    readiness = pred.get("readiness", {})
    recs = pred.get("recommendations", [])
    ok = code == 200 and len(recs) == 10 and readiness.get("readiness_score", 0) > 0
    top = recs[0] if recs else {}
    results.append(("POST /profile/analyze (XGBoost ML)", ok, f"Readiness Score: {readiness.get('readiness_score')}/100, Top Match: {top.get('job_title')} ({top.get('match_score')}%, Offer: {top.get('offer_likelihood')}%)"))
except Exception as e:
    results.append(("POST /profile/analyze (XGBoost ML)", False, str(e)))

# 7. Recommendations via CSV Profile
try:
    code, raw = get("/recommendations/STU0001?limit=5")
    data = json.loads(raw.decode("utf-8"))
    recs = data.get("recommendations", [])
    ok = code == 200 and len(recs) == 5
    top = recs[0] if recs else {}
    results.append(("GET /recommendations/{student_id}", ok, f"Student STU0001 -> Top Match: {top.get('job_title')} ({top.get('match_score')}%)"))
except Exception as e:
    results.append(("GET /recommendations/{student_id}", False, str(e)))

print("=" * 80)
print(f"{'ENDPOINT / FEATURE':<36} {'STATUS':<12} {'DETAILS'}")
print("=" * 80)
all_ok = True
for name, passed, detail in results:
    status_str = "[PASS]" if passed else "[FAIL]"
    if not passed:
        all_ok = False
    print(f"{name:<36} {status_str:<12} {detail}")
print("=" * 80)
if all_ok:
    print("\n>>> ALL TESTS PASSED! Everything in SkillStudio is working perfectly. <<<")
else:
    print("\n>>> Some tests failed. Please review the output above. <<<")
