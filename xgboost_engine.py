"""
XGBoost ML Inference Engine for SkillStudio
Loads and executes the trained multi-head XGBoost model from xgboost_skillstudio.pt.
"""

from pathlib import Path
import io
import math
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent

PROF_MAP = {
    "beginner": 0.4,
    "intermediate": 0.7,
    "advanced": 1.0,
    "expert": 1.0
}
ROLE_NOISE = {
    "junior", "senior", "intern", "lead", "associate",
    "trainee", "graduate", "entry", "level", "internship"
}


def _norm(val: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(val or "").lower())


def _split_values(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    text = str(val).strip()
    if not text:
        return []
    parts = re.split(r"[;,|\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _core_role(title: Any) -> str:
    tokens = [
        w for w in re.findall(r"[a-z0-9+#./]+", str(title or "").lower())
        if w not in ROLE_NOISE
    ]
    return " ".join(tokens)


def _level_index(title: Any) -> int:
    w = str(title or "").lower().split()
    if "intern" in w:
        return 0
    if "junior" in w:
        return 1
    if "senior" in w or "lead" in w:
        return 3
    return 2


def _parse_proficiency(text: Any) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for part in _split_values(text):
        if ":" in part:
            name, lvl = part.rsplit(":", 1)
            result[_norm(name)] = PROF_MAP.get(lvl.strip().lower(), 0.5)
        elif "-" in part:
            name, lvl = part.rsplit("-", 1)
            result[_norm(name)] = PROF_MAP.get(lvl.strip().lower(), 0.5)
        else:
            result[_norm(part)] = 0.5
    return result


class DummyTensor:
    def __init__(self, *args, **kwargs):
        self.args = args


class DummyStorage(list):
    pass


class SafeTorchUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return pid

    def find_class(self, module, name):
        if module == "torch" and "Storage" in name:
            return DummyStorage
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return DummyTensor
        return super().find_class(module, name)


class SkillStudioXGBoostEngine:
    def __init__(self, model_path: Optional[Path] = None):
        self.is_loaded = False
        self.model_path = model_path or self._find_model_path()
        self.skills: List[str] = []
        self.sk_idx: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.S_rel: Optional[np.ndarray] = None
        self.rel_gate_scalar: float = 0.0
        self.boosters: Dict[str, xgb.Booster] = {}
        self.metrics: Dict[str, Any] = {}

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _find_model_path(self) -> Optional[Path]:
        candidates = [
            BASE_DIR / "xgboost_skillstudio.pt",
            Path.home() / "Downloads" / "xgboost_skillstudio.pt",
            BASE_DIR / "skillstudio_model.pt",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def _load_model(self):
        try:
            import zipfile

            with zipfile.ZipFile(self.model_path) as z:
                # Find prefix inside archive (usually 'xgboost_skillstudio' or 'skillstudio_model' or root)
                names = z.namelist()
                prefix = ""
                for n in names:
                    if n.endswith("data.pkl"):
                        prefix = n[:-len("data.pkl")]
                        break

                raw_meta = z.read(f"{prefix}data.pkl")
                meta = SafeTorchUnpickler(io.BytesIO(raw_meta)).load()

                self.feature_names = meta.get("features", [])
                self.metrics = meta.get("metrics", {})
                fz = meta.get("featurizer", {})
                self.skills = [s.lower() for s in fz.get("skills", [])]
                self.sk_idx = {s: i for i, s in enumerate(self.skills)}

                K = len(self.skills)
                if K > 0 and f"{prefix}data/10" in z.namelist():
                    b10 = z.read(f"{prefix}data/10")
                    self.S_rel = np.frombuffer(b10, dtype=np.float32).reshape(K, K)
                else:
                    self.S_rel = np.eye(K, dtype=np.float32)

                if f"{prefix}data/11" in z.namelist():
                    b11 = z.read(f"{prefix}data/11")
                    raw_scalar = np.frombuffer(b11, dtype=np.float32)[0]
                    self.rel_gate_scalar = 1.0 / (1.0 + math.exp(-raw_scalar))
                else:
                    self.rel_gate_scalar = 0.5

                # Boosters
                for name, storage_id in [("somewhat", "12"), ("relevant", "13"), ("offer", "14")]:
                    target_file = f"{prefix}data/{storage_id}"
                    if target_file in z.namelist():
                        bst = xgb.Booster()
                        bst.load_model(bytearray(z.read(target_file)))
                        self.boosters[name] = bst

            if "relevant" in self.boosters:
                self.is_loaded = True
                print(f"[XGBoostEngine] Successfully loaded {self.model_path.name} with {len(self.boosters)} boosters.")
        except Exception as e:
            print(f"[XGBoostEngine] Could not load model from {self.model_path}: {e}")
            self.is_loaded = False

    def predict_jobs(self, profile: dict, jobs_df: pd.DataFrame, top_n: int = 10) -> List[dict]:
        """Vectorized job matching using the XGBoost boosters."""
        if not self.is_loaded or jobs_df.empty:
            return []

        N = len(jobs_df)
        K = len(self.skills)

        # 1. Encode student skills & proficiencies
        prof_dict = _parse_proficiency(profile.get("skill_proficiency", ""))
        all_student_skills = (
            _split_values(profile.get("technical_skills", ""))
            + _split_values(profile.get("programming_languages", ""))
            + _split_values(profile.get("project_technologies", ""))
        )
        s_norm_set = {_norm(s) for s in all_student_skills if _norm(s)}

        s_strength = np.zeros(K, dtype=np.float32)
        for s in all_student_skills:
            ns = _norm(s)
            if ns in self.sk_idx:
                s_strength[self.sk_idx[ns]] = prof_dict.get(ns, 0.7)
            else:
                # partial matching against vocab
                for sk, idx in self.sk_idx.items():
                    if _norm(sk) == ns or (len(ns) >= 3 and _norm(sk) in ns):
                        s_strength[idx] = max(s_strength[idx], prof_dict.get(ns, 0.6))

        sx = (s_strength > 0).astype(np.float32)
        adv = (s_strength >= 0.95).astype(np.float32)
        if self.S_rel is not None:
            cr = np.maximum(s_strength, self.rel_gate_scalar * (s_strength @ self.S_rel))
        else:
            cr = s_strength

        # Project skill indicators
        proj_skills = {_norm(s) for s in _split_values(profile.get("project_technologies", ""))}
        s_proj = np.zeros(K, dtype=np.float32)
        for sk, idx in self.sk_idx.items():
            if _norm(sk) in proj_skills:
                s_proj[idx] = 1.0

        # Experience & numbers
        projects_count = float(profile.get("projects_count") or 0)
        intern_m = float(profile.get("internship_months") or 0)
        work_m = float(profile.get("work_experience_months") or 0)
        total_exp = intern_m + work_m
        certs_count = float(len(_split_values(profile.get("certifications", ""))))

        # Preferred roles & filters
        pref_roles = {_core_role(r) for r in _split_values(profile.get("preferred_roles", "")) if _core_role(r)}
        student_degree = _norm(profile.get("degree", ""))
        pref_mode = _norm(profile.get("preferred_work_mode", ""))
        pref_type = _norm(profile.get("job_type_preference", ""))
        pref_locs = {_norm(l) for l in _split_values(profile.get("preferred_locations", ""))}

        # 2. Build feature matrix X (N rows, 28 columns)
        X = np.zeros((N, len(self.feature_names)), dtype=np.float32)

        job_records = jobs_df.to_dict("records")
        for i, job in enumerate(job_records):
            req_list = _split_values(job.get("required_skills", ""))
            pref_list = _split_values(job.get("preferred_skills", ""))

            req_vec = np.zeros(K, dtype=np.float32)
            for r in req_list:
                nr = _norm(r)
                if nr in self.sk_idx:
                    req_vec[self.sk_idx[nr]] = 1.0
                else:
                    for sk, idx in self.sk_idx.items():
                        if _norm(sk) == nr:
                            req_vec[idx] = 1.0

            pref_vec = np.zeros(K, dtype=np.float32)
            for p in pref_list:
                np_ = _norm(p)
                if np_ in self.sk_idx:
                    pref_vec[self.sk_idx[np_]] = 1.0

            rs_raw = float(req_vec.sum())
            ps_raw = float(pref_vec.sum())
            rs = max(rs_raw, 1.0)
            ps = max(ps_raw, 1.0)

            hits = float((sx * req_vec).sum())
            req_cov_exact = hits / rs
            req_cov_credit = float((cr * req_vec).sum()) / rs
            req_cov_strength = float((s_strength * req_vec).sum()) / rs
            req_cov_advanced = float((adv * req_vec).sum()) / rs
            req_hits = hits / 7.0
            req_missing = (rs - hits) / 7.0

            pref_cov_exact = float((sx * pref_vec).sum()) / ps
            pref_cov_credit = float((cr * pref_vec).sum()) / ps
            proj_cov = float((s_proj * req_vec).sum()) / rs

            j_role = _core_role(job.get("job_title", ""))
            role_match = 1.0 if (not pref_roles or j_role in pref_roles or any(pr in j_role for pr in pref_roles)) else 0.0

            req_degs = [_norm(d) for d in _split_values(job.get("required_degree", ""))]
            degree_ok = 1.0 if (not req_degs or "anydegree" in req_degs or student_degree in req_degs) else 0.0

            job_min_exp = float(job.get("min_experience_months") or 0)
            gap = np.clip((job_min_exp - total_exp) / 24.0, -1.0, 1.0)
            exp_ok = 1.0 if gap <= 0 else 0.0

            j_mode = _norm(job.get("work_mode", ""))
            mode_ok = 1.0 if (not pref_mode or pref_mode in ("flexible", "any") or pref_mode == j_mode) else 0.0

            j_type = _norm(job.get("job_type", ""))
            type_ok = 1.0 if (not pref_type or pref_type in ("flexible", "any") or pref_type == j_type) else 0.0

            j_loc = _norm(job.get("location", ""))
            loc_ok = 1.0 if (j_mode == "remote" or not pref_locs or "remote" in pref_locs or j_loc in pref_locs) else 0.0

            lvl_idx = _level_index(job.get("job_title", ""))

            # Populate row into X matching feature_names ordering
            # ['req_cov_exact', 'req_cov_credit', 'req_cov_strength', 'req_cov_advanced',
            #  'req_hits', 'req_missing', 'pref_cov_exact', 'pref_cov_credit',
            #  'proj_cov', 'role_match', 'degree_ok', 'exp_gap', 'exp_ok',
            #  'mode_ok', 'type_ok', 'loc_ok', 'projects', 'internship_m',
            #  'work_m', 'certs', 'exp_total', 'job_min_exp',
            #  'lvl_intern', 'lvl_junior', 'lvl_regular', 'lvl_senior',
            #  'n_required', 'n_preferred']
            row = [
                req_cov_exact, req_cov_credit, req_cov_strength, req_cov_advanced,
                req_hits, req_missing, pref_cov_exact, pref_cov_credit,
                proj_cov, role_match, degree_ok, gap, exp_ok,
                mode_ok, type_ok, loc_ok, projects_count, intern_m,
                work_m, certs_count, total_exp, job_min_exp,
                1.0 if lvl_idx == 0 else 0.0,
                1.0 if lvl_idx == 1 else 0.0,
                1.0 if lvl_idx == 2 else 0.0,
                1.0 if lvl_idx == 3 else 0.0,
                rs_raw, ps_raw
            ]
            X[i] = row

        # 3. XGBoost batch prediction
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        p_somewhat = self.boosters["somewhat"].predict(dmat)
        p_relevant = self.boosters["relevant"].predict(dmat)
        p_offer = self.boosters["offer"].predict(dmat) if "offer" in self.boosters else np.zeros(N)

        results = []
        for i, job in enumerate(job_records):
            p_s = float(p_somewhat[i])
            p_r = float(p_relevant[i])
            p_o = float(p_offer[i])

            match_score = round(float((p_s + p_r) / 2.0 * 100.0), 1)

            fit_label = "Strong Fit" if p_r >= 0.55 or match_score >= 70 else (
                "Moderate Fit" if match_score >= 40 else "Potential Fit"
            )

            # Skill matching diagnostics
            req_skills = _split_values(job.get("required_skills", ""))
            pref_skills = _split_values(job.get("preferred_skills", ""))

            matching_req = [
                s for s in req_skills
                if _norm(s) in s_norm_set or any(_norm(s) in sk for sk in s_norm_set)
            ]
            missing_req = [
                s for s in req_skills
                if not (_norm(s) in s_norm_set or any(_norm(s) in sk for sk in s_norm_set))
            ]
            matching_pref = [
                s for s in pref_skills
                if _norm(s) in s_norm_set or any(_norm(s) in sk for sk in s_norm_set)
            ]

            reasons = []
            if matching_req:
                reasons.append(f"Direct match on {len(matching_req)} required skill(s): {', '.join(matching_req[:3])}.")
            if p_r >= 0.6:
                reasons.append("High relevance probability estimated by the trained XGBoost model.")
            if p_o >= 0.4:
                reasons.append(f"Strong offer likelihood ({round(p_o * 100, 1)}%) based on background fit.")
            if X[i, 9] > 0.5:
                reasons.append("Aligns directly with your stated preferred roles.")
            if missing_req:
                reasons.append(f"Consider acquiring: {', '.join(missing_req[:3])}.")

            results.append({
                "job_id": str(job.get("job_id", "")),
                "job_title": str(job.get("job_title", "")),
                "company": str(job.get("company", "")),
                "location": str(job.get("location", "")),
                "work_mode": str(job.get("work_mode", "")),
                "job_type": str(job.get("job_type", "")),
                "required_skills": req_skills,
                "preferred_skills": pref_skills,
                "match_score": match_score,
                "fit_label": fit_label,
                "offer_likelihood": round(p_o * 100.0, 1),
                "p_relevant": round(p_r * 100.0, 1),
                "matching_required_skills": matching_req,
                "missing_required_skills": missing_req,
                "matching_preferred_skills": matching_pref,
                "explanation": reasons,
                "job_description": str(job.get("job_description", "")),
            })

        results.sort(key=lambda x: (x["match_score"], x["offer_likelihood"]), reverse=True)
        return results[:top_n]


# Singleton instance
ml_engine = SkillStudioXGBoostEngine()

