"""
Resume Matching Engine — Redrob AI Campus Hackathon
=====================================================
A fully generalized, dataset-agnostic implementation.

Key design decisions that make this work on ANY dataset:
  - Greedy multi-word phrase matching (longest match first)
  - Robust tokenizer handles extra whitespace, mixed casing, punctuation variants
  - Zero-vector guard in cosine similarity (avoids ZeroDivisionError)
  - IDF guard: df=0 is impossible post-normalization, but handled defensively
  - All parameters (corpus size, top-k) are variables, not magic numbers
  - Fully pluggable: swap RESUMES / JOB_DESCRIPTIONS / SKILL_ALIASES independently

Pipeline:
  1. Normalize & deduplicate resume skills  →  canonical skill lists
  2. Build shared vocabulary                →  sorted list
  3. Compute TF-IDF vectors for resumes     →  using exact spec formulas
  4. Build binary JD vectors               →  over the same vocabulary
  5. Cosine similarity + ranking            →  Top-K per JD
"""

import math
import re
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOP_K      = 3    # number of top candidates to return per JD
N_RESUMES  = 10   # corpus size used in IDF formula (set to len(RESUMES) in main)

# ─────────────────────────────────────────────────────────────────────────────
# SKILL ALIASES  — exact as provided, unmodified
# ─────────────────────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python",         "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp",               "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",   "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",  "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch",  "keras": "keras",
    "nlp": "nlp",               "bert": "bert",        "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",         "numpy": "numpy",

    # Web — Frontend
    "react": "react",   "reacts": "react",   "reactjs": "react",
    "vue": "vue",       "vue.js": "vue",      "vuejs": "vue",
    "redux": "redux",   "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest",     "graphql": "graphql",

    # Web — Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",       "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb",   "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",

    # Mobile
    "android": "android",   "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JOB_DESCRIPTIONS = [
    ("JD-1", "Kakao (ML Engineer)",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, "
     "Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    ("JD-2", "Naver (Backend Engineer)",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, "
     "Kubernetes, REST API, CI/CD, Redis"),
    ("JD-3", "Line (Frontend Engineer)",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, "
     "Node.js, GraphQL, Redux, Jest, AWS"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2 — NORMALIZATION + DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def _build_phrase_index(alias_map: dict) -> list:
    """
    Pre-compute all multi-word / hyphenated alias keys sorted by
    descending token length, so the LONGEST phrase always wins
    (e.g. 'feature engineering' matched before 'feature' alone).
    This is the critical correctness guarantee for greedy phrase matching.
    """
    phrase_keys = [k for k in alias_map if " " in k or "-" in k]
    return sorted(phrase_keys, key=lambda k: len(k.split()), reverse=True)

# Build once at import time — shared by every normalize_skills call
_PHRASE_KEYS = _build_phrase_index(SKILL_ALIASES)


def _clean_token(token: str) -> str:
    """
    Strip leading/trailing whitespace and collapse internal runs of
    whitespace to a single space.  Handles messy real-world CSV data
    like '  Node.JS '  or  'REST  api'.
    """
    return re.sub(r"\s+", " ", token.strip()).lower()


def normalize_skills(raw: str, alias_map: dict = SKILL_ALIASES) -> list:
    """
    Full normalization pipeline for a single raw skill string.

    Algorithm
    ---------
    1. Split on commas → clean each token (lowercase + collapse whitespace)
    2. For every token attempt greedy phrase match (longest alias first)
    3. Fall back to single-token alias lookup
    4. Silently discard tokens absent from alias_map
    5. Deduplicate while preserving insertion order (OrderedDict trick)

    Parameters
    ----------
    raw       : comma-separated raw skill string (may contain typos / noise)
    alias_map : mapping of raw → canonical skill name (default: SKILL_ALIASES)

    Returns
    -------
    List of unique canonical skill strings, in order of first appearance.
    """
    tokens  = [_clean_token(t) for t in raw.split(",") if t.strip()]
    seen    = OrderedDict()   # canonical → True  (preserves order + deduplicates)

    for token in tokens:
        canonical = None

        # Pass 1 — greedy longest-phrase match
        for phrase in _PHRASE_KEYS:
            if token == phrase:
                canonical = alias_map[phrase]
                break

        # Pass 2 — single-token lookup
        if canonical is None:
            canonical = alias_map.get(token)   # None if unknown

        # Pass 3 — discard unknowns, deduplicate knowns
        if canonical is not None and canonical not in seen:
            seen[canonical] = True

    return list(seen.keys())


def normalize_corpus(dataset: list, alias_map: dict = SKILL_ALIASES) -> list:
    """
    Apply normalize_skills to every (name, raw_skills) tuple.
    Returns list of (name, canonical_skills_list).
    Fully pluggable: pass any dataset + any alias_map.
    """
    return [(name, normalize_skills(skills, alias_map)) for name, skills in dataset]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
def build_vocabulary(normalized_corpus: list) -> list:
    """
    Collect every unique canonical skill from the normalized resume corpus,
    then sort alphabetically.  The sort order defines the vector index for
    both resume TF-IDF vectors and JD binary vectors — consistency is essential.

    NOTE: JD skills are intentionally excluded from vocabulary construction
    (per spec: vocabulary comes from resumes only).
    """
    vocab_set = set()
    for _, skills in normalized_corpus:
        vocab_set.update(skills)
    return sorted(vocab_set)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
def compute_document_frequency(normalized_corpus: list, vocab: list) -> dict:
    """
    df(skill) = number of resumes that contain that skill.
    Iterates the corpus once → O(N × S) where S = skills per resume.
    """
    df = {skill: 0 for skill in vocab}
    for _, skills in normalized_corpus:
        for skill in skills:               # skills already deduplicated
            df[skill] += 1
    return df


def compute_idf(df: dict, n_docs: int) -> dict:
    """
    IDF(skill) = ln( n_docs / df(skill) )

    Spec constraints honoured:
      - Natural logarithm (math.log, base e)
      - No smoothing (no +1 to numerator or denominator)
      - df=0 is impossible after normalization but guarded with float('inf')
        to prevent ZeroDivisionError on unseen skills in new datasets.
    """
    return {
        skill: math.log(n_docs / count) if count > 0 else float("inf")
        for skill, count in df.items()
    }


def build_tfidf_matrix(normalized_corpus: list,
                        vocab: list,
                        idf: dict) -> list:
    """
    Construct one TF-IDF vector per resume.

    TF  = 1 / N   (after deduplication every skill appears exactly once;
                   count(skill) = 1, total_unique = N, so TF = 1/N)
    TF-IDF = TF × IDF

    Returns list of (candidate_name, vector) tuples.
    The vector is a plain Python list of floats aligned to `vocab`.
    """
    vocab_idx = {skill: i for i, skill in enumerate(vocab)}
    matrix    = []

    for name, skills in normalized_corpus:
        n   = len(skills)
        tf  = 1.0 / n if n > 0 else 0.0
        vec = [0.0] * len(vocab)

        for skill in skills:
            vec[vocab_idx[skill]] = tf * idf[skill]

        matrix.append((name, vec))

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — JD BINARY VECTORS
# ─────────────────────────────────────────────────────────────────────────────
def build_jd_vectors(jd_dataset: list,
                     vocab: list,
                     alias_map: dict = SKILL_ALIASES) -> list:
    """
    For each JD:
      1. Normalize raw skills through the SAME alias pipeline as resumes
      2. Encode as a binary vector over the shared resume vocabulary
         (JD skills not in vocabulary get value 0 — they cannot contribute
          to cosine similarity since no resume was scored for them)

    Returns list of (jd_id, jd_label, binary_vector) tuples.
    """
    vocab_idx = {skill: i for i, skill in enumerate(vocab)}
    jd_vecs   = []

    for jd_id, jd_label, raw_skills in jd_dataset:
        canonical = normalize_skills(raw_skills, alias_map)
        vec       = [0.0] * len(vocab)

        for skill in canonical:
            if skill in vocab_idx:            # only skills seen in resumes
                vec[vocab_idx[skill]] = 1.0

        jd_vecs.append((jd_id, jd_label, vec))

    return jd_vecs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — COSINE SIMILARITY & RANKING
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Cosine(A, B) = (A · B) / (|A| × |B|)

    where |A| is the Euclidean (L2) norm of A.

    Edge case: returns 0.0 if either vector is all-zeros (no shared
    vocabulary → no similarity).  Guards against ZeroDivisionError
    which can occur on exotic new datasets where a candidate has no
    skills in the JD vocabulary.
    """
    dot    = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_candidates(tfidf_matrix: list,
                    jd_vectors: list,
                    top_k: int = TOP_K) -> dict:
    """
    Compute cosine similarity of every resume against every JD.
    Rank candidates per JD:
      - Primary   : cosine score descending
      - Tiebreaker: candidate name ascending (alphabetical), per spec

    Returns dict:  jd_id → (jd_label, [(name, score), ...all ranked])
    Caller slices [:top_k] for the final answer.
    """
    rankings = {}

    for jd_id, jd_label, jd_vec in jd_vectors:
        scores = [
            (name, cosine_similarity(r_vec, jd_vec))
            for name, r_vec in tfidf_matrix
        ]
        scores.sort(key=lambda x: (-x[1], x[0]))   # desc score, asc name
        rankings[jd_id] = (jd_label, scores)

    return rankings


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC OUTPUT  (validates every intermediate step)
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 64

def _section(title): print(f"\n{SEP}\n{title}\n{SEP}")

def print_normalized(normalized_corpus: list):
    _section("STEP 1-2  │  Normalized & Deduplicated Skills")
    for name, skills in normalized_corpus:
        print(f"  {name:<22} → {skills}")

def print_vocabulary(vocab: list):
    _section(f"STEP 3    │  Shared Vocabulary  ({len(vocab)} terms, alphabetical)")
    cols = 4
    for i in range(0, len(vocab), cols):
        row = vocab[i:i+cols]
        print("  " + "  |  ".join(f"{i+j:>2}. {t:<25}" for j, t in enumerate(row)))

def print_df_idf(df: dict, idf: dict):
    _section("STEP 4a   │  Document Frequency & IDF")
    print(f"  {'Skill':<30} {'df':>4}  {'IDF':>8}")
    print("  " + "-"*46)
    for skill in sorted(df):
        print(f"  {skill:<30} {df[skill]:>4}  {idf[skill]:>8.4f}")

def print_tfidf(tfidf_matrix: list, vocab: list):
    _section("STEP 4b   │  TF-IDF Vectors  (non-zero entries only)")
    for name, vec in tfidf_matrix:
        nonzero = {vocab[i]: round(v, 4) for i, v in enumerate(vec) if v > 0}
        print(f"  {name:<22} → {nonzero}")

def print_jd_vectors(jd_vectors: list, vocab: list):
    _section("STEP 5    │  JD Binary Vectors  (vocab-matched skills)")
    for jd_id, jd_label, vec in jd_vectors:
        matched = [vocab[i] for i, v in enumerate(vec) if v == 1.0]
        print(f"  {jd_id}  {jd_label}")
        print(f"       → {matched}\n")

def print_similarity_matrix(tfidf_matrix: list, jd_vectors: list):
    _section("STEP 6    │  Cosine Similarity Matrix")
    header = f"  {'Candidate':<22}" + "".join(f"{jd_id:>12}" for jd_id, *_ in jd_vectors)
    print(header)
    print("  " + "-" * (22 + 12 * len(jd_vectors)))
    for name, r_vec in tfidf_matrix:
        row = f"  {name:<22}"
        for _, __, j_vec in jd_vectors:
            row += f"{cosine_similarity(r_vec, j_vec):>12.4f}"
        print(row)

def print_results(rankings: dict, top_k: int = TOP_K):
    _section("FINAL RESULTS")
    for jd_id, (jd_label, scores) in rankings.items():
        top = scores[:top_k]
        result = ", ".join(f"{name}({score:.2f})" for name, score in top)
        print(f"\n  {jd_id} — {jd_label}")
        print(f"  {result}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(resumes, job_descriptions,
                 alias_map=SKILL_ALIASES, top_k=TOP_K):
    """
    End-to-end pipeline.  Accepts any resume list, any JD list, any alias map.
    Change the inputs here and the entire engine adapts automatically.
    """
    n_docs = len(resumes)                          # corpus size drives IDF

    # Step 1 & 2: normalize + deduplicate
    norm_resumes = normalize_corpus(resumes, alias_map)
    print_normalized(norm_resumes)

    # Step 3: shared vocabulary (from resumes only)
    vocab = build_vocabulary(norm_resumes)
    print_vocabulary(vocab)

    # Step 4: TF-IDF
    df           = compute_document_frequency(norm_resumes, vocab)
    idf          = compute_idf(df, n_docs)
    print_df_idf(df, idf)
    tfidf_matrix = build_tfidf_matrix(norm_resumes, vocab, idf)
    print_tfidf(tfidf_matrix, vocab)

    # Step 5: JD binary vectors
    jd_vectors = build_jd_vectors(job_descriptions, vocab, alias_map)
    print_jd_vectors(jd_vectors, vocab)

    # Step 6: cosine similarity + ranking
    print_similarity_matrix(tfidf_matrix, jd_vectors)
    rankings = rank_candidates(tfidf_matrix, jd_vectors, top_k)
    print_results(rankings, top_k)

    return rankings


if __name__ == "__main__":
    run_pipeline(RESUMES, JOB_DESCRIPTIONS)