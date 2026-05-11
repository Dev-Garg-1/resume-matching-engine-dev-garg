# 🔍 Resume Matching Engine
### Redrob AI Campus Hackathon — Individual Submission

> Built using **Redrob AI** as the coding assistant | Powered by McKinley Rice

---

## 📌 Problem Statement

Given **10 resumes** from Indian university students and **3 Job Descriptions (JDs)** from Korean technology companies, build a program that:

- Normalizes noisy resume skill data (typos, abbreviations, mixed casing)
- Computes TF-IDF vectors for resumes
- Builds binary vectors for job descriptions
- Calculates cosine similarity between resumes and JDs
- Outputs the **Top 3 matching candidates per JD**

---

## 🏆 Final Results

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## 📁 Project Structure

```
resume-matching-engine/
│
├── main.py                ← Main solution
└── README.md              ← readme file
```

---

## ⚙️ How to Run

```bash
# No external libraries needed — standard Python only
python3 main.py
```

**Requirements:** Python 3.6+ · Standard library only (`math`, `re`, `collections`)

---

## 🧠 How It Works — Step-by-Step Pipeline

The engine follows a 6-step pipeline. Each step is a pure, independently testable function.

---

### Step 1 & 2 — Skill Normalization + Deduplication

**Function:** `normalize_skills(raw, alias_map)` → `normalize_corpus(dataset, alias_map)`

Raw resume data contains significant noise — typos, mixed casing, abbreviations, and inconsistent formatting. The normalizer handles all of this in a deterministic pipeline:

```
Raw Input:   "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"
             ↓ split on commas
             ↓ lowercase + collapse whitespace (re.sub handles "REST  api" → "rest api")
             ↓ greedy longest-phrase match first (multi-word before single-token)
             ↓ apply SKILL_ALIASES
             ↓ discard unknowns, deduplicate (OrderedDict preserves order)
Output:      ['python', 'machine_learning', 'sql', 'pandas', 'numpy', 'deep_learning']
```

**Key design: greedy longest-match first**

Multi-word phrases (e.g. `"feature engineering"`, `"spring boot"`, `"data structure"`) are sorted by length descending and matched **before** single-token lookup. This prevents `"spring"` or `"boot"` from being checked individually when the full phrase `"spring boot"` exists.

```python
_PHRASE_KEYS = sorted(
    [k for k in alias_map if " " in k or "-" in k],
    key=len, reverse=True          # longest first = greedy
)
```

**Alias map examples:**

| Raw token | Canonical skill |
|---|---|
| `Pyhton` | `python` |
| `MachineLearning` | `machine_learning` |
| `kubernates` | `kubernetes` |
| `JavaScrpit` | `javascript` |
| `deep-learning` | `deep_learning` |
| `matplotlib` | `data_visualization` |
| `power-bi` | `data_visualization` |
| `Sklearn` | `machine_learning` |

**Deduplication** is handled inside the same function using `OrderedDict` — preserves insertion order while preventing duplicates. After deduplication, every skill appears exactly once per resume, which is the prerequisite for `TF = 1/N`.

---

### Step 3 — Vocabulary Construction

**Function:** `build_vocabulary(normalized_corpus)` → sorted `list`

A shared vocabulary is built from **all normalized resume skills combined**, then sorted alphabetically. This sorted order defines the vector index used consistently across all resume TF-IDF vectors and JD binary vectors.

```
48 unique canonical skills → alphabetically sorted
['algorithms', 'android', 'aws', 'bert', 'ci_cd', ...]
```

> **Important:** JD skills are excluded from vocabulary construction (per spec). Only skills seen in resumes get a vector dimension.

---

### Step 4a — Document Frequency & IDF

**Functions:** `compute_document_frequency()` · `compute_idf(df, n_docs)`

**Document Frequency (df):** For each skill, count how many resumes contain it.

```
python           → df=6  (appears in 6 of 10 resumes)
machine_learning → df=3
deep_learning    → df=1
```

**IDF formula** (exact as specified — natural log, no smoothing):

```
IDF(skill) = ln( N / df(skill) )

where N = total number of resumes (dynamic, not hardcoded)
```

```
IDF(python)           = ln(10/6) = 0.5108   ← low, common skill
IDF(deep_learning)    = ln(10/1) = 2.3026   ← high, rare skill
IDF(machine_learning) = ln(10/3) = 1.2040   ← medium
```

Skills that appear in fewer resumes get higher IDF — they are more **discriminating** for ranking.

---

### Step 4b — TF-IDF Vectors

**Function:** `build_tfidf_matrix(normalized_corpus, vocab, idf)`

**TF formula** (after deduplication, each skill appears exactly once per resume):

```
TF(skill, resume) = 1 / N     where N = total unique skills in that resume
```

**TF-IDF:**

```
TF-IDF(skill, resume) = TF × IDF = (1/N) × ln(10 / df(skill))
```

Example — Arjun Sharma has 6 skills, so TF = 1/6 for each:

```
deep_learning:    (1/6) × 2.3026 = 0.3838   ← high weight (rare skill)
machine_learning: (1/6) × 1.2040 = 0.2007   ← medium
python:           (1/6) × 0.5108 = 0.0851   ← low (very common)
```

This means rare skills that a candidate holds **contribute more** to their similarity score — a fairer measure than raw skill count.

---

### Step 5 — JD Binary Vectors

**Function:** `build_jd_vectors(jd_dataset, vocab, alias_map)`

JD skills are passed through the **same normalization pipeline** as resumes (same alias map, same phrase matching). The result is encoded as a binary vector over the shared vocabulary:

```
1  → skill is required or preferred by this JD
0  → skill not mentioned
```

Only skills present in the resume vocabulary receive a `1` — JD skills absent from all resumes simply have no vector dimension to match on.

```
JD-1 (ML Engineer) matched skills:
['bert', 'data_visualization', 'deep_learning', 'feature_engineering',
 'machine_learning', 'nlp', 'python', 'sql', 'statistics', 'tensorflow']
```

---

### Step 6 — Cosine Similarity & Ranking

**Functions:** `cosine_similarity(vec_a, vec_b)` · `rank_candidates(tfidf_matrix, jd_vectors, top_k)`

**Formula:**

```
Cosine(A, B) = (A · B) / (|A| × |B|)

where:
  A   = Resume TF-IDF vector
  B   = JD binary vector
  |A| = Euclidean (L2) norm of A
```

The dot product `A · B` sums TF-IDF weights only for skills the resume **and** JD share. Dividing by norms scales to [0, 1] regardless of vector length, making scores comparable across candidates with different skill counts.

**Ranking rules (per spec):**
- Sort by cosine score **descending**
- Break ties **alphabetically by candidate name** (ascending)
- Return top `TOP_K = 3` candidates

**Cosine similarity matrix:**

```
Candidate              JD-1    JD-2    JD-3
Arjun Sharma          0.3958  0.0000  0.0000
Priya Nair            0.0000  0.1172  0.5756
Rahul Gupta           0.0000  0.8109  0.0000
Sneha Patel           0.5696  0.0000  0.0000
Vikram Singh          0.0349  0.0000  0.0000
Ananya Krishnan       0.0298  0.2833  0.3458
Karan Mehta           0.5341  0.0000  0.0000
Deepika Rao           0.0000  0.1906  0.0862
Aditya Kumar          0.0000  0.0000  0.6657
Meera Iyer            0.3345  0.0000  0.0000
```

---

## 🛡️ Robustness Features

This implementation is designed to work correctly on **any dataset**, not just the provided one.

| Feature | How it's handled |
|---|---|
| Typos in skills | `SKILL_ALIASES` maps 100+ raw variants to canonical forms |
| Extra whitespace | `re.sub(r"\s+", " ", token)` collapses double spaces |
| Mixed casing | `.lower()` applied before every lookup |
| Multi-word phrases | Pre-sorted by length, matched greedily before single tokens |
| Dynamic corpus size | `n_docs = len(resumes)` — IDF never hardcodes `10` |
| Zero-overlap candidates | `cosine_similarity` returns `0.0` instead of crashing |
| Fully pluggable | `run_pipeline(resumes, jds, alias_map, top_k)` accepts any inputs |

---

## 📐 Mathematical Summary

```
TF(skill, resume)   =  1 / N
                        where N = |unique skills in resume|

IDF(skill)          =  ln( N_corpus / df(skill) )
                        where df = resumes containing skill

TF-IDF(skill, r)    =  TF × IDF

Cosine(resume, JD)  =  Σ(TF-IDF_i × JD_i)  /  (||TF-IDF|| × ||JD||)
```

No external libraries (numpy, pandas, scikit-learn) are used anywhere. All math is implemented from scratch using Python's `math` standard library.

---

## 🤖 How I Used Redrob AI — Prompt Workflow

The solution was built using **Redrob AI** as the coding assistant, following a staged prompting strategy (which scores highest under the evaluation rubric).

---

### Stage 1 — Explore & Understand the Data

**Prompt sent to Redrob AI:**
> *"I have 10 resumes with noisy skill strings like 'Pyhton', 'MachineLearning', 'kubernates'. I need to normalize these using the SKILL_ALIASES map provided. Show me what each resume looks like after normalization and deduplication."*

**What Redrob AI helped with:**
- Identifying which raw tokens map to which canonical skills
- Catching tricky cases like `"data-viz"` and `"matplotlib"` both mapping to `data_visualization`
- Validating that `"Sneha Patel"` ends up with 6 unique skills (not 7) after dedup since `data-viz` and `matplotlib` merge

**Output validated:** All 10 normalized + deduplicated skill lists confirmed correct.

---

### Stage 2 — Build the Normalization Logic

**Prompt sent to Redrob AI:**
> *"Write a normalize_skills() function in Python that: splits on commas, lowercases, matches multi-word phrases BEFORE single tokens (longest first), applies SKILL_ALIASES, discards unknown tokens, and deduplicates while preserving order. Use only standard libraries."*

**What Redrob AI helped with:**
- Suggesting `OrderedDict` as an elegant dedup-with-order-preservation solution
- Identifying the critical bug in naive approaches: joining tokens across comma boundaries
- Implementing `_build_phrase_index()` — pre-sorting phrase keys by word count descending at import time (not re-sorted on every call)
- Adding `re.sub(r"\s+", " ", token)` for whitespace robustness

**Key insight from this stage:** Multi-word phrases like `"feature engineering"` arrive as a **single comma-separated token** — they should be matched within that token, not by joining adjacent tokens.

---

### Stage 3 — Compute TF-IDF (Manual Implementation)

**Prompt sent to Redrob AI:**
> *"Implement TF-IDF from scratch using only Python's math library. TF = 1/N after deduplication. IDF = ln(N_corpus / df). No numpy, no sklearn. Show me the intermediate df and IDF values for all 48 vocabulary terms so I can verify."*

**What Redrob AI helped with:**
- Implementing `compute_document_frequency()` and `compute_idf()` as separate functions for clarity
- Flagging that IDF should use `n_docs = len(resumes)` dynamically, not hardcode `10`
- Adding the `float("inf")` guard for df=0 (defensive programming for new datasets)
- Printing the full df/IDF table for intermediate validation

**Validated:** `python` IDF = 0.5108 (low, appears in 6/10 resumes), `deep_learning` IDF = 2.3026 (high, appears in 1/10).

---

### Stage 4 — Build JD Vectors + Cosine Similarity

**Prompt sent to Redrob AI:**
> *"Build binary JD vectors using the same normalization pipeline. Then implement cosine similarity and rank the top 3 candidates per JD. Handle the edge case where a candidate has zero overlap with a JD (avoid division by zero). Break ties alphabetically."*

**What Redrob AI helped with:**
- Ensuring JD normalization reuses the exact same `normalize_skills()` function (not a separate implementation)
- Implementing the zero-vector guard: `if norm_a == 0.0 or norm_b == 0.0: return 0.0`
- Using `scores.sort(key=lambda x: (-x[1], x[0]))` for the correct composite sort
- Printing the full cosine similarity matrix for verification before reading top-3

---

### Stage 5 — Generalization & Code Quality

**Prompt sent to Redrob AI:**
> *"Refactor the entire solution so it works on ANY dataset — dynamic corpus size, pluggable inputs, no magic numbers. Add docstrings, type hints, and a diagnostic printer for each step. Also identify any bugs in this alternative implementation [pasted index.py]."*

**What Redrob AI helped with:**
- Wrapping everything in `run_pipeline(resumes, jds, alias_map, top_k)` for full pluggability
- Adding `TOP_K = 3` and `n_docs = len(resumes)` as named constants
- Identifying 6 bugs in the reference `index.py` (dead import, hardcoded IDF, cross-boundary phrase matching, etc.)
- Writing the full diagnostic output suite (`print_df_idf`, `print_similarity_matrix`, etc.)
- Verifying generalization by running the engine on a completely new 4-resume dataset

---

## 📊 Scoring Rubric Alignment

| Criterion | Points | What was done |
|---|---|---|
| **JD-1 Result** | 6 | Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40) |
| **JD-2 Result** | 7 | Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19) |
| **JD-3 Result** | 7 | Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35) |
| **Redrob AI Usage** | 40 | 5 staged prompts (explore → normalize → TF-IDF → vectors → generalize) |
| **Code Quality** | 20 | Pure functions, docstrings, type hints, diagnostic printers, no magic numbers |
| **F2F Discussion** | 20 | See pipeline explanation above |

---

## 🔬 Why This Implementation Is Correct

1. **TF = 1/N** — After deduplication every skill appears exactly once. `count(skill) = 1`, `total_unique = N` → `TF = 1/N` ✅

2. **IDF = ln(N/df)** — Natural log (`math.log`), no smoothing, no +1 ✅

3. **Vocabulary from resumes only** — JD skills are normalized but not added to vocab ✅

4. **Same alias map for JDs** — JD skills go through identical `normalize_skills()` to ensure consistent canonical forms ✅

5. **|A| = Euclidean norm of resume vector** — `math.sqrt(sum(a*a for a in vec))` ✅

6. **Tie-breaking alphabetically** — `sort(key=lambda x: (-x[1], x[0]))` ✅

---

*Submitted for Redrob AI Campus Hackathon · Powered by McKinley Rice*
