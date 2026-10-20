# ML Assignment 2: Bagging, Boosting and Bias–Variance on Olivetti Faces

**Name:** Omkar D. Mane
**Roll No.:** M26CPS005

This repository contains the code, saved data split, results and figures for the report.
It compares three tree-based models on the Olivetti Faces dataset:

- a single decision tree
- bagging (written from scratch)
- multiclass AdaBoost / SAMME (written from scratch)

Each model is tested on clean test images and on masked images, where a 16×16 block in the centre of the face is set to zero.

---

## Repository structure

```
ML_ASSIGNMENT_2/
├── 01_Dataset_split.ipynb      # creates the train/val/test split
├── 02_QuestionSolutions.ipynb  # all experiments (Q1, Q2, Q3)
├── person_data.npz             # saved split (240 train / 80 val / 80 test)
├── requirements.txt            # Python packages and versions
├── clean_vs_masked.pdf         # example clean and masked face
├── README.md
├── que1_results/               # Q1: decision tree and bagging results
├── que2_results/               # Q2: boosting results and comparisons
└── que3_results/               # Q3: bias and variance results
```

---

## Environment

- Python 3.14.6
- Packages are listed in `requirements.txt`

---

## How to reproduce

```bash
git clone https://github.com/maneom/ML_ASSIGNMENT_2.git
cd ML_ASSIGNMENT_2
git checkout <final-commit-hash>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute 02_QuestionSolutions.ipynb \
    --output 02_QuestionSolutions_run.ipynb --ExecutePreprocessor.timeout=-1
```

**Notes:**
- The Olivetti Faces dataset is downloaded automatically by scikit-learn on the first run, so internet is needed once.
- The saved split `person_data.npz` is used directly. You do **not** need to run `01_Dataset_split.ipynb` again. It is included only to show how the split was made.
- All random seeds come from the roll number: **seed = 26005**.
- The full run takes about `[X]` hours on `[your CPU / RAM]`. Most of the time is spent on Q3.

---

## What each question produces

| Question | What it does | Output folder |
|---|---|---|
| Q1 | Single tree baseline, bagging from scratch, OOB accuracy, disagreement, vote margin, clean vs masked test | `que1_results/` |
| Q2 | SAMME boosting from scratch, weight diagnostics, bagging vs boosting, clean vs masked test | `que2_results/` |
| Q3 | Bias² and variance over 30 resampled training sets, effect of tree depth, final comparison | `que3_results/` |

---
