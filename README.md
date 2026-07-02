# drug-review-nlp

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![tests](https://github.com/gbadedata/drug-review-nlp/actions/workflows/tests.yml/badge.svg)](https://github.com/gbadedata/drug-review-nlp/actions/workflows/tests.yml)

Classical NLP and machine learning on the **UCI Drug Review dataset** (~215,000 patient
reviews of medications). It answers two things a medical or analytics team actually asks
of free-text patient feedback: can we classify sentiment reliably at scale, and what are
patients complaining about, broken down by condition. Both are answered on real, held-out
data, with the messy source data cleaned explicitly and every model measured against a
baseline.

> **Headline results.** A text model classifies review sentiment at **90.4% accuracy**,
> **0.876 macro-F1** and **0.954 ROC-AUC** on the dataset's own held-out test split,
> against a 72.4% majority baseline. Structured metadata adds almost nothing on top of the
> text. A weighted log-odds analysis turns the same reviews into interpretable,
> condition-level complaint themes.

---

## Contents

- [Overview](#overview)
- [Why this exists](#why-this-exists)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [The data and its quirks](#the-data-and-its-quirks)
- [Method](#method)
- [Results](#results)
- [Design decisions](#design-decisions)
- [Limitations and how far to trust the results](#limitations-and-how-far-to-trust-the-results)
- [Reproducing the results](#reproducing-the-results)
- [Project structure](#project-structure)
- [Tests](#tests)
- [References](#references)
- [License](#license)

---

## Overview

Patient review sites hold large volumes of unstructured feedback on medicines. This project
does two useful things with it. First, it builds a reliable, interpretable sentiment
classifier and shows exactly how much lift it gives over a naive baseline. Second, it mines
the negative reviews for the terms that most distinguish each condition, turning free text
into structured, condition-level insight about what patients struggle with. The whole thing
is a proper Python package with a command-line interface, unit tests, and reproduced results.

---

## Why this exists

Two questions come up repeatedly when a team is handed a pile of patient feedback. Can we
tell positive from negative automatically, so feedback can be triaged, tracked and reported
at scale rather than read by hand. And can we go beyond a sentiment score to say what is
actually being said, per condition, so the signal is actionable. This project answers both
on a large, real dataset, and pairs a GenAI-heavy retrieval project with the classical
statistics-and-ML foundations that underpin it: clean the data properly, attach a baseline
to every model, isolate each component's contribution, and be careful about what the
results do and do not support.

---

## Pipeline

```
load (clean) ─▶ features (text TF-IDF + structured) ─┬─▶ model (sentiment classification)
                                                     └─▶ themes (complaint terms per condition)
```

| Module | What it does | Why it is built this way |
|---|---|---|
| `load.py` | Reads the raw files, fixes HTML and quoting, drops artifact rows, derives the sentiment label and a block of structured features | Cleaning is a first-class step; ignoring the quirks changes the numbers |
| `features.py` | Builds three views: TF-IDF text, scaled structured features, and the two combined | Keeping the views separate lets each one's contribution be measured |
| `model.py` | Trains and evaluates sentiment on the dataset's own train/test split against a majority baseline, on accuracy, macro-F1 and ROC-AUC | A proper held-out split and attached baselines make the result trustworthy |
| `themes.py` | Ranks the terms that distinguish a condition's negative reviews using weighted log-odds with an informative prior | A principled method surfaces characteristic terms, not merely frequent ones |
| `cli.py` | Exposes `info`, `train`, and `themes` | Usable from the shell, not only as a library |

---

## Installation

```bash
pip install -r requirements.txt
# then place drugsComTrain_raw.tsv / drugsComTest_raw.tsv in data/ (see data/README.md)
```

A small cleaned sample (`data/sample_reviews.csv`, 400 reviews) is committed so the tests
and a quick look run without the full download.

---

## Usage

```bash
python -m drugnlp.cli info   --data data/sample_reviews.csv
python -m drugnlp.cli train  --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv
python -m drugnlp.cli themes --data data/drugsComTrain_raw.tsv --top 8
```

---

## The data and its quirks

The **UCI Drug Review Dataset (Drugs.com)** contains about 215,000 reviews (161,000 train
and 54,000 test as shipped), each with a drug name, condition, free-text review, a 1-10
rating, a date, and a count of how many readers found it useful.

Three data-quality issues are handled explicitly, because they change the results if left
in:

- About **900 rows** carry a scraping artifact in the `condition` field
  ("N&lt;/span&gt; users found this comment helpful."), and about **900 more** have no
  condition at all, so roughly 1,800 rows are removed.
- Review text is **HTML-escaped** (`&amp;`, `&#039;`) and wrapped in stray quotes, so it is
  unescaped and normalised.
- Some condition names are **truncated** in the source ("Bipolar Disorde", "Overactive
  Bladde"), which is noted and left visible rather than silently altered.

Ratings cluster hard at the extremes, so sentiment is derived with an explicit neutral band
that is dropped rather than guessed: rating **>= 7 positive**, **<= 4 negative**, and 5-6
excluded. After cleaning, the training set is **145,337 reviews across 798 conditions**,
with a **72.4% positive base rate**, which is the accuracy any classifier must beat.

The full data is not redistributed here (see `data/README.md`); it is licensed CC BY 4.0
and available from the UCI repository.

---

## Method

**Sentiment classification.** Three feature views are each scored so their contributions can
be read off directly: TF-IDF over 1-2 grams (text), a scaled block of structured signals
(review length, word count, usefulness votes, how common the drug and condition are, the
year), and the two stacked together. Each is a logistic regression, evaluated on the
dataset's own train/test split rather than a re-shuffle, so nothing leaks between fit and
score. All three are reported next to the majority-class baseline on accuracy, macro-F1 and
ROC-AUC. ROC-AUC is included because it separates a model that genuinely ranks reviews
correctly from one that merely guesses the majority class.

**Complaint mining.** For each high-volume condition, the negative reviews are compared
against the negative reviews of every other condition, and the terms that most distinguish
them are ranked by the **weighted log-odds ratio with an informative Dirichlet prior**
(Monroe, Colaresi & Quinn, 2008). This method is used instead of raw frequency or plain
TF-IDF because it corrects both for how common a word is overall and for the sampling
variance of rare words, so the terms it surfaces are characteristic of the condition rather
than simply frequent.

---

## Results

### Sentiment classification (dataset's own test split, n = 48,438)

| Features | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Majority baseline (always "positive") | 0.724 | 0.420 | 0.500 |
| Structured only (length, votes, freq, year) | 0.725 | 0.453 | 0.707 |
| **Text (TF-IDF 1-2 grams)** | **0.904** | **0.876** | **0.954** |
| Text + structured | 0.906 | 0.877 | 0.957 |

**Interpretation.** The text model lifts accuracy 18 points over the baseline and macro-F1
from 0.42 to 0.88, with an ROC-AUC of 0.954. Structured metadata alone carries only weak
signal (AUC 0.71, accuracy at the baseline), and adds essentially nothing once the text is
present (a 0.2-point gain). The sentiment of a review lives in its words, and a TF-IDF plus
linear model captures it well.

**Implication.** This is the right baseline to establish before reaching for anything
heavier. It is fast, interpretable (you can read the coefficients), and already strong, so
a transformer would need to beat 0.95 AUC by a real margin to justify its cost and opacity.
The near-null lift from structured features is itself the useful finding: the correct
engineering decision is to ship the text model and not carry the metadata for this task.

### What patients complain about (distinctive terms in negative reviews)

Ranked by weighted log-odds, so terms are characteristic of the condition rather than
merely frequent:

| Condition | Top distinctive negative terms |
|---|---|
| Birth Control | period, pill, periods, mood swings, sex |
| Depression | antidepressant names (wellbutrin, effexor, prozac, lexapro, zoloft), and the term "suicidal" |
| Abnormal Uterine Bleeding | bleeding, depo shot, mirena, spotting, clots |
| Vaginal Yeast Infection | burning, itching, cream, itch |
| Acne | cystic acne, pimples, dermatologist, chin |
| Pain | tramadol, oxycodone, norco, percocet, ER |

Lowest-satisfaction conditions with at least 300 reviews include **Vaginal Yeast
Infection** (mean rating 3.95, 66% negative), **Abnormal Uterine Bleeding** (4.24) and
**Osteoporosis** (4.78).

**Interpretation.** The method separates the expected (a condition's own name) from the
specific and actionable: mood-related discontinuation for birth control, injection and
device bleeding for abnormal uterine bleeding, and specific opioids and acute contexts for
pain. The appearance of "suicidal" among depression negatives is a safety-relevant signal.

**Implication.** Free text can be turned into structured, condition-level insight of the
kind medical and safety teams monitor. The "suicidal" signal is a pointer for
pharmacovigilance review, not a causal claim: the appropriate response is to quantify it by
drug, check it against known labelling, and escalate to domain experts. The pipeline turns
a large volume of unstructured feedback into a shortlist worth a human looking at.

---

## Design decisions

- **Proper held-out evaluation.** The dataset's own train/test split is used, not a
  re-shuffle, so nothing leaks between fit and score.
- **Baselines attached, on three metrics.** Every model sits next to the majority-class
  baseline, and ROC-AUC is reported alongside accuracy so a model that only guesses the
  majority cannot look good.
- **Structured and unstructured, isolated.** Text, structured, and combined views are
  scored separately so each contribution is legible, and a null result is reported as
  readily as a positive one.
- **Principled term ranking.** Complaint terms use weighted log-odds (Monroe et al., 2008)
  rather than raw counts or plain TF-IDF.
- **Cleaning as a first-class step.** The artifact rows and HTML escaping are removed up
  front, because they change both the metrics and the legibility of the themes.

---

## Limitations and how far to trust the results

- **Self-selected, polarised sample.** People who review medicines online skew towards
  strong experiences, and ratings pile up at 1 and 10. Trust the model's ability to
  discriminate sentiment on this data; do not read the satisfaction rates as representative
  of the whole patient population.
- **Binary framing drops the middle.** The neutral 5-6 band (about 14,000 reviews) is
  excluded, so the classifier is not asked to handle genuine ambiguity. A three-class or
  ordinal model is the next step if the middle matters.
- **A linear bag-of-words model.** N-grams capture some negation and phrasing but not deep
  semantics. The 0.95 AUC baseline is strong and interpretable; a transformer is a
  candidate only if it beats it by a real margin.
- **Distinctiveness is not causation.** The log-odds terms include condition
  self-references, and "suicidal" appearing among depression negatives is a signal to
  investigate, not evidence that a drug causes harm.
- **Vintage data.** The reviews are roughly 2008-2017, so the drug landscape has moved on.

**Net:** trust the classifier's discrimination and the coherence of the mined themes; treat
prevalence rates and any single surfaced term as starting points for review rather than
conclusions.

---

## Reproducing the results

```bash
# after placing the two TSVs in data/
python -m drugnlp.cli train  --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv
python -m drugnlp.cli themes --data data/drugsComTrain_raw.tsv --top 8
cat results/sentiment_metrics.json    # the classification numbers above
cat results/negative_themes.json      # the per-condition terms
```

---

## Project structure

```
drug-review-nlp/
├── drugnlp/
│   ├── load.py       # cleaning, sentiment labelling, structured feature derivation
│   ├── features.py   # TF-IDF text + scaled structured + combined views
│   ├── model.py      # sentiment training + evaluation vs baseline (accuracy/F1/AUC)
│   ├── themes.py     # weighted log-odds complaint terms + satisfaction table
│   └── cli.py        # info, train, themes commands
├── tests/            # 4 unit tests
├── data/
│   ├── README.md     # provenance, licence, download instructions
│   └── sample_reviews.csv   # small cleaned sample for tests and demo
├── results/          # reproduced metrics and themes
├── requirements.txt
└── LICENSE
```

---

## Tests

```bash
python -m pytest -q      # 4 tests: cleaning, labelling, neutral-drop, feature alignment
```

Run automatically on push via GitHub Actions across Python 3.10, 3.11 and 3.12.

---

## References

- Kallumadi, S., Grässer, F. (2018). *Drug Review Dataset (Drugs.com).* UCI Machine
  Learning Repository. https://doi.org/10.24432/C5SK5S (CC BY 4.0).
- Gräßer, F., Kallumadi, S., Malberg, H., Zaunseder, S. (2018). *Aspect-Based Sentiment
  Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.* Proceedings of
  the 2018 International Conference on Digital Health.
- Monroe, B. L., Colaresi, M. P., Quinn, K. M. (2008). *Fightin' Words: Lexical Feature
  Selection and Evaluation for Identifying the Content of Political Conflict.* Political
  Analysis.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
