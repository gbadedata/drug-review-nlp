# drug-review-nlp

Classical NLP and machine learning on the **UCI Drug Review dataset** (~215k patient
reviews of medications). Two things a medical/analytics team actually asks of free-text
feedback: *can we classify sentiment reliably*, and *what are patients complaining about,
by condition*. Both are answered here on real held-out data, with every model measured
against a baseline.

```
load (clean) ──▶ features (text TF-IDF + structured) ──▶ model (sentiment)
                                                     └──▶ themes (complaint terms per condition)
```

## Install

```bash
pip install -r requirements.txt
# then place drugsComTrain_raw.tsv / drugsComTest_raw.tsv in data/ (see data/README.md)
```

A small cleaned sample (`data/sample_reviews.csv`, 400 reviews) is committed so the tests
and a quick look run without the full download.

## Use

```bash
python -m drugnlp.cli info   --data data/sample_reviews.csv
python -m drugnlp.cli train  --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv
python -m drugnlp.cli themes --data data/drugsComTrain_raw.tsv --top 8
```

## Results

### Data cleaning (matters, and is done explicitly)

The raw files carry ~1,800 rows with a missing or scraped-artifact `condition`
("N</span> users found this comment helpful."), HTML-escaped and quote-wrapped review
text, and some truncated condition names ("Bipolar Disorde"). After cleaning and dropping
the neutral 5-6 rating band, the training set is **145,337 reviews across 798 conditions**,
with a **72.4% positive base rate** (so that is the accuracy any model must beat).

### Sentiment classification (dataset's own test split, n=48,438)

| Features | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Majority baseline (always "positive") | 0.724 | 0.420 | 0.500 |
| Structured only (length, votes, freq, year) | 0.725 | 0.453 | 0.707 |
| **Text (TF-IDF 1-2 grams)** | **0.904** | **0.876** | **0.954** |
| Text + structured | 0.906 | 0.877 | 0.957 |

The text model lifts accuracy 18 points and macro-F1 from 0.42 to 0.88 over the baseline.
Structured metadata alone carries only weak signal (AUC 0.71, accuracy at the baseline),
and adds essentially nothing once the text is present (+0.2 points). The review wording
is where the sentiment lives; the metadata is largely redundant for this task.

### What patients complain about (distinctive terms in negative reviews)

Ranked by weighted log-odds with an informative prior, so terms are characteristic rather
than merely frequent:

| Condition | Top distinctive negative terms |
|---|---|
| Birth Control | period, pill, periods, mood swings, sex |
| Depression | (drug names) wellbutrin, effexor, prozac, lexapro, zoloft; **suicidal** |
| Abnormal Uterine Bleeding | bleeding, depo shot, mirena, spotting, clots |
| Vaginal Yeast Infection | burning, itching, cream, itch |
| Acne | cystic acne, pimples, dermatologist, chin |
| Pain | tramadol, oxycodone, norco, percocet, ER |

Lowest-satisfaction conditions (>=300 reviews) include Vaginal Yeast Infection (mean
rating 3.95, 66% negative), Abnormal Uterine Bleeding (4.24), and Osteoporosis (4.78).
The method surfaces both the expected (condition self-reference) and the specific and
actionable (mood-related discontinuation, injection bleeding, the "suicidal" signal in
depression reviews).

Numbers are reproduced in `results/`.

## Design notes

- **Proper held-out evaluation.** The dataset's own train/test split is used, not a
  re-shuffle, so nothing leaks between fit and score.
- **Baselines attached.** Every model sits next to the majority-class baseline on three
  metrics, including ROC-AUC, which is the one that separates "beats the base rate by
  guessing the majority" from "actually ranks reviews correctly".
- **Structured + unstructured, isolated.** Text, structured, and combined feature views
  are scored separately so each one's contribution is legible.
- **Principled term ranking.** Complaint terms use weighted log-odds (Monroe et al.,
  2008) rather than raw counts or plain TF-IDF.

## Data and provenance

UCI Drug Review Dataset (Drugs.com); Kallumadi & Grässer (2018), UCI ML Repository,
https://doi.org/10.24432/C5SK5S. See `data/README.md`. The full data is not redistributed
here.

## Tests

```bash
python -m pytest -q        # 4 tests: cleaning, labelling, neutral-drop, feature alignment
```

## Licence

MIT. Author: Oluwagbade Odimayo · gbadejosef@gmail.com · github.com/gbadedata
