# Data

This project uses the **UCI Drug Review Dataset (Drugs.com)**.

- Kallumadi, S. & Grässer, F. (2018). *Drug Review Dataset (Drugs.com)*.
  UCI Machine Learning Repository. https://doi.org/10.24432/C5SK5S
- Grässer, F. et al. (2018). *Aspect-Based Sentiment Analysis of Drug Reviews Applying
  Cross-Domain and Cross-Data Learning.* Proceedings of DH '18.

The full data is **not committed** (per the dataset's usage terms it is for research use
and should be obtained from the source). Download `drugsComTrain_raw.tsv` and
`drugsComTest_raw.tsv` from the UCI repository above, or a public mirror such as
`github.com/Rakesh9100/ML-Project-Drug-Review-Dataset` (datasets/), and place them here.

`sample_reviews.csv` is a small cleaned sample (a few hundred rows across five conditions)
committed only so the tests and a quick demo run without the full download.
