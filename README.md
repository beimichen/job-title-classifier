# Job Title Classifier

A lightweight, dependency-light **string classifier** that maps free-text job
titles to a canonical set of labels using word embeddings and k-nearest
neighbors. It first tries an exact match against the reference set, then falls
back to a kNN vote over averaged token vectors (Euclidean distance).

Although it ships with job-title data, the algorithm is domain-agnostic — swap
in your own `(string, class)` reference labels and token vectors to classify
any short strings.

## How it works

1. **Reference labels** — a set of `(string, class)` pairs representative of
   what you want to classify (`data/classifier_labels.tsv`).
2. **Token vectors** — token → embedding pairs used to vectorize strings by
   averaging the vectors of their `nltk.wordpunct` tokens
   (`data/token_vectors.tsv`).
3. **Classify** — exact-match lookup first; otherwise kNN over the reference
   vectors with tunable `_range`, `_k`, and `confidence` thresholds.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from job_title_classifier import JobTitleClassifier, load_string_associated_vectors

labels = sorted(x.split("\t") for x in
                open("data/classifier_labels.tsv").read().split("\n") if len(x))
vectors = load_string_associated_vectors("data/token_vectors.tsv")

clf = JobTitleClassifier(labels, vectors)
print(sorted(clf.classify("javascript developer"), reverse=True)[:5])
```

### Examples

```bash
python examples/classify_one.py      # single title
python examples/classify_batch.py    # parallel batch over examples/job_titles.csv -> output.csv
```

## Data files

| File | Description |
|------|-------------|
| `data/classifier_labels.tsv`  | Tab-separated `(string, class)` reference labels |
| `data/token_vectors.tsv`      | Tab-separated `token \t v1,v2,...` embeddings (300-dim) |
| `data/master_job_titles.txt`  | A larger raw list of job titles for experimentation |
| `examples/job_titles.csv`     | Sample input for the batch example |

## Tuning

`classify(search_string, _range=5.0, _k=20, confidence=0.90)`

- `_range` — max Euclidean distance for a reference vector to count as a neighbor
- `_k` — number of nearest neighbors to vote
- `confidence` — minimum winning vote share to accept a prediction

## License

MIT — see [LICENSE](LICENSE).
