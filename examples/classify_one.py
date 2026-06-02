"""Classify a single job title against the reference label set."""
from job_title_classifier import JobTitleClassifier, load_string_associated_vectors

# Reference strings: (string, class) pairs, tab-separated
labels = sorted(
    x.split("\t")
    for x in open("data/classifier_labels.tsv").read().split("\n")
    if len(x)
)
# Token embeddings used to vectorize strings
vectors = load_string_associated_vectors("data/token_vectors.tsv")

clf = JobTitleClassifier(labels, vectors)

result = clf.classify("javascript developer")
# result is a list of (confidence, class) tuples
ranked = sorted(result, reverse=True)
print("Top matches:")
for confidence, label in ranked[:5]:
    print(f"  {confidence:.3f}  {label}")
