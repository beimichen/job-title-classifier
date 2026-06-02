"""Classify a CSV of job titles in parallel and write the results to output.csv.

Usage:
    python examples/classify_batch.py
"""
import csv
import multiprocessing as mp

from tqdm import tqdm

from job_title_classifier import JobTitleClassifier, load_string_associated_vectors

CONFIDENCE_THRESHOLD = 0.9


def classify_title(job_title):
    labels = sorted(
        x.split("\t")
        for x in open("data/classifier_labels.tsv").read().split("\n")
        if len(x)
    )
    vectors = load_string_associated_vectors("data/token_vectors.tsv")
    clf = JobTitleClassifier(labels, vectors)

    classifications = clf.classify(job_title)
    try:
        titles = [c[1] for c in classifications]
        confidences = [c[0] for c in classifications]
        valid = ""
        for i, conf in enumerate(confidences):
            if conf >= CONFIDENCE_THRESHOLD:
                valid = titles[i]
        return job_title, valid, classifications, titles, confidences
    except Exception:
        print(f"The job title {job_title!r} broke the classifier")
        return ["", "", "", "", ""]


if __name__ == "__main__":
    titles = []
    with open("examples/job_titles.csv", newline="", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f)):
            titles.append(row["title"])

    workers = mp.cpu_count()
    print("workers available:", workers)
    with mp.Pool(workers) as pool:
        results = pool.map(classify_title, tqdm(titles))

    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_title",
                "valid_job",
                "classifications",
                "job_titles_list",
                "job_confidences",
            ],
        )
        writer.writeheader()
        for job_title, valid, classifications, t_list, confs in results:
            if job_title:
                writer.writerow(
                    {
                        "job_title": job_title,
                        "valid_job": valid,
                        "classifications": classifications,
                        "job_titles_list": t_list,
                        "job_confidences": confs,
                    }
                )
