from lda import (
    lda_primary,
    lda_satellite,
    lda_primary_inference,
    lda_satellite_inference,
)
import sys

def run_all():
    topics_count = 100
    max_rows = 500000
    alpha = 0
    beta = 5

    original_stdout = sys.stdout  # save original stdout

    print("\n=== Training LDA for Primary Prompts ===")
    lda_primary(topics_count, max_rows)
    sys.stdout = original_stdout  # restore

    print("\n=== Training LDA for Satellite Prompts ===")
    lda_satellite(topics_count, max_rows)
    sys.stdout = original_stdout  # restore

    print("\n=== Running Inference on Primary Prompts ===")
    lda_primary_inference(topics_count, max_rows, alpha, beta)
    sys.stdout = original_stdout  # restore

    print("\n=== Running Inference on Satellite Prompts ===")
    lda_satellite_inference(topics_count, max_rows, alpha, beta)
    sys.stdout = original_stdout  # restore

if __name__ == "__main__":
    run_all()
