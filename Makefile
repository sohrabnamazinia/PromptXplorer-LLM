# Usage:
#   make run PROMPT_TYPE=primary NUM_ROWS=1000 NUM_CLUSTERS=10
# Default values are provided if not overridden

PYTHON ?= python3
PROMPT_TYPE ?= satellite
NUM_ROWS ?= 20000000
NUM_CLUSTERS ?= 10

.PHONY: run

run:
	$(PYTHON) cluster_prompts.py --prompt_type $(PROMPT_TYPE) --num_rows $(NUM_ROWS) --num_clusters $(NUM_CLUSTERS)

clean:
	rm -f model/* result_lda/*
