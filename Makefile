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
	rm -f id2word.dict lda_model.model lda_model.model.expElogbeta.npy lda_model.model.id2word lda_model.model.state lda_topics.txt
