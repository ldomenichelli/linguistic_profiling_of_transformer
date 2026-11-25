PYTHON ?= python3

.PHONY: site check serve audit-data

site:
	$(PYTHON) scripts/build_site.py
	$(PYTHON) scripts/check_site.py

check:
	$(PYTHON) scripts/check_site.py
	$(PYTHON) scripts/validate_notebooks.py

serve:
	$(PYTHON) -m http.server 8766 --directory docs

audit-data:
	$(PYTHON) scripts/audit_dataset.py --csv code/data/en_ewt-ud-train_sentences.csv --top-k 5
	$(PYTHON) scripts/audit_dataset.py --csv code/data/it_isdt-ud-train_sentences.csv --top-k 5
