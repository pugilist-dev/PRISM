#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON = python3.12
PIP = pip
VENV_DIR = phenotype
PROJECT_DIR = $(CURDIR)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Create a virtual environment
venv:
	$(PYTHON)	-m	venv	$(VENV_DIR)
## Install dependencies
install:	venv
	$(VENV_DIR)/bin/$(PIP)	install	-r	requirements.txt
## Delete all compiled Python files
clean:
	rm -rf $(VENV_DIR)
	rm -rf wandb
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
# Lint code
lint:
	$(VENV_DIR)/bin/flake8 prism/
train:
	PYTHONPATH=$(PROJECT_DIR)	$(VENV_DIR)/bin/$(PYTHON)	prism/modeling/train_cl.py
.PHONY:	venv	install	lint	train	clean