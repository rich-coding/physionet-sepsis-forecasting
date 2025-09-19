# Simple automation for training, serving, docker, and AWS deploy

PY=python
PIP=pip
UVICORN=uvicorn
VENV=.venv
ACTIVATE=. $(VENV)/bin/activate

IMAGE=sepsis2019-api
TAG=latest
PORT=8080
AWS_REGION=us-east-1
REPO_NAME=$(IMAGE)
CLUSTER=sepsis2019-cluster
SERVICE=sepsis2019-svc

.PHONY: help venv install train serve docker-build docker-run test checks ecr-push ecs-deploy clean

help:
	@echo "Targets:"
	@echo "  make venv          # create virtualenv"
	@echo "  make install       # install python deps"
	@echo "  make train         # train HGB (reads data/raw/*.parquet) -> models/"
	@echo "  make serve         # run FastAPI locally"
	@echo "  make docker-build  # build docker image"
	@echo "  make docker-run    # run docker container"
	@echo "  make test          # run pytest via tox"
	@echo "  make checks        # run lint/type checks via tox"
	@echo "  make ecr-push      # build & push to ECR"
	@echo "  make ecs-deploy    # register task def & update service"
	@echo "  make clean         # remove caches and venv"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(ACTIVATE) && $(PIP) install -r requirements.txt
	@echo "Tip: for tests & typing, also: pip install -r test_requirements.txt -r typing_requirements.txt"

train:
	$(ACTIVATE) && $(PY) src/train_hgb.py

serve:
	$(ACTIVATE) && $(UVICORN) app.main:app --host 0.0.0.0 --port $(PORT)

docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run --rm -p $(PORT):8080 -e MODELS_DIR=/app/models/production/ $(IMAGE):$(TAG)

test:
	$(ACTIVATE) && tox -e test_app

checks:
	$(ACTIVATE) && tox -e checks

ecr-push:
	AWS_REGION=$(AWS_REGION) REPO_NAME=$(REPO_NAME) IMAGE_TAG=$(TAG) bash aws/ecr_push.sh

ecs-deploy:
	AWS_REGION=$(AWS_REGION) CLUSTER=$(CLUSTER) SERVICE=$(SERVICE) bash aws/ecs_deploy.sh

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache __pycache__ mlruns artifacts
	find . -name "*.pyc" -delete
