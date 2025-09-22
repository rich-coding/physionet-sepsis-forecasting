# ---------- VARS ----------
PY=python
PIP=pip
UVICORN=uvicorn
VENV=.venv
ACTIVATE=. $(VENV)/bin/activate
BREW=brew

# Imágenes locales
API_IMAGE=sepsis2019-api
API_TAG=latest
FE_IMAGE=sepsis2019-frontend
FE_TAG=latest

# ECR (ajusta tu account y región)
AWS_ACCOUNT_ID=728377120672
AWS_REGION=us-east-1
ECR_URI=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

# Repos ECR (uno por imagen)
API_REPO_NAME=$(API_IMAGE)
FE_REPO_NAME=$(FE_IMAGE)

# URIs completas (con tag)
API_IMAGE_URI=$(ECR_URI)/$(API_REPO_NAME):$(API_TAG)
FE_IMAGE_URI=$(ECR_URI)/$(FE_REPO_NAME):$(FE_TAG)

# EC2/TF
CLUSTER=sepsis2019-cluster
SERVICE=sepsis2019-svc
INSTANCE_TYPE=t3.micro
KEY_NAME=FirstEC2InstancePairKey
HOST_PORT=80
APP_PORT=80
INSTANCE_PROFILE_NAME=LabInstanceProfile
PROJECT_NAME=sepsis

.PHONY: help venv install train serve docker-build-api docker-build-frontend docker-build-all \
        up down logs ps docker-run test checks \
        ecr-login ecr-create-repos ecr-push-api ecr-push-frontend ecr-push-all \
        ec2-deploy ec2-destroy ensure-brew ensure-terraform terraform-install clean

help:
	@echo "Targets:"
	@echo "  make venv          # create virtualenv"
	@echo "  make install       # install python deps"
	@echo "  make train         # train HGB (reads data/raw/*.parquet) -> models/"
	@echo "  make serve         # run FastAPI locally"
	@echo "  make docker-build  # build docker image"
	@echo "  make docker-run    # run docker container with API"
	@echo "  make up    		# deploy docker compose locally with API and Front"
	@echo "  make down    		# destroy docker compose locally with API and Front"
	@echo "  make test          # run pytest via tox"
	@echo "  make checks        # run lint/type checks via tox"
	@echo "  make ecr-push      # build & push to ECR"
	@echo "  make ec2-deploy    # deploy EC2 with ECR images registred"
	@echo "  make ec2-destroy   # destroy infraestructure deployed"
	@echo "  make clean         # remove caches and venv"
	@echo "  make docker-build-all     # build API+Front"
	@echo "  make ecr-create-repos     # crea repos ECR (si no existen)"
	@echo "  make ecr-push-all         # push API+Front a ECR"
	@echo "  make ec2-deploy           # EC2 con ambas imágenes (Terraform)"
	@echo "  make ec2-destroy          # destruye la infra"

# ---------- Local Dev ----------
venv:
	$(PY) -m venv $(VENV)

install: venv
	$(ACTIVATE) && $(PIP) install -r requirements.txt

train:
	$(ACTIVATE) && $(PY) src/train_hgb.py

serve:
	$(ACTIVATE) && $(UVICORN) app.main:app --host 0.0.0.0 --port 8080

# ---------- Docker builds ----------
docker-build-api:
	docker build -t $(API_IMAGE):$(API_TAG) -f app/Dockerfile .

docker-build-frontend:
	docker build -t $(FE_IMAGE):$(FE_TAG) ./front

docker-build-all: docker-build-api docker-build-frontend

# ---------- Compose local ----------
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

docker-run:
	docker run --rm -p 8081:80 -e MODELS_DIR=/app/models/production/ $(API_IMAGE):$(API_TAG)

test:
	$(ACTIVATE) && tox -e test_app
checks:
	$(ACTIVATE) && tox -e checks

# ---------- ECR ----------
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) \
	| docker login --username AWS --password-stdin $(ECR_URI)

ecr-create-repos:
	-aws ecr describe-repositories --repository-names $(API_REPO_NAME) --region $(AWS_REGION) >/dev/null 2>&1 || \
	  aws ecr create-repository --repository-name $(API_REPO_NAME) --region $(AWS_REGION)
	-aws ecr describe-repositories --repository-names $(FE_REPO_NAME) --region $(AWS_REGION) >/dev/null 2>&1 || \
	  aws ecr create-repository --repository-name $(FE_REPO_NAME) --region $(AWS_REGION)

ecr-push-api: ecr-login
	docker tag $(API_IMAGE):$(API_TAG) $(API_IMAGE_URI)
	docker push $(API_IMAGE_URI)

ecr-push-frontend: ecr-login
	docker tag $(FE_IMAGE):$(FE_TAG) $(FE_IMAGE_URI)
	docker push $(FE_IMAGE_URI)

ecr-push-all: docker-build-all ecr-create-repos ecr-push-api ecr-push-frontend
	@echo "Pushed:"
	@echo "  API  -> $(API_IMAGE_URI)"
	@echo "  Front-> $(FE_IMAGE_URI)"

# ---------- Terraform EC2 ----------
ec2-deploy: ensure-terraform
	@test -n "$(AWS_REGION)" || (echo "Falta AWS_REGION"; exit 1)
	terraform -chdir=infra init -upgrade
	terraform -chdir=infra apply -auto-approve \
		-var "aws_region=$(AWS_REGION)" \
		-var "instance_type=$(INSTANCE_TYPE)" \
		-var "instance_profile_name=$(INSTANCE_PROFILE_NAME)" \
		-var "key_name=$(KEY_NAME)" \
		-var "image_uri_api=$(API_IMAGE_URI)" \
		-var "image_uri_frontend=$(FE_IMAGE_URI)" \
		-var "compose_project=$(PROJECT_NAME)" \
		-var "host_port_api=8081" \
		-var "host_port_front=8080" \
		-var "app_port=80"

ec2-destroy: ensure-terraform
	@test -n "$(AWS_REGION)" || (echo "Falta AWS_REGION"; exit 1)
	terraform -chdir=infra destroy -auto-approve \
		-var "aws_region=$(AWS_REGION)"

# ---------- Tooling ----------
ensure-brew:
	@if [ "$$(uname)" = "Darwin" ]; then \
		if ! command -v $(BREW) >/dev/null 2>&1; then \
			echo "Homebrew not found. Install from https://brew.sh first."; exit 1; \
		fi; \
	fi

ensure-terraform: ensure-brew
	@if ! command -v terraform >/dev/null 2>&1; then \
		if [ "$$(uname)" = "Darwin" ]; then \
			echo "Installing Terraform via Homebrew..."; \
				$(BREW) tap hashicorp/tap || true; \
				$(BREW) install hashicorp/tap/terraform;  \
		else \
			echo "Terraform not found. Install: https://developer.hashicorp.com/terraform/install"; exit 1; \
		fi \
	fi
	@terraform -version

terraform-install: ensure-terraform
	@true

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache __pycache__ mlruns artifacts
	find . -name "*.pyc" -delete
