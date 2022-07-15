.DEFAULT_GOAL=all

all: deps check

deps:
	pip install --upgrade pip
	pip install -r requirements.txt
	[ -d .git ] && pre-commit install || echo "no git repo to install hooks"
check:
	black --check .
	flake8 --max-line-length=120 .
	pylint **/*.py
format:
	black .
	isort **/*.py
unittest:
	python -m pytest unittests
