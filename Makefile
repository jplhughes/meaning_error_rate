.DEFAULT_GOAL=all
includes="*.py"

all: deps check

deps:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt
	[ -d .git ] && pre-commit install || echo "no git repo to install hooks"
check:
	black --check "${includes}"
	flake8 --max-line-length=100 "${includes}"
	pylint "${includes}"
format:
	black "${includes}"
