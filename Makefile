install:
	pip install -e .[dev]
	pre-commit install

dev:
	pip install -e .[dev,docs]

test:
	pytest -s

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	python .github/write_cells_docs.py
	jb build docs

.PHONY: drc doc docs
