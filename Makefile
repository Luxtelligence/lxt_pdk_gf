install:
	pip install -e .[dev]
	pre-commit install

dev:
	pip install -e .[dev,docs] --config-settings editable_mode=compat

test:
	pytest -s

test-force:
	pytest --force-regen -s tests/

test-ports:
	pytest -s tests/test_components.py::test_optical_port_positions

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	python .github/write_cells_lnoi400.py
	python .github/write_cells_ltoi300.py
	jb build docs

.PHONY: drc doc docs
