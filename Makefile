install:
	uv pip install -e .[dev]
	uv run pre-commit install

dev:
	uv pip install -e .[dev,docs] --config-settings editable_mode=compat
	curl -sf https://raw.githubusercontent.com/doplaydo/pdk-ci-workflow/main/templates/.pre-commit-config.yaml -o .pre-commit-config.yaml
	uv run pre-commit install

test:
	uv run pytest -s

test-force:
	uv run pytest --force-regen -s tests/

test-ports:
	uv run pytest -s tests/test_components.py::test_optical_port_positions

update-pre:
	uv run pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	uv pip install build
	uv run python -m build

docs:
	uv run python .github/write_cells_lnoi400.py
	uv run python .github/write_cells_ltoi300.py
	uv run jb build docs

.PHONY: drc doc docs
