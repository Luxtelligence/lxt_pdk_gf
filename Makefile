install:
	uv sync --extra dev

dev:
	uv sync --extra dev --extra docs
	uv pip install -e .[dev,docs] --config-settings editable_mode=compat
	curl -sf https://raw.githubusercontent.com/doplaydo/pdk-ci-workflow-public/main/templates/.pre-commit-config.yaml -o .pre-commit-config.yaml
	uv run pre-commit clean
	uv run pre-commit install

test:
	uv run pytest -s

test-force: install
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

docs-pdf:
	uv run python .github/write_cells_lnoi400.py
	uv run python .github/write_cells_ltoi300.py
	uv run mkdocs build -f mkdocs-pdf.yml

docs:
	uv run python .github/write_cells_lnoi400.py
	uv run python .github/write_cells_ltoi300.py
	uv run --extra docs zensical build -f docs/zensical.toml

docs-serve:
	uv run python .github/write_cells_lnoi400.py
	uv run python .github/write_cells_ltoi300.py
	uv run --extra docs zensical serve -f docs/zensical.toml -a localhost:8080

update-changelog:
	claude -p "remove links and make a user friendly changelog from @CHANGELOG.md to @docs/changelog.md"

.PHONY: drc drc-sample doc docs docs-pdf build update-changelog
