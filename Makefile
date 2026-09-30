format:
	uv run ruff format

lint:
	uv run ruff check

fix:
	uv run ruff check --fix

stubgen:
	uv run stubgen -m placo -o stubs

check:
	uv run ty check
