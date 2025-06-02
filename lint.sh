#!/bin/bash

# Fix linting errors using ruff via uv
uv run ruff check --fix .

# Format code using ruff via uv
uv run ruff format .