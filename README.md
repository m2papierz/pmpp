# Programming Massively Parallel Processors (PMPP) book chapters code and exercises

For pure C++ chapters:
```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

For Python with CUDA extension chapters:
```bash
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export TORCH_CUDA_ARCH_LIST="8.9"
```

Python code formatting/linting:
```bash
uv run ruff check --fix .
uv run isort .
uv run black .
```
