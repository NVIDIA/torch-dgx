# AGENTS.md — AI Agent Guidelines for torch-dgx

## Environment

- `uv` is the package and Python project manager
- Query the current active virtual environment with `printenv VIRTUAL_ENV` and `command -v python`, and use that environment for builds and tests
- Ask for elevated permissions prior to build and test so that CUDA devices will be visible
- Do not run code formatting unless it is immediately prior to commit

## Python Code Style

- Format Python code with **ruff**.

  ```
  ruff format .
  ```

## C++ Code Style

- This is PyTorch extension so the style should be consistent with PyTorch
  source code when possible
- Format C++ source files under `source/` with **clang-format 18** using the
  repository's `.clang-format` style file:

  ```
  find source -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) \
    -exec clang-format-18 -i --style=file {} +
  ```

- Applicable extensions: `.h`, `.cpp`, `.cc`, `.cu`, `.cuh`.

## Building

- Build via
  ```
  uv pip install --no-build-isolation -e . --link-mode=copy
  ```

## Testing

- Run tests via
  ```
  pytest -v -s tests/
  ```
