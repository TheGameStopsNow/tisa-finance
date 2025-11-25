# Publishing TISA to PyPI

This guide explains how to publish the TISA package to PyPI so it can be installed via `pip install tisa-finance`.

## Prerequisites

1.  **PyPI Account**: Create an account at [pypi.org](https://pypi.org/).
2.  **API Token**: Go to Account Settings > API Tokens and create a new token with "Upload to project" scope (or "Entire account" for the first upload).

## 1. Build the Package

The package has already been built in the `dist/` directory, but if you need to rebuild:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build
```

This creates two files in `dist/`:
- `tisa_finance-0.2.0.tar.gz` (Source distribution)
- `tisa_finance-0.2.0-py3-none-any.whl` (Built distribution)

## 2. Check the Package

Verify the package description will render correctly on PyPI:

```bash
twine check dist/*
```

## 3. Upload to TestPyPI (Optional but Recommended)

TestPyPI is a separate instance for testing.

1.  Create an account at [test.pypi.org](https://test.pypi.org/).
2.  Upload:

```bash
twine upload --repository testpypi dist/*
```

3.  Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps tisa-finance
```

## 4. Upload to PyPI (Production)

When ready to release to the world:

```bash
twine upload dist/*
```

You will be prompted for your username (`__token__`) and your API token as the password.

## 5. Installation

Once published, anyone can install it via:

```bash
pip install tisa-finance
```

## Automating with GitHub Actions

The included `.github/workflows/ci.yml` runs tests. You can add a release workflow to publish automatically on tag creation.

See [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) for details.
