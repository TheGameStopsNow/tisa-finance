# Contributing to TISA

Thank you for your interest in contributing to TISA! We welcome contributions from the community.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/TISA.git
    cd TISA
    ```
3.  **Create a virtual environment** and install dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    ```

## Development Workflow

1.  Create a new branch for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  Make your changes.
3.  Run tests to ensure everything is working:
    ```bash
    pytest
    ```
4.  Commit your changes with clear messages.
5.  Push to your fork and submit a **Pull Request**.

## Code Style

- We use `black` for code formatting.
- We use `isort` for import sorting.
- Please ensure your code is typed and passes `mypy` checks where applicable.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with details about the problem or suggestion.

## License

By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE).
