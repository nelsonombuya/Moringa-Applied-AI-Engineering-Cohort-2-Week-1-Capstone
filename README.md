# Moringa Applied AI Engineering (Cohort 2)

Labs, exercises, and projects from my **Applied AI Engineering** course at [Moringa School](https://moringaschool.com), Cohort 2. The course is part-time, remote, and runs about 14 instructional weeks. It started on 12 October 2026 and ends on January 29 2026 with a capstone presentation and NVIDIA certification.

Each notebook works as a lab report and as executable code. It records the problem, the approach, the results, and the reasoning.

## Repository Layout

```
Applied AI Engineering (Cohort 2)/
├── .venv/                  # Shared virtual environment (not committed)
├── .env                    # API keys (not committed)
├── pyproject.toml          # Single shared dependency manifest
├── uv.lock                 # Locked dependency versions
├── week_1/
│   └── week_1_monday.ipynb # Files added as the course progresses
├── week_2/                 # Added as the course progresses
└── ...
```

All weeks share **one** `uv` project and **one** virtual environment at the repo root. Week folders are plain folders, not separate projects or workspace members.

## Tech Stack

- Python 3.14 or later, managed with [uv](https://docs.astral.sh/uv/)
- JupyterLab for notebooks
- `openai` and `openrouter` for LLM access
- `httpx` for HTTP and API calls
- `numpy` for numerical work
- `python-dotenv` for loading secrets from `.env`

## Getting Started

### 1. Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- An API key for your LLM provider (OpenAI and/or OpenRouter)

### 2. Clone and install

```bash
git clone [https://github.com/nelsonombuya/Moringa-Applied-AI-Engineering-Cohort-2.git](https://github.com/nelsonombuya/Moringa-Applied-AI-Engineering-Cohort-2.git)
cd Moringa-Applied-AI-Engineering-Cohort-2
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
```

> `.env` is git-ignored. Never commit API keys.

### 4. Launch JupyterLab

```bash
uv run jupyter lab
```

Open a notebook from a `week_*` folder and run it top to bottom.

## Working Conventions

- Run `uv init` only once, at the repo root. Never run it inside a week folder.
- Add packages with `uv add <package>` so `pyproject.toml` and `uv.lock` stay in sync.
- Each notebook should run top to bottom without errors.
- Each notebook should state the problem and objectives, justify its decisions, and interpret its results.
- Secrets live in `.env` and are loaded with `python-dotenv`.

## About

Built by **Nelson Ombuya**, a software and platform engineer based in Kenya. I focus on backend systems, Linux infrastructure, and LLM applications.

- GitHub: [@nelsonombuya](https://github.com/nelsonombuya)
- LinkedIn: [@nelsonombuya](https://linkedin.com/in/nelsonombuya)

## License
For Educational Use Only
