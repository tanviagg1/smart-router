# Smart Router

## Purpose
Intelligent LLM model router that directs requests to the right Ollama model based on task complexity, type, and prompt characteristics. Routes between fast/cheap, capable, and specialized models to balance speed, quality, and cost.

## Stack
- LLM Backend: Ollama (local, no API key needed)
- Models: llama3.2:3b (fast), llama3.1:8b (capable), codellama:7b (code)
- Framework: Python + FastAPI
- Router: Rule-based + LLM-based classification
- Tests: pytest

## Folder Structure
```
smart-router/
├── router/           # Core routing logic
│   ├── classifier.py # Task type classifier
│   ├── registry.py   # Model registry — maps task types to models
│   └── engine.py     # Router engine — orchestrates classify → select → run
├── models/           # Model wrappers
│   └── ollama_client.py
├── api/              # FastAPI app
│   └── main.py
├── prompts/          # Prompt templates as .md files
├── tests/            # pytest unit + integration tests
├── samples/          # Sample prompts for testing
├── CLAUDE.md         # This file
├── PHASES.md         # Build roadmap
└── requirements.txt
```

## How to Run

### Prerequisites
```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull codellama:7b
```

### Run the router on a prompt
```bash
python main.py --prompt "What is recursion?"
python main.py --prompt "Write a binary search in Python"
python main.py --prompt "Explain the CAP theorem in depth"
```

### Run the API
```bash
uvicorn api.main:app --reload
```

### Run tests
```bash
pytest tests/ -v
pytest tests/ -m "not integration"
```

## Key API Endpoints
- `POST /route` — classify prompt and run on the selected model
- `GET /models` — list available models and their capabilities
- `GET /stats` — routing decision history and model usage stats

## Conventions
- Prompts live in `prompts/` as `.md` files
- Router decisions are logged with reasoning
- Each model has a defined capability profile in the registry
- Integration tests that call real Ollama are marked `@pytest.mark.integration`

## Branch Strategy
- `main` — stable
- `feature/phase-1-classifier` — rule-based router + model registry
- `feature/phase-2-llm-router` — LLM-based classifier for smarter routing
- `feature/phase-3-api` — FastAPI + routing stats + history
- `feature/phase-4-ui` — Streamlit dashboard
