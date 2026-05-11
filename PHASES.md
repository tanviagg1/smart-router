# Smart Router — Build Phases

---

## Phase 1: Rule-Based Router + Model Registry
**Branch:** `feature/phase-1-classifier`

### What gets built
- `ModelRegistry` — maps task types to Ollama models with capability profiles
- `RuleBasedClassifier` — classifies prompts by keyword, length, and pattern matching
- `RouterEngine` — orchestrates classify → select model → run → return response
- `OllamaClient` — thin wrapper around ollama Python SDK
- CLI (`main.py`) — run a prompt through the router
- Unit tests for classifier and registry

### Routing logic
| Task Type   | Signals                              | Model         |
|-------------|--------------------------------------|---------------|
| simple_qa   | short prompt, factual question       | llama3.2:3b   |
| code        | keywords: def, function, algorithm   | codellama:7b  |
| complex     | long prompt, reasoning, explanation  | llama3.1:8b   |
| default     | anything else                        | llama3.1:8b   |

### What you learn
- Classification as a routing primitive
- Model capability profiling
- Rule-based vs. ML-based decision making

### Definition of Done
- `python main.py --prompt "..."` routes correctly
- Short factual questions hit llama3.2:3b
- Code prompts hit codellama:7b
- All unit tests pass

---

## Phase 2: LLM-Based Classifier
**Branch:** `feature/phase-2-llm-router`

### What gets built
- `LLMClassifier` — uses llama3.2:3b to classify task type (fast, cheap)
- Fallback: if LLM classifier fails → fall back to rule-based
- Confidence scoring: if confidence < threshold → escalate to larger model
- Routing explanation: every decision logged with reasoning

### What you learn
- Using a small model to route to a large model
- Confidence thresholds and escalation
- LLM-as-judge pattern

### Definition of Done
- LLM classifier outperforms rule-based on ambiguous prompts
- Fallback works when classifier model is unavailable
- Every routing decision includes a reason

---

## Phase 3: FastAPI + Stats
**Branch:** `feature/phase-3-api`

### What gets built
- `POST /route` — classify and run prompt, return response + routing metadata
- `GET /models` — list models with capability profiles
- `GET /stats` — model usage counts, avg response times, routing distribution
- In-memory routing history (last 100 decisions)

### What you learn
- API design for AI routing systems
- Collecting and exposing operational metrics
- Request/response modeling with Pydantic

### Definition of Done
- `/route` returns response + which model was used + why
- `/stats` shows accurate routing distribution

---

## Phase 4: Streamlit Dashboard
**Branch:** `feature/phase-4-ui`

### What gets built
- Prompt input + model selector override
- Real-time routing decision display ("sent to codellama because: code detected")
- Model usage pie chart
- Response time comparison across models
- Routing history table

### Definition of Done
- Full routing flow visible in browser
- Charts update after each request
