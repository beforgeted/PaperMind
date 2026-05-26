# PaperMind

RAG-based scientific-paper Q&A backend (FastAPI + MinIO + Kafka + Elasticsearch + Docling + Qwen Embedding).

## Project layout

```
.
├── app/
│   ├── main.py                 # FastAPI entry + lifespan
│   ├── api/v1/                 # HTTP layer (split by use case)
│   │   ├── upload.py           # PDF / multipart upload
│   │   ├── tasks.py            # task status & delete
│   │   ├── query.py            # hybrid retrieval
│   │   ├── answer.py           # routed Q&A + agent + stream
│   │   └── router.py           # aggregates /api/v1/papers/*
│   ├── agents/                 # LangGraph routing + ReAct agent
│   │   └── routing/
│   │       ├── handlers.py     # shared route executors (graph + stream)
│   │       ├── decide.py       # route-only decision
│   │       └── graph.py        # LangGraph pipeline
│   ├── core/                   # config, logging, Pydantic schemas
│   ├── services/               # MinIO, Kafka, ES, embedding, retrieval, …
│   ├── utils/                  # chunking, paper structure parsing
│   └── workers/                # Kafka consumer, index rebuild CLI
├── fronted/                    # Vite + React frontend (note: folder name typo)
├── tests/routing/              # routing eval scripts + fixtures
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

### Service modules (quick reference)

| Module | Role |
|--------|------|
| `retrieval_service` | BM25 + vector kNN + app-side RRF on child chunks |
| `retriever_service` | LangChain `ParentDocumentRetriever` factory (indexing) |
| `vectorstore_service` | Elasticsearch `dense_vector` store |
| `paper_profile_service` | LLM extraction of paper metadata |
| `paper_index_service` | Write paper profiles to ES |
| `paper_search_service` | Paper-level search + answer rendering |

## Quick start

```bash
# 1. infra
cp .env.example .env
docker compose up -d

#    MinIO console : http://localhost:9001  (papermind / papermind123)
#    Elasticsearch : http://localhost:9200
#    Kafka         : localhost:9092

# 2. python deps
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. run API
uvicorn app.main:app --reload
# -> http://localhost:8000/health
# -> http://localhost:8000/docs
```

## Startup commands

```powershell
# Terminal 1: Backend API
conda activate papermind
cd D:\Project\PaperMind
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
# Terminal 2: Kafka consumer worker
conda activate papermind
cd D:\Project\PaperMind
python -m app.workers.consumer
```

```powershell
# Terminal 3: Frontend (Vite)
cd D:\Project\PaperMind\fronted
npm run dev
```

## Recommended startup order

```text
1) Start infrastructure first
   - cd D:\Project\PaperMind
   - docker compose up -d

2) Start backend API and worker
   - Terminal 1: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   - Terminal 2: python -m app.workers.consumer

3) Start frontend last
   - Terminal 3: cd D:\Project\PaperMind\fronted && npm run dev
```

## Routing evaluation

```bash
python tests/evaluate_routing.py --test-set tests/routing/fixtures/routing_test_set.json
python tests/benchmark_routing_latency.py --test-set tests/routing/fixtures/routing_test_set.json
```
