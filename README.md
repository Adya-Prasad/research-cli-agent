<div align="center">

<h1>
  <kbd>&nbsp;𝚁𝚎𝚜𝚎𝚊𝚛𝚌𝚑 𝙲𝙻𝙸 𝙰𝚐𝚎𝚗𝚝t&nbsp;</kbd>
</h1>

</div>

### Internal Flow
```
model arguments
    ↓
registry checks tool name
    ↓
Pydantic validates input
    ↓
tool executes
```

### Testing CLI
```
research-agent demo "hello agent"
```


testing commands:
```
uv run pytest -v

uv run ruff check src tests

uv run ruff format src tests

uv run pyright src tests

uv run pyright

uv run python -c "import embedding_adapter"
```
### External research architecture
```
Worker assignment
    ↓
RoleSourceRouter
    ├── papers             → ArxivPaperAdapter
    ├── documentation_code → OfficialDocsAdapter + GitHubCodeAdapter
    └── skeptic            → all adapters with skeptical query terms
    ↓
typed raw source records
    ↓
provenance normalization
    ↓
prompt-injection scanner
    ├── eligible evidence
    └── quarantined evidence retained for audit
    ↓
worker result
    ↓
claim–evidence ledger
    ↓
cited / unsupported / quarantined-citation status
```

### Evidence source completion architecture
```
Paper worker
  ├── arXiv abstract search
  ├── Crossref metadata discovery
  └── Semantic Scholar abstract/citation search
          │
          ▼
    scholarly identity resolver
          │ DOI > arXiv ID > Semantic Scholar ID
          ▼
    cross-provider deduplicator
          │
          ▼
    evidence security gate
          │
          ▼
    bounded federation result

Documentation/code worker
  ├── official documentation targets
  ├── exact GitHub files at immutable refs
  └── controlled allowlisted web pages

Skeptic worker
  └── same controlled providers,
      but receives the skeptic assignment/query
```

### Central Architecture of Evaluation Suite
```
EvalSuite
  └── EvalTask
       └── N isolated Trials
            ├── ResearchSystem
            ├── SystemOutput
            ├── Observable trajectory
            ├── Usage/latency/cost
            └── Multiple Graders
                   ↓
              TrialArtifact
                   ↓
              EvalSummary
                   ↓
             RegressionGate
                   ↓
       report.json + summary.md + trajectories.jsonl
```

### Deterministic offline evaluation
```
uv run research-agent evaluate-agent `
  --profile deterministic `
  --trials 1 `
  --suite evals/agent_tasks.json `
  --corpus examples/corpus `
  --device cpu `
  --output evals/results/deterministic

```
### If the deterministic result is acceptable, establish a baseline:
```
uv run research-agent evaluate-agent `
  --profile deterministic `
  --trials 1 `
  --output evals/results/deterministic `
  --write-baseline evals/baselines/deterministic.json
  ```

### Then test regression comparison:
```
uv run research-agent evaluate-agent `
  --profile deterministic `
  --trials 1 `
  --baseline evals/baselines/deterministic.json `
  --output evals/results/deterministic-regression
```

#### Run a genuine three-call Gemini smoke test when full quota is available:
```
uv run research-agent evaluate-agent `
  --profile gemini `
  --trials 1 `
  --task-limit 3 `
  --model gemini-2.5-flash `
  --output evals/results/gemini-smoke-3
```