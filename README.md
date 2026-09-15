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