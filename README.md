execution-path
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

uv run pyright src tests
```
