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

