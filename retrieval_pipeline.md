## `retrieve` — The Retrieval Pipeline

**What it does:** Reads a folder of `.md`/`.txt` files, builds a searchable index (embeddings + BM25), and returns the most relevant chunks for your query. No agent loop, no tool calls — it's a **pure information retrieval pipeline**.

**Step-by-step internal flow:**

```
You type: uv run research-agent retrieve --corpus examples/corpus --mode hybrid --top-k 3 --device cpu "why do agents need bounded tool execution?"
```

1. **CLI parsing** — [cli.py:134–177](file:///d:/AI-agents/research-cli-agent/src/research_agent/cli.py#L134-L177): Typer parses `corpus`, `mode=hybrid`, `top_k=3`, `device=cpu`, and the query string.

2. **Build the lab** — [cli.py:67–77](file:///d:/AI-agents/research-cli-agent/src/research_agent/cli.py#L67-L77) → [RetrievalLab.from_corpus()](file:///d:/AI-agents/research-cli-agent/src/research_agent/retrieval/lab.py#L40-L63). This one call triggers the entire pipeline:

   - **Ingestion** — [load_documents()](file:///d:/AI-agents/research-cli-agent/src/research_agent/retrieval/ingestion.py#L26): Walks `examples/corpus/`, reads each `.md`/`.txt` file, creates a `ResearchDocument` per file with a content-hash-based `doc_id`.
   
   - **Chunking** — `WordWindowChunker().split(documents)`: Slides a word-window over each document's text to produce overlapping `Chunk` objects (each with its own deterministic `chunk_id`).
   
   - **Dense index** — `DenseRetriever(chunks, embedder)`: The `SentenceTransformerEmbedder` encodes every chunk's text into a float vector (this is the `Loading weights: 100%` you see). Stores the matrix in memory.
   
   - **BM25 index** — `BM25Retriever(chunks)`: Builds a term-frequency index over the chunks (no neural network, pure keyword matching).
   
   - **Hybrid** — `HybridRetriever(dense, bm25)`: Wraps both, will fuse their results at search time via reciprocal-rank fusion.

3. **Search** — [lab.search(query, mode="hybrid", top_k=3)](file:///d:/AI-agents/research-cli-agent/src/research_agent/retrieval/lab.py#L74-L80):
   - Dispatches to `HybridRetriever.search()`
   - Dense retriever: embeds the query → cosine similarity against all chunk vectors → ranks by score
   - BM25 retriever: tokenizes the query → term-frequency matching → ranks by BM25 score
   - Hybrid fuses both rank lists using reciprocal-rank fusion → returns top 3 `SearchHit`s

4. **Render** — [render_search_hits()](file:///d:/AI-agents/research-cli-agent/src/research_agent/cli.py#L80-L102): For each hit, looks up the full `Chunk` via `lab.chunk_for(chunk_id)` and prints the Rich table.

---

## Key Difference

| | `demo` | `retrieve` |
|---|---|---|
| **Pattern** | Agent loop (decide → act → observe → repeat) | Single-pass retrieval pipeline |
| **Core module** | [loop.py](file:///d:/AI-agents/research-cli-agent/src/research_agent/loop.py) | [lab.py](file:///d:/AI-agents/research-cli-agent/src/research_agent/retrieval/lab.py) |
| **Uses AI model?** | Yes (fake `DemoModel`, but slot for a real LLM) | Yes (embedding model for dense search) |
| **Has a loop?** | Yes, bounded by `max_steps` | No, one-shot query |
| **Uses tools?** | Yes (`WordCountTool`) | No |
| **Reads files?** | No | Yes (corpus ingestion) |
| **Purpose** | Proves the agent runtime architecture works | Proves the retrieval subsystem works |

They're two independent subsystems of your research agent — `demo` validates the **agent loop** pattern, and `retrieve` validates the **retrieval pipeline**. Eventually they'll compose together: the agent loop will use retrieval as one of its tools.