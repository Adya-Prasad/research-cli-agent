# Agent evaluation report

- Dataset: `day7-local-v1`
- Profile: `deterministic`
- System: `multi-agent-deterministic`
- Tasks: 20
- Trials: 20
- Trial pass rate: 1.000
- Consistent task rate: 1.000
- Pass@k: 1.000
- Mean citation integrity: 1.000
- p50 latency: 25.92 ms
- p95 latency: 30.87 ms
- Model calls: 0
- Tool calls: 60
- Input tokens: 0
- Output tokens: 0
- Nominal cost (micro-USD): 0
- Regression gate: PASS

## Trial results

| Task | Trial | Passed | Error |
|---|---:|---:|---|
| runtime-tool-validation | 1 | yes |  |
| runtime-bounded-loop | 1 | yes |  |
| runtime-working-state | 1 | yes |  |
| runtime-model-authority | 1 | yes |  |
| embedding-definition | 1 | yes |  |
| embedding-semantic-neighborhood | 1 | yes |  |
| embedding-normalization | 1 | yes |  |
| chunking-purpose | 1 | yes |  |
| memory-working | 1 | yes |  |
| memory-long-term | 1 | yes |  |
| memory-message-history | 1 | yes |  |
| memory-selective-retention | 1 | yes |  |
| retrieval-vector-index | 1 | yes |  |
| retrieval-cosine-dot-product | 1 | yes |  |
| agent-time-limit | 1 | yes |  |
| agent-cost-limit | 1 | yes |  |
| agent-observation-loop | 1 | yes |  |
| memory-retrieval-on-demand | 1 | yes |  |
| embedding-document-length | 1 | yes |  |
| production-agent-boundaries | 1 | yes |  |
