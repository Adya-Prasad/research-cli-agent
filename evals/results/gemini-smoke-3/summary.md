# Agent evaluation report

- Dataset: `day7-local-v1-subset-first-3`
- Profile: `gemini`
- System: `multi-agent-gemini-synthesis`
- Tasks: 3
- Trials: 3
- Trial pass rate: 0.333
- Consistent task rate: 0.333
- Pass@k: 0.333
- Mean citation integrity: 1.000
- p50 latency: 7061.44 ms
- p95 latency: 7061.44 ms
- Model calls: 1
- Tool calls: 3
- Input tokens: 497
- Output tokens: 148
- Nominal cost (micro-USD): 519
- Regression gate: FAIL

## Trial results

| Task | Trial | Passed | Error |
|---|---:|---:|---|
| runtime-tool-validation | 1 | no | ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}} |
| runtime-bounded-loop | 1 | yes |  |
| runtime-working-state | 1 | no | ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}} |

## Regression failures

- trial pass rate below absolute threshold
- consistent task rate below absolute threshold
