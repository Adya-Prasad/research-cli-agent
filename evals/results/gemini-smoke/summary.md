# Agent evaluation report

- Dataset: `day7-local-v1`
- Profile: `gemini`
- System: `multi-agent-gemini-synthesis`
- Tasks: 20
- Trials: 20
- Trial pass rate: 0.300
- Consistent task rate: 0.300
- Pass@k: 0.300
- Mean citation integrity: 1.000
- p50 latency: 5519.70 ms
- p95 latency: 9702.33 ms
- Model calls: 6
- Tool calls: 18
- Input tokens: 2987
- Output tokens: 1739
- Nominal cost (micro-USD): 5243
- Regression gate: FAIL

## Trial results

| Task | Trial | Passed | Error |
|---|---:|---:|---|
| runtime-tool-validation | 1 | yes |  |
| runtime-bounded-loop | 1 | yes |  |
| runtime-working-state | 1 | yes |  |
| runtime-model-authority | 1 | yes |  |
| embedding-definition | 1 | yes |  |
| embedding-semantic-neighborhood | 1 | yes |  |
| embedding-normalization | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| chunking-purpose | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| memory-working | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| memory-long-term | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| memory-message-history | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| memory-selective-retention | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| retrieval-vector-index | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| retrieval-cosine-dot-product | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| agent-time-limit | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| agent-cost-limit | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| agent-observation-loop | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| memory-retrieval-on-demand | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| embedding-document-length | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |
| production-agent-boundaries | 1 | no | ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google. |

## Regression failures

- trial pass rate below absolute threshold
- consistent task rate below absolute threshold
