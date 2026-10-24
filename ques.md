# Production Agent Engineering — Interview Questions and Answers

This document contains 20 advanced interview questions derived from the research-agent project.

Each question includes:

- a concise interview answer;
- a deeper engineering explanation;
- how the project demonstrates it;
- common mistakes to avoid.

---

# 1. What is an embedding in an AI agent, and how do you build an embedding subsystem?

## Interview answer

An embedding is a learned numeric vector representing semantic properties of data such as text. In an agent, embeddings are commonly used to retrieve relevant documents, memories, examples, or tool descriptions by comparing a query vector with stored vectors. I build the subsystem behind a provider-neutral `Embedder` interface with separate document and query methods, validate vector dimensions and zero vectors, normalize vectors for cosine similarity, version the embedding model, and evaluate retrieval quality using labeled Recall@k rather than assuming semantic similarity is correct.

## Deeper explanation

An embedding model is a function:

```text
f(text) → vector in Rᵈ
```

Text with related meaning should produce nearby vectors.

For normalized vectors:

```text
cosine similarity = document_vector · query_vector
```

Building embeddings involves more than calling an API:

1. choose a model appropriate for the domain;
2. define chunking;
3. embed corpus chunks;
4. preserve the mapping from vector to source chunk;
5. embed queries using the correct query method;
6. validate dimensions;
7. rank by similarity;
8. evaluate on labeled queries;
9. rebuild or migrate the index when the model changes.

Embeddings are probabilistic representations. They do not prove relevance or factual correctness.

## In this project

`SentenceTransformerEmbedder` implements the `Embedder` protocol.

`DenseRetriever`:

- embeds documents;
- normalizes vectors;
- validates shape;
- rejects zero vectors;
- embeds the query;
- validates dimensions;
- computes cosine rankings.

## Common weak answer

“Embeddings convert text to numbers and then we store them in a vector database.”

That answer misses:

- model versioning;
- query/document distinction;
- normalization;
- chunk identity;
- evaluation;
- migration;
- retrieval failure modes.

---

# 2. Why combine dense retrieval and BM25, and why use Reciprocal Rank Fusion?

## Interview answer

Dense and BM25 retrieval solve complementary problems. Dense retrieval captures semantic similarity, while BM25 is strong for exact terms, identifiers, error codes, and rare technical phrases. Their raw scores are not comparable, so I use Reciprocal Rank Fusion, which combines rank positions using `1 / (k + rank)` instead of adding cosine and BM25 scores. This produces a simple, robust hybrid ranking without requiring score calibration.

## Deeper explanation

Dense retrieval may find:

```text
“recovering workflow progress”
```

for a query containing:

```text
“checkpoint resume”
```

BM25 may find exact strings such as:

```text
CancelledError
gemini-2.5-flash
HTTP 429
```

The raw scales differ:

```text
cosine similarity: 0.84
BM25 score:        15.23
```

Adding them directly gives arbitrary importance to BM25.

RRF uses:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

A document ranked highly by both systems receives contributions from both.

## In this project

`HybridRetriever` retrieves an expanded candidate set from dense and BM25 retrieval, fuses by rank, and applies deterministic chunk-ID tie-breaking.

## Common weak answer

“Hybrid is always more accurate.”

Hybrid often improves robustness, but it must be evaluated. Fusion can also introduce irrelevant candidates or reduce precision.

---

# 3. How do you choose a chunking strategy for RAG?

## Interview answer

Chunking is an information-retrieval design decision, not just text splitting. I choose chunk size and overlap based on document structure, expected query granularity, embedding behavior, and context budget, then evaluate the choice using labeled retrieval cases. Smaller chunks improve precision but can lose context; larger chunks preserve context but add noise. Overlap reduces boundary loss but increases storage and duplication. Chunk IDs must also encode enough version information to prevent stale indexes or evaluation labels after the chunking strategy changes.

## Deeper explanation

A chunk is the retrievable unit. Its design affects:

- what the embedding represents;
- whether evidence is self-contained;
- retrieval precision;
- source citation quality;
- generation context size;
- index cost.

Possible strategies include:

- fixed words;
- fixed tokens;
- sentences;
- paragraphs;
- headings and sections;
- semantic boundaries;
- contextual chunking.

A production chunk should preserve metadata such as:

- document ID;
- chunk index;
- boundaries;
- source path;
- extraction version;
- chunker version;
- content hash.

## In this project

`WordWindowChunker` uses deterministic overlapping word windows. It is intentionally simple and testable.

The current chunk ID uses document ID plus chunk index. A future production revision should also include chunking version or chunk-text identity.

## Common weak answer

“I use 500 tokens with 50-token overlap.”

That is a configuration value, not an engineering justification.

---

# 4. Why is retrieval deterministic infrastructure rather than an agent?

## Interview answer

A retriever maps a query to ranked evidence using a fixed algorithm. It does not choose goals, decide among tools, maintain an action loop, or dynamically decide when to stop. Therefore, retrieval is infrastructure used by an agent, not an agent itself. Keeping it deterministic makes failures measurable and allows us to distinguish retrieval problems from reasoning problems.

## Deeper explanation

A retriever performs:

```text
query → ranked evidence
```

An agent performs:

```text
state → decision → action → observation → next decision
```

Calling every AI-related component an agent destroys useful architectural distinctions.

Retrieval should be testable independently with metrics such as:

- Recall@k;
- precision;
- MRR;
- nDCG;
- latency.

The agent then decides:

- when to retrieve;
- what query to issue;
- whether evidence is sufficient;
- whether another tool is required;
- when to terminate.

## In this project

`RetrievalLab` is deterministic infrastructure.

`SpecialistWorker` is an agent-like bounded loop that uses retrieval through `AsyncSearchTool`.

## Common weak answer

“The retriever is an agent because it uses embeddings.”

Embeddings do not create agency. Agency concerns control flow and action selection.

---

# 5. Why can a correctly executed agent still retrieve incorrect evidence?

## Interview answer

Execution correctness only proves that the system followed its procedure. It does not prove that the query was good, the corpus was complete, the source was trustworthy, the ranking was relevant, or the evidence entailed the answer. I therefore separate operational correctness, retrieval quality, citation integrity, and semantic answer correctness into different tests and evaluation layers.

## Deeper explanation

A retrieval call can be technically successful but return poor evidence because:

- the query omitted important terms;
- the corpus lacks the answer;
- chunking separated necessary context;
- embeddings misunderstood domain language;
- lexical matching preferred a misleading exact term;
- the document is outdated;
- source metadata is wrong;
- top-k is too small;
- fusion produced the wrong ranking.

A tool returning HTTP 200 is not evidence of semantic quality.

## In this project

Operational behavior is tested by unit and integration tests.

Retrieval is evaluated with Recall@k.

Citation mechanics are evaluated by the claim ledger.

Full outputs are evaluated by task graders and regression gates.

Semantic entailment remains an explicit limitation.

## Common weak answer

“If the API call succeeded and the similarity score is high, the evidence is correct.”

Similarity is not truth.

---

# 6. What is the difference between a model proposing a tool call and the runtime executing it?

## Interview answer

A model emits an untrusted proposal. The runtime owns execution authority. It validates the decision type, tool name, argument schema, policy, budget, and side-effect rules before invoking anything. This separation prevents hallucinated tool names, malformed arguments, unauthorized actions, and unbounded loops from becoming real operations.

## Deeper explanation

The model may produce:

```json
{
  "kind": "tool_call",
  "tool_name": "create_appointment",
  "arguments": {
    "date": "tomorrow"
  }
}
```

The runtime must decide:

- does this tool exist?
- is the user authorized?
- are required fields present?
- is the date valid?
- is confirmation required?
- has the tool budget been exhausted?
- does the operation require an idempotency key?

Models reason probabilistically. Execution must be governed deterministically.

## In this project

`ToolCallDecision` represents the proposal.

`ToolRegistry` controls available tools.

Each tool validates arguments using Pydantic.

`AgentLoop` enforces the step limit.

## Common weak answer

“The model calls the function, and the framework handles it.”

Professional systems must explain what the framework validates and what the application still owns.

---

# 7. Why should the agent loop depend on a protocol such as `ModelClient` instead of a vendor client?

## Interview answer

The runtime should depend on the capability it needs, not on one vendor SDK. A `ModelClient` protocol makes the runtime provider-neutral, testable, and stable. Vendor adapters translate SDK responses into application-owned decisions. This allows deterministic tests, provider changes, caching, logging, guardrails, and fallback policies without rewriting the loop.

## Deeper explanation

Direct coupling looks like:

```text
AgentLoop → Gemini SDK
```

Dependency inversion looks like:

```text
AgentLoop → ModelClient ← Gemini adapter
                        ← OpenAI adapter
                        ← deterministic test model
```

The application owns the contract.

Provider-specific concerns remain at the boundary:

- authentication;
- request shape;
- response parsing;
- usage metadata;
- provider errors;
- model identifiers.

## In this project

`AgentLoop` imports `ModelClient`, not a model SDK.

`DemoModel` provides deterministic behavior for offline testing.

## Common weak answer

“It lets us switch providers.”

That is true but incomplete. Testability, error normalization, application-owned schemas, and middleware are equally important.

---

# 8. What should happen when agents fail halfway through a workflow?

## Interview answer

The system should preserve completed work, record the failed operation, classify whether it is retryable, and resume only unfinished work from durable state. It should never silently restart everything or present partial work as complete. A stable run ID and plan fingerprint prevent accidental reuse of checkpoints for a different request.

## Deeper explanation

A durable workflow needs:

- serializable state;
- stable node identity;
- checkpoint storage;
- attempt records;
- terminal statuses;
- retry policy;
- resume semantics;
- idempotent or reconciled side effects.

Suppose three workers run:

```text
papers             → completed
documentation_code → completed
skeptic            → process interruption
```

After restart, only the skeptic branch should continue.

Re-running successful branches wastes resources and may duplicate side effects.

## In this project

The LangGraph runtime stores role results and attempt records in SQLite.

A plan fingerprint is compared on resume.

Successful sibling branches remain checkpointed.

The final result is explicitly classified as completed, partial, or failed.

## Common weak answer

“I would retry the whole workflow.”

That may repeat expensive or irreversible work.

---

# 9. What should happen when a tool call times out?

## Interview answer

A timeout should become a structured result, and retry should occur only if policy authorizes it. The system must distinguish read-only operations from side effects because a timeout does not prove the remote operation failed. For side-effecting tools, retries require idempotency keys or reconciliation. Retries must be bounded and use backoff with jitter.

## Deeper explanation

A timeout means:

```text
the caller did not receive a confirmed result before its deadline
```

It may mean:

- the request never arrived;
- the operation is still running;
- the operation completed but the response was lost;
- the connection stalled;
- the provider is overloaded.

Safe retry depends on operation semantics.

Read-only search is usually safer to retry than:

- payments;
- appointment creation;
- emails;
- database mutation.

## In this project

`asyncio.timeout` bounds each durable worker attempt.

`WorkerExecutionPolicy` decides whether another attempt is authorized.

Backoff is exponential, bounded, jittered, and testable.

## Common weak answer

“I catch the timeout and retry three times.”

That ignores idempotency and can duplicate side effects.

---

# 10. Why is blindly retrying a successful-but-unacknowledged side effect dangerous?

## Interview answer

The remote system may have committed the operation before the response was lost. Retrying can duplicate the action. The professional solution is an idempotency key representing the logical operation, combined with provider-side deduplication or a status lookup before retry. For operations that cannot be idempotent, the workflow may require compensation or human review.

## Deeper explanation

Example:

```text
agent sends CreateAppointment(operation_id=abc)
provider creates appointment
response is lost
agent sees timeout
agent retries without operation ID
provider creates a second appointment
```

With idempotency:

```text
CreateAppointment(idempotency_key=abc)
```

both requests refer to one logical operation.

A durable workflow and an idempotent tool solve different problems:

- durability remembers progress;
- idempotency prevents duplicate effects.

## In this project

The project teaches this principle through durable retries, although the current tools are retrieval-focused and mostly read-only.

A production appointment agent would add idempotency at the tool boundary.

## Common weak answer

“Checkpointing prevents duplication.”

Checkpointing cannot determine what happened inside a remote system before a connection failure.

---

# 11. What is the difference between `max_workers`, `max_concurrency`, decision budget, and tool-call budget?

## Interview answer

`max_workers` limits total planned work. `max_concurrency` limits how many workers may execute simultaneously. A decision budget limits reasoning-loop iterations per worker, while a tool-call budget limits external actions. These controls address different failure modes: queue growth, resource saturation, infinite reasoning, and excessive side effects or cost.

## Deeper explanation

Example:

```text
max_workers = 10
max_concurrency = 3
```

The plan may contain at most ten assignments, but only three run simultaneously.

Per-worker:

```text
max_decisions = 4
max_tool_calls = 2
```

A worker may reason four times but invoke tools only twice.

Budgets should be checked before calls. Checking afterward means the forbidden action already happened.

## In this project

`ResearchBudget` controls total workers and concurrency.

`WorkerBudget` controls decisions and tools.

`ResearchSupervisor` uses a semaphore.

`SpecialistWorker` enforces per-worker budgets.

## Common weak answer

“They are all limits to reduce cost.”

They also protect latency, capacity, termination, safety, and provider stability.

---

# 12. Why does `async def` not automatically make synchronous retrieval concurrent?

## Interview answer

`async def` creates a coroutine interface, but synchronous CPU or blocking I/O inside it still occupies the event-loop thread. To prevent blocking, synchronous work must be moved to a thread or process, replaced with a truly asynchronous implementation, or separately scheduled. Shared model objects may also require locking if thread safety is uncertain.

## Deeper explanation

This still blocks:

```python
async def search():
    return synchronous_embedding_model.encode(text)
```

The function is async in syntax only.

A boundary such as:

```python
await asyncio.to_thread(sync_function, arguments)
```

moves the synchronous call away from the event loop.

For CPU-heavy workloads, a process pool or separate inference service may be more appropriate.

## In this project

`LocalRetrievalTool` uses `asyncio.to_thread`.

It also protects the shared Sentence Transformer with an async lock.

## Common weak answer

“Python async runs functions in parallel.”

Async provides cooperative concurrency. It does not automatically parallelize arbitrary synchronous code.

---

# 13. Why should worker failures become data while cancellation propagates?

## Interview answer

A worker failure is a domain outcome that sibling workers and synthesis may still handle, so it becomes a structured `WorkerResult`. Cancellation is a control signal from the enclosing operation. Swallowing it would violate structured concurrency and could make shutdown appear successful while work continues. Therefore ordinary failures become data, but cancellation is re-raised.

## Deeper explanation

Worker failure:

```text
one source unavailable
```

may allow:

```text
two successful workers + one failed worker → partial report
```

Cancellation means:

```text
the parent task, user, deployment, or timeout has requested termination
```

It is not a normal research outcome.

## In this project

`ResearchSupervisor` catches ordinary exceptions and creates failed results.

It explicitly re-raises `asyncio.CancelledError`.

The evaluation harness follows the same pattern.

## Common weak answer

“We catch every exception so the program never crashes.”

Catching cancellation indiscriminately can make systems impossible to stop correctly.

---

# 14. Why preserve plan order instead of completion order?

## Interview answer

Completion order depends on network and scheduling latency, so it is nondeterministic. Preserving plan order gives stable reports, reproducible tests, consistent caching and comparison, and simpler downstream synthesis. Concurrency should improve latency without changing the semantic ordering contract.

## Deeper explanation

Suppose the plan order is:

```text
papers
documentation_code
skeptic
```

One run may complete:

```text
skeptic, papers, documentation_code
```

Another may complete:

```text
documentation_code, skeptic, papers
```

If output order follows completion, identical logic produces unstable artifacts.

The supervisor can still trace actual scheduling order separately.

## In this project

The supervisor preallocates result positions and writes each result by assignment index.

Trace events retain actual start and finish timing.

## Common weak answer

“Order does not matter because tasks are independent.”

It may not matter for execution, but it matters for deterministic outputs and downstream consumers.

---

# 15. Why should workers share artifacts but not one mutable message history?

## Interview answer

Workers should share immutable, typed artifacts such as assignments, evidence, and final results. They should not concurrently mutate one message history because that creates race conditions, ordering ambiguity, context contamination, and unclear ownership. Each worker should have isolated working state and communicate through explicit outputs that the supervisor merges.

## Deeper explanation

A shared mutable conversation creates questions such as:

- whose message comes first?
- can one worker’s tool result alter another worker’s reasoning?
- what happens during retry?
- how is the state checkpointed?
- how do we reproduce a run?
- who owns truncation?

Artifact-based communication provides:

- stable schemas;
- explicit ownership;
- validation;
- deterministic merge behavior;
- independent retries.

## In this project

Workers receive independent `WorkerAssignment` objects and return immutable `WorkerResult` objects.

The supervisor merges results, not internal message lists.

Durable branches merge role-keyed mappings.

## Common weak answer

“Shared history is bad because it uses more tokens.”

Token use is secondary. The primary problems are consistency, isolation, and ownership.

---

# 16. How should long-term agent memory be designed?

## Interview answer

Long-term memory should be an explicit policy-controlled subsystem, not automatic storage of conversation history. I separate untrusted candidates from approved memories, allow only durable categories, reject secrets and invalid identities, scope every operation to a user, preserve application-owned IDs, and keep the storage provider behind a `MemoryStore` interface. Recall results are personal context, not factual evidence.

## Deeper explanation

Memory design must answer:

- what may be remembered?
- who owns it?
- how long is it retained?
- how can it be corrected or deleted?
- can it contain secrets?
- how are duplicates handled?
- what if old memory conflicts with new information?
- can recalled memory support factual claims?

Memory is both a product feature and a data-governance responsibility.

## In this project

`MemoryPolicy` approves candidates before provider writes.

`MemoryService` coordinates policy and storage.

`SupermemoryStore` translates SDK responses into application models.

Stable IDs and container tags preserve user scope.

## Common weak answer

“I store the conversation in a vector database and retrieve similar messages.”

That confuses history, memory, ownership, durability, and factual evidence.

---

# 17. How do you secure an agent that retrieves live external content?

## Interview answer

I treat all external content as untrusted data. Network access is constrained with HTTPS, host allowlists, content-type checks, response-size limits, timeouts, and no automatically trusted redirects. Source authority is assigned by role. Retrieved evidence retains provenance, receives a content hash, and is scanned for instruction-like attacks. Suspicious evidence is quarantined from synthesis rather than silently deleted.

## Deeper explanation

External content can contain:

- prompt injection;
- hidden instructions;
- malicious HTML;
- oversized responses;
- redirects to untrusted hosts;
- secrets or credential requests;
- unsupported claims.

Security requires layered controls:

1. restrict where requests may go;
2. restrict what content may be accepted;
3. separate data from instructions;
4. preserve provenance;
5. quarantine suspicious content;
6. validate citations;
7. limit tool authority;
8. avoid placing secrets in model context.

## In this project

`bounded_get` enforces network policy.

`RoleSourceRouter` enforces least privilege.

`EvidenceSecurityGate` adds hashes and risk flags.

`ClaimLedgerVerifier` rejects quarantined citations.

## Common weak answer

“I tell the model in the prompt to ignore malicious instructions.”

Prompt instructions alone are not a security boundary.

---

# 18. What is an agent evaluation harness, and how is it different from unit testing?

## Interview answer

An agent evaluation harness executes complete versioned tasks through the full system, often across multiple trials, collects observable trajectories and resource metrics, applies independent graders, and enforces regression gates. Unit tests validate individual components and invariants; the harness measures end-to-end behavior. Both are necessary because a system can have correct components and still produce poor outcomes.

## Deeper explanation

Unit tests answer:

- does BM25 rank correctly?
- does policy reject secrets?
- does retry stop after the limit?
- does checkpoint resume?

The harness answers:

- does the complete system finish the task?
- does it stay within budgets?
- are required sources present?
- are citations mechanically valid?
- are unsafe sources excluded?
- is behavior stable across trials?
- did latency or cost regress?

A harness requires:

- versioned tasks;
- isolated trials;
- system adapters;
- graders;
- metrics;
- artifacts;
- baselines;
- gates.

## In this project

`EvalHarness` runs `ResearchSystem` implementations and returns `EvalReport`.

The same harness evaluates deterministic and Gemini synthesis profiles.

## Common weak answer

“An eval is a list of prompts and expected answers.”

That may be a dataset, but it is not a complete operational harness.

---

# 19. What is the difference between trial pass rate, consistent task rate, and pass@k?

## Interview answer

Trial pass rate measures successful executions across all trials. Consistent task rate measures tasks for which every trial passed, so it captures reliability. Pass@k measures tasks where at least one of k trials succeeded, so it captures whether repeated attempts can find a success. Pass@k can be high while reliability is poor, which is why I report all three.

## Deeper explanation

Suppose one task has three trials:

```text
pass, fail, fail
```

Then:

- trial pass contribution: 1/3;
- consistent-task contribution: 0;
- pass@3 contribution: 1.

Pass@k is useful for systems where multiple attempts are allowed, such as coding or search.

It is dangerous if interpreted as production reliability when users receive only one attempt.

## In this project

`build_summary` groups artifacts by task and calculates:

- `trial_pass_rate`;
- `consistent_task_rate`;
- `pass_at_k`;
- p50 and p95 latency;
- usage and cost.

## Common weak answer

“Pass@k is accuracy after k tries.”

That description misses the probability and reliability interpretation.

---

# 20. If an agent runs smoothly but gives a wrong answer, how do you detect and handle it?

## Interview answer

A smooth execution trace proves operational success, not answer correctness. I evaluate correctness in layers: retrieval quality, source coverage, citation integrity, safety, expected content, semantic entailment, and task-level outcomes. Claims must cite existing eligible evidence with exact quotations, but mechanical citation checks are not enough; semantic support may still require an entailment grader or human review. In production I also monitor user corrections, source freshness, failure clusters, and quality regressions.

## Deeper explanation

A wrong answer can result from:

- missing evidence;
- irrelevant retrieval;
- outdated sources;
- hallucinated synthesis;
- citation mismatch;
- incorrect inference;
- source contradiction;
- insufficient context;
- bad task decomposition.

Detection layers include:

```text
retrieval Recall@k
      ↓
source coverage
      ↓
citation existence
      ↓
exact quote verification
      ↓
quarantine exclusion
      ↓
semantic entailment
      ↓
task-specific correctness
      ↓
human review for high-risk decisions
```

Handling may involve:

- refusing when evidence is insufficient;
- marking uncertainty;
- retrieving again with a revised query;
- comparing independent sources;
- rejecting unsupported claims;
- escalating to a human;
- blocking deployment through a regression gate.

## In this project

The claim ledger verifies:

- evidence existence;
- content hash;
- exact quotation;
- synthesis eligibility.

The evaluation harness applies citation, safety, source, trajectory, budget, and completion graders.

The project explicitly acknowledges that citation integrity does not prove semantic truth.

## Common weak answer

“I add another LLM as a critic.”

An LLM critic can help, but it is another probabilistic component. It needs calibration, tests, observable criteria, and often deterministic checks around it.

---

# Final interview framework

When answering an agent-engineering question, structure your response in this order:

1. Define the concept precisely.
2. State the failure mode it addresses.
3. Identify which layer owns the decision.
4. Describe the deterministic controls.
5. Explain how it is observed and tested.
6. Mention trade-offs and limitations.
7. Give a concrete example from this project.

A strong answer sounds like:

> “The model proposes the search action, but the worker runtime owns execution. It validates the typed decision and enforces decision and tool budgets before calling the provider-neutral search interface. The supervisor bounds concurrency and converts ordinary worker failures into structured results while propagating cancellation. We test the loop deterministically and evaluate complete behavior separately through tasks, trials, graders, trajectories, and regression gates.”

That answer demonstrates architecture, failure awareness, and production judgment—not only library familiarity.