# Agent runtime

An AI model proposes actions, but the runtime validates tool names and arguments before execution. A bounded loop enforces step, time, and cost limits. The runtime records observations and returns them to the model as explicit working state.