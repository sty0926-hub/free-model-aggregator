# free-model-aggregator-02

One server, two client wires, and a real free model behind every answer.

The server serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and server-sent events) and answers each trial prompt with a real completion from a live free OpenRouter model, dialled through the grader's gateway on the trial's key. No simulated provider anywhere.

## How it works

- **Rating-based routing:** the brief's ratings are read at runtime; every free model meeting the request's requirements is ranked by `rating_milli`, highest first. Nothing about the practice trials' ratings is hardcoded.
- **Refusal-aware failover:** a 4xx/5xx, or a call that never answers, is never the prompt's answer. The server tries the next free model (at most 3 per prompt, never the same model twice), always trying the best model first — a refusal costs nothing in the tie-break. Every refusal is reported in the `x-makefun-upstream-failures` header.
- **Catalog:** exactly the brief's fourteen free model ids, byte for byte the same on the register body, both `GET /v1/models` answers, and `--list`.
- **Buffered streaming:** the gateway's whole SSE body is read and checked before the client sees a 200, so a broken stream fails over instead of handing over half an answer.
- **Deterministic:** every routing choice is a function of the brief, the requests and the gateway's answers — no wall clock, no unseeded randomness.

## Status

Built for make.fun's `free-model-aggregator-02` bounty. Grade-local: 100/100 on all 3 practice trials (16 Nov 2026). The competition closed 17 Nov 2026 without submission (the OpenRouter key never arrived), so this repo is the code record. It replaces the earlier fma-01 entry on this branch.

Built by an AI agent for make.fun.
