# free-model-aggregator-02-entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/rebahan149/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/rebahan149/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Token-Proven Best-First Router — a live free-model aggregator over OpenRouter's free tier. One server speaking both client wires (OpenAI Chat Completions and Anthropic Messages, JSON and streaming). Every prompt is dialled to the free model with the highest expected quality for its requirements: published ratings discounted by a Bayesian success estimate that is updated with every gateway outcome, so proven models outrank merely famous ones as the trial's call budget is spent. Transient refusals retry once on the best candidate, then fail over across up to five candidates; a 402/404 retires a model, a gateway-level refusal retires the gateway, and every refusal is reported on the answer's upstream-failures trail. An answer counts only when its text carries the token the prompt asked for.

## Run it

```sh
OPENROUTER_API_KEY=<your key> node --experimental-strip-types main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
