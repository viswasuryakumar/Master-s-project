# Nikhil Dupally: backend support and API tests

This change adds the backend support modules and API tests for the log
observatory web prototype (`web/`). They were refactored for readability before
being added here; behavior is unchanged.

## Files in this contribution

- `web/backend/engine.py`: deterministic synthetic log generation, line parsing,
  pattern-based classification and local investigation answers. Public names
  (`PROFILES`, `SCENARIOS`, `parse`, `classify`, `generate`, `explain`) are
  unchanged because `app.py` and `comparison.py` import them.
- `web/backend/recommendations.py`: evidence-based investigation checklists.
  Rules are a named tuple and the limits are named constants.
- `web/backend/replay_agent.py`: independent source worker that forwards replay
  assignments; the request logic is split out of the polling loop.
- `web/tests/test_app.py` and `web/tests/test_recommendations.py`: API and
  recommendation tests.

## Not included

- `web/backend/legacy_app.py`: historical code, retained as is and not started by
  the application. No new work is needed unless the team plans to maintain it.
- `ft_model_BGL/` and `ft_model_HDFS/`: fine-tuned adapter weights that come from
  the upstream LogLLM repository (guanwei49/LogLLM, MIT licensed). They are not
  part of this contribution, and the web prototype does not load them.

## Dependencies

These files import `web/backend/app.py` and `web/backend/comparison.py`, and
`test_recommendations.py` reuses fixtures from `web/tests/test_comparison.py`.
Those files belong to other contributors and are not in this change, so the tests
run only once the rest of `web/` is present.

## Validation

- Differential test: the original and refactored `engine` and `recommendations`
  modules produced identical output on 273,816 comparisons across all four
  profiles (generation with varied counts, namespaces and epochs; parsing;
  classification; every question category for `explain`; and `recommend` on
  generated, malformed and randomized messages).
- The unchanged original tests and the refactored tests each pass: 34 tests in
  the full `web/tests` suite, including tests from other contributors.
- Deliberately breaking `MAX_ISSUES`, a rule key and the classifier result made
  the refactored tests fail, so they still detect regressions.

## Scope

The engine is a deterministic simulation. It generates synthetic logs and
classifies them with fixed failure patterns; it does no model inference, and its
scores are fixed illustrative values, not calibrated probabilities. This change
does not claim original authorship of LogLLM, BERT or LLaMA.
