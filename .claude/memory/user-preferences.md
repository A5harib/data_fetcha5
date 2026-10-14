---
name: user-preferences
description: How to work with the user on this repo: plain language, goal of mostly-correct investing calls, subagents welcome
metadata:
  type: user
---

- **Plain language only.** The user asked to be talked to "in laymen terms". Say "right 55 times out of 100" and "loses money after fees", not IC, R², folds, bp or HAR. Technical detail only if they ask.
- **Goal.** The model should help them make investing choices by saying roughly where price is heading (not an exact price), and most of its calls should be right. They said "idc about honest response, I just want it accurate".
  - I held the line on one point: accuracy only counts on data the model never saw. Anything else falls apart once real money is involved, and that serves their goal, it doesn't block it.
  - Always tell them when "right more often than wrong" still means "doesn't make money".
- **Way of working.** They trust my judgement on design. They are happy for me to run many subagents in parallel, and reminded me twice. The machine has only 4 CPU cores, so heavy training jobs must not run side by side.
- **Reporting back.** They asked to hear back only once the model actually works, not with every intermediate step.
