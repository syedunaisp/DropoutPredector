## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for **ANY non-trivial task** (3+ steps or architectural decisions).
- Write detailed specs upfront. Reduce ambiguity before writing a single line of code.
- Use plan mode for **verification steps**, not just building.
- If something goes sideways mid-task — **STOP. Re-plan immediately.** Do not keep pushing through uncertainty.
- Never start implementation without a written, checked plan.

### 2. Subagent Strategy

- Use subagents liberally to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems — **throw more compute at it via subagents.**
- One task per subagent. Focused execution only.

### 3. Self-Improvement Loop

- After **ANY correction from the user**: update `tasks/lessons.md` with the pattern.
- Write rules for yourself that **prevent the same mistake from happening again.**
- Ruthlessly iterate on these lessons until the mistake rate drops to near-zero.
- Review `tasks/lessons.md` at the **start of every session** for this project.

### 4. Verification Before Done

- **Never mark a task complete without proving it works.**
- Diff behavior between `main` and your changes when relevant.
- Before submitting: ask yourself — *"Would a staff engineer approve this?"*
- Run tests. Check logs. Demonstrate correctness. Evidence, not assumption.

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask *"Is there a more elegant way?"*
- If a fix feels hacky: *"Knowing everything I know now — implement the elegant solution."*
- Skip this for simple, obvious fixes. Do **not** over-engineer.
- Challenge your own work before presenting it. First draft is rarely the best draft.

### 6. Autonomous Bug Fixing

- When given a bug report: **just fix it.** Do not ask for hand-holding.
- Point at logs, errors, and failing tests — then resolve them.
- Zero context switching required from the user.
- Go fix failing CI tests without being told how.

---

## Task Management

Every non-trivial task follows this lifecycle:

1. **Plan First** — Write the plan to `tasks/todo.md` with checkable items.
2. **Verify Plan** — Check in with the user before starting implementation.
3. **Track Progress** — Mark items complete as you go. Keep `todo.md` live.
4. **Explain Changes** — Provide a high-level summary at each step. No silent commits.
5. **Document Results** — Add a review/outcome section to `tasks/todo.md` when done.
6. **Capture Lessons** — Update `tasks/lessons.md` after any correction or post-mortem.

---

## Core Principles

- **Simplicity First** — Make every change as simple as possible. Impact minimal code.
- **No Laziness** — Find root causes. No temporary fixes. Hold yourself to senior developer standards.
- **Minimal Impact** — Changes should only touch what's necessary. No side effects. No introduced bugs.
- **No Magic** — Never do something the user didn't ask for. Surprise changes break trust.
- **Fail Loudly** — If you're blocked or uncertain, surface it immediately. Don't silently produce wrong output.

---

## File Conventions

| File | Purpose |
|------|---------|
| `tasks/todo.md` | Active task plan with checkboxes |
| `tasks/lessons.md` | Accumulated self-improvement rules |
| `tasks/decisions.md` | Architectural decisions and reasoning log |

---

## What Good Looks Like

- Plans are written before code is written.
- Every shipped change has evidence it works.
- Bugs are fixed autonomously, not escalated unnecessarily.
- Mistakes are logged and never repeated.
- Code is clean, minimal, and ready for a staff engineer review.

---

*Last updated: April 2026*