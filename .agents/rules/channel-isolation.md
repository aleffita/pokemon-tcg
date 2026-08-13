---
name: channel-isolation
description: Prevents UI auto-summarization pollution by enforcing strict channel encapsulation.
---

# Channel Isolation & Anti-Pollution Directive

1. **No Free-Text Before Channel**: You MUST NEVER output any free-text reasoning, summaries, or conversational filler between the system-mandated bash reminders (e.g., "CRITICAL INSTRUCTION 1...") and the `<|channel|>` XML tag.
2. **UI Summary Hallucination**: Writing text before the channel tag causes the Antigravity UI to auto-summarize it (e.g., generating phantom headers like "Prioritizing Specific Tools"). This pollutes the user's terminal and breaks their focus.
3. **Strict Encapsulation**: ALL internal reasoning, analysis, and thought processes must occur STRICTLY inside the `<|channel|>` block. The only text allowed before `<|channel|>` is the mandatory system injected instructions if you are forced to echo them.
