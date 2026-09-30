---
status: accepted
---

# Path-routed rules fire on file access, not on intent

Guidance in `.claude/rules/*.md` loads when a matching path is touched. That
suits a rule about editing a file. It fits a rule about *performing a task*
only indirectly: when the user asks for a paper to be summarised, nothing has
been touched yet, so the rule is not yet in context.

This is why the PDFs and their summaries share one directory tree. An agent
summarising a paper must read the PDF from `papers/<topic>/` and write the
summary beside it, so the `papers/**` rule fires either way — on the read if the
work starts there, on the write if it does not.

Verified empirically before relying on it: a probe rule scoped to one file was
injected immediately after that file was read, and a second test confirmed both
glob and exact-path patterns match. Note the timing is "when a matching path is
named or touched", not strictly on first access — a request that names a path
loads the rule earlier.

The alternative was a skill, which triggers on a described task rather than a
path. That was rejected to keep one mechanism in this repository, at the cost of
the rule arriving a moment later than ideal.
