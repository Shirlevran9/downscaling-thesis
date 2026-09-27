#!/bin/bash
# Stop hook: if this session changed decision-bearing files but added no ADR,
# print one reminder. Never writes, never blocks, never calls a model.
#
# "Decision-bearing" = a file whose change could make a past ADR wrong.
# Deliberately excludes plots/, notebooks/, data/ and the rules themselves,
# which change constantly without representing a decision.

set -u
exit_quietly() { exit 0; }
trap exit_quietly ERR

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0
command -v git >/dev/null 2>&1 || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Debounce: at most one reminder every 30 minutes per project.
STAMP="${TMPDIR:-/tmp}/adr-reminder-$(echo "$PWD" | tr -c 'A-Za-z0-9' '-')"
if [ -f "$STAMP" ]; then
  LAST=$(cat "$STAMP" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  [ $((NOW - LAST)) -lt 1800 ] && exit 0
fi

CHANGED=$(git status --porcelain -- src scripts summaries 2>/dev/null | head -20)
[ -z "$CHANGED" ] && exit 0

# An ADR added or modified in the working tree counts as "recorded".
ADR=$(git status --porcelain -- docs/adr 2>/dev/null | head -5)
[ -n "$ADR" ] && exit 0

date +%s > "$STAMP" 2>/dev/null || true
N=$(printf '%s\n' "$CHANGED" | wc -l | tr -d ' ')
echo "ADR check: ${N} decision-bearing file(s) changed, no ADR added."
echo "  If a real decision was made here, consider docs/adr/. If not, ignore this."
exit 0
