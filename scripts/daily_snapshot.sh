#!/usr/bin/env bash
# Daily chain-snapshot pull into the write-once cache. Free yfinance
# snapshots accumulated now become a validation slice to cross-check the
# paid historical data against later.
#
# MUST run during US market hours (Mon-Fri, 9:30-16:00 ET): after hours
# Yahoo zeroes bid/ask and the snapshot has no tradable mids.
#
# Install (macOS, launchd — survives cron's environment quirks):
#   1. edit scripts/com.gammaexit.snapshot.plist: replace __REPO__ with the
#      absolute repo path, and set Hour/Minute to mid-session in YOUR local tz
#      (e.g. 10:30 PT / 13:30 ET).
#   2. cp scripts/com.gammaexit.snapshot.plist ~/Library/LaunchAgents/
#   3. launchctl load ~/Library/LaunchAgents/com.gammaexit.snapshot.plist
# Or plain cron (runs only if the machine is awake):
#   30 10 * * 1-5 /path/to/repo/scripts/daily_snapshot.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG="logs/snapshot_$(date +%Y%m%d).log"
mkdir -p logs

status=0
# universe comes from the experiment config — the single source of truth
for t in $($PY -c "from gamma_exit.config import load_config; print(' '.join(load_config().data.universe))"); do
  echo "=== $t @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"$LOG"
  if ! $PY -m gamma_exit.data.snapshot "$t" --max-expiries 6 >>"$LOG" 2>&1; then
    echo "snapshot FAILED for $t (see $LOG)" >&2
    status=1
  fi
done
exit $status
