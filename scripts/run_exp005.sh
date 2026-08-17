#!/usr/bin/env bash
# EXP-005: second-cycle coexistence. The complete, exact experiment procedure.
#
# Usage:
#   scripts/run_exp005.sh <db> <report>
#
#   <db>      fresh SQLite evidence database, e.g. /srv/storage/grove/grove-exp005.db
#   <report>  JSON report path, e.g. /srv/storage/grove/evaluations/exp005.json
#
# One invocation runs both growth cycles against one store: the first cycle
# admits the escaped_path expert, and the second cycle -- with that expert
# still admitted, deployed and routable, no reset in between -- captures,
# trains and gates the path_restructure expert, recording multi-expert
# coexistence measurements at every checkpoint.
#
# Exit codes, propagated unchanged:
#   real-cycle: 0 ran to completion, 2 setup refused / unusable (propagates).
#   checker:    0 all sealed rules held, 1 a predeclared rule failed (a
#               result), 2 the run cannot be judged.
set -u

if [ "$#" -ne 2 ]; then
  echo "usage: scripts/run_exp005.sh <db> <report>" >&2
  exit 2
fi

DB="$1"
REPORT="$2"
SPEC="experiments/EXP-005-second-cycle-coexistence.json"

uv run grove --db "$DB" real-cycle --reset --cycles 2 \
  --spec "$SPEC" \
  --arm primary \
  --correction-source canonical \
  --report "$REPORT"
run_status=$?
if [ "$run_status" -ne 0 ]; then
  # A refused or failed run grades nothing; its exit code -- exit 2
  # in particular -- must reach the caller unchanged.
  exit "$run_status"
fi

uv run python scripts/check_experiment_spec.py \
  --spec "$SPEC" \
  --report "$REPORT"
