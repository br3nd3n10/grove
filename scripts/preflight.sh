#!/bin/bash
set -euo pipefail

failure=0

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    printf 'ok      command %s\n' "$1"
  else
    printf 'missing command %s\n' "$1"
    failure=1
  fi
}

check_path() {
  if [[ -e "$1" ]]; then
    printf 'ok      path %s\n' "$1"
  else
    printf 'missing path %s\n' "$1"
    failure=1
  fi
}

worker_identity="${GROVE_WORKER_IDENTITY:-${HOME}/.ssh/grove_worker}"

check_command git
check_command uv
check_command tailscale
check_command ssh
check_command lxc
check_path /dev/kvm
check_path /srv/storage
check_path "${worker_identity}"

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
printf 'info    python %s\n' "${python_version}"
printf 'info    storage %s\n' "$(df -h /srv/storage | awk 'NR == 2 {print $4 " available"}')"

if [[ "${failure}" -eq 0 ]]; then
  uv run grove worker-preflight
fi

exit "${failure}"
