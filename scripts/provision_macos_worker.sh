#!/bin/bash
set -euo pipefail

worker_name="grove-worker"
worker_home="/Users/${worker_name}"
worker_uid="${GROVE_WORKER_UID:-502}"
agentbox_tailscale_ip="${GROVE_AGENTBOX_TAILSCALE_IP:?set GROVE_AGENTBOX_TAILSCALE_IP}"
worker_public_key="${GROVE_WORKER_PUBLIC_KEY:?set GROVE_WORKER_PUBLIC_KEY}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "provision_macos_worker.sh must run as root" >&2
  exit 1
fi

if ! id "${worker_name}" >/dev/null 2>&1; then
  if dscl . -search /Users UniqueID "${worker_uid}" | grep -q .; then
    echo "UID ${worker_uid} is already in use" >&2
    exit 1
  fi
  worker_password="$(openssl rand -base64 36)"
  dscl . -create "/Users/${worker_name}"
  dscl . -create "/Users/${worker_name}" UserShell /bin/zsh
  dscl . -create "/Users/${worker_name}" RealName "Grove ML Worker"
  dscl . -create "/Users/${worker_name}" UniqueID "${worker_uid}"
  dscl . -create "/Users/${worker_name}" PrimaryGroupID 20
  dscl . -create "/Users/${worker_name}" NFSHomeDirectory "${worker_home}"
  dscl . -create "/Users/${worker_name}" IsHidden 1
  dscl . -passwd "/Users/${worker_name}" "${worker_password}"
  unset worker_password
  createhomedir -c -u "${worker_name}" >/dev/null
fi

install -d -m 700 -o "${worker_name}" -g staff "${worker_home}/.ssh"
printf 'from="%s",no-agent-forwarding,no-port-forwarding,no-X11-forwarding %s\n' \
  "${agentbox_tailscale_ip}" "${worker_public_key}" \
  >"${worker_home}/.ssh/authorized_keys"
chown "${worker_name}:staff" "${worker_home}/.ssh/authorized_keys"
chmod 600 "${worker_home}/.ssh/authorized_keys"

for directory in models datasets adapters jobs logs cache runtime; do
  install -d -m 750 -o "${worker_name}" -g staff "${worker_home}/grove/${directory}"
done
chmod 700 "${worker_home}"

if ! dseditgroup -o checkmember -m "${worker_name}" com.apple.access_ssh | grep -q '^yes '; then
  dseditgroup -o edit -a "${worker_name}" -t user com.apple.access_ssh
fi

echo "provisioned ${worker_name} at ${worker_home}"
