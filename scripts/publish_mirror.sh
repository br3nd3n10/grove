#!/bin/bash
set -euo pipefail

# Resolve the source from this script so callers can run it from any directory.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
repo_parent="$(dirname -- "${repo_root}")"
mirror_override="${GROVE_MIRROR_ROOT-}"
mirror_root="${GROVE_MIRROR_ROOT:-${repo_parent}/grove-public}"

usage() {
  printf 'usage: %s [--force] [-m message]\n' "$0" >&2
  printf '  default: incremental update — one new commit on the existing mirror\n' >&2
  printf '  --force: destroy and rebuild the mirror as a single fresh commit\n' >&2
  printf '  -m:      commit message for the mirror commit\n' >&2
  exit 2
}

force_rebuild=0
commit_message=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --force) force_rebuild=1; shift ;;
    -m)
      [[ "$#" -ge 2 ]] || usage
      commit_message="$2"; shift 2 ;;
    *) usage ;;
  esac
done

# Keep deletion narrowly scoped: only the expected sibling directory may be removed.
if [[ "$(basename -- "${mirror_root}")" != "grove-public" || "${mirror_root}" != /* ]]; then
  printf 'refusing unexpected mirror path: %s\n' "${mirror_root}" >&2
  exit 1
fi
if [[ -z "${mirror_override}" && "$(dirname -- "${mirror_root}")" != "${repo_parent}" ]]; then
  printf 'refusing unexpected mirror path: %s\n' "${mirror_root}" >&2
  exit 1
fi

# Decide mode. Incremental requires an existing mirror git repo with a clean
# worktree; anything else needs an explicit --force rebuild.
incremental=0
if [[ -e "${mirror_root}" ]]; then
  if [[ ! -d "${mirror_root}" || -L "${mirror_root}" ]]; then
    printf 'refusing to touch non-directory mirror target: %s\n' "${mirror_root}" >&2
    exit 1
  fi
  if [[ "${force_rebuild}" -eq 1 ]]; then
    existing_remotes=""
    if [[ -e "${mirror_root}/.git" ]]; then
      existing_remotes="$(git -C "${mirror_root}" remote 2>/dev/null || true)"
    fi
    if [[ -n "${existing_remotes}" ]]; then
      remote_names="${existing_remotes//$'\n'/, }"
      printf 'warning: --force will remove mirror remote(s): %s\n' "${remote_names}" >&2
    fi
    rm -rf -- "${mirror_root}"
    mkdir -- "${mirror_root}"
  else
    if [[ ! -e "${mirror_root}/.git" ]]; then
      printf 'mirror exists but is not a git repository: %s\n' "${mirror_root}" >&2
      printf 'rerun with --force to rebuild it from scratch.\n' >&2
      exit 1
    fi
    if [[ -n "$(git -C "${mirror_root}" status --porcelain)" ]]; then
      printf 'mirror worktree is not clean: %s\n' "${mirror_root}" >&2
      exit 1
    fi
    incremental=1
  fi
else
  mkdir -- "${mirror_root}"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' EXIT
tracked_list="${tmp_dir}/tracked-files"
experiment_list="${tmp_dir}/experiment-files"
mirror_experiment_list="${tmp_dir}/mirror-experiment-files"

# The index is the allow-list. This prevents ignored caches and runtime state
# from entering the mirror even when they exist beside the checkout.
git -C "${repo_root}" ls-files -z -- > "${tracked_list}"

# In incremental mode, clear the previous published tree first so deletions in
# the source propagate to the mirror. Only tracked files are removed; the .git
# directory and any operator files are untouched.
if [[ "${incremental}" -eq 1 ]]; then
  git -C "${mirror_root}" ls-files -z | while IFS= read -r -d '' path; do
    rm -f -- "${mirror_root}/${path}"
  done
  find "${mirror_root}" -mindepth 1 -type d -not -path "${mirror_root}/.git*" -empty -delete
fi

copied_count=0
while IFS= read -r -d '' path; do
  case "${path}" in
    /*|../*|*/../*)
      printf 'refusing unsafe tracked path: %s\n' "${path}" >&2
      exit 1
      ;;
    .atomic/*|.venv/*|.pytest_cache/*|.ruff_cache/*|__pycache__/*|*/__pycache__/*)
      printf 'refusing forbidden runtime path in index: %s\n' "${path}" >&2
      exit 1
      ;;
  esac
  source_path="${repo_root}/${path}"
  destination_path="${mirror_root}/${path}"
  if [[ ! -f "${source_path}" ]]; then
    printf 'refusing non-regular tracked path: %s\n' "${path}" >&2
    exit 1
  fi
  mkdir -p -- "$(dirname -- "${destination_path}")"
  # -p carries executable bits and other file metadata into the mirror.
  cp -p -- "${source_path}" "${destination_path}"
  ((copied_count += 1))
done < "${tracked_list}"
printf 'files copied: %s\n' "${copied_count}"

if [[ "${incremental}" -eq 0 ]]; then
  git -C "${mirror_root}" init >/dev/null
fi

# The operator path and legacy search terms live in an untracked private file,
# so the public copy of this utility contains no private identifiers — not even
# as reconstructible fragments.
terms_file="${GROVE_SCRUB_TERMS:-${repo_root}/.scrub-terms.env}"
if [[ ! -f "${terms_file}" ]]; then
  printf 'missing scrub terms file: %s\n' "${terms_file}" >&2
  printf 'it must define legacy_key, legacy_host, legacy_name, operator_home, operator_key.\n' >&2
  exit 1
fi
if git -C "${repo_root}" ls-files --error-unmatch -- "${terms_file}" >/dev/null 2>&1; then
  printf 'scrub terms file must not be tracked: %s\n' "${terms_file}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${terms_file}"
for term_var in legacy_key legacy_host legacy_name operator_home operator_key; do
  if [[ -z "${!term_var-}" ]]; then
    printf 'scrub terms file does not define %s\n' "${term_var}" >&2
    exit 1
  fi
done

# Scrub only prose. Sealed experiment specifications are deliberately excluded
# and are compared byte-for-byte below.
python3 - "${mirror_root}" "${operator_key}" "${operator_home}" "${legacy_key}" "${legacy_host}" "${legacy_name}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
replacements = (
    (sys.argv[2].encode(), b"~/.ssh/grove_worker"),
    (sys.argv[3].encode(), b"~"),
    (sys.argv[4].encode(), b"grove_worker"),
    (sys.argv[5].encode(), b"grove-worker-1"),
    (sys.argv[6].encode(), b"grove-worker-1"),
)
# Alternatives are tried in order at the first matching position, not by
# longest match, so the full key path must precede the host and catch-all.
pattern = re.compile(
    b"|".join(re.escape(old) for old, _ in replacements), re.IGNORECASE
)
replacement_by_term = {old.lower(): new for old, new in replacements}

for path in root.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    if relative.parts and relative.parts[0] in {"experiments", ".git"}:
        continue
    if path.suffix.lower() not in {".md", ".txt"}:
        continue
    original = path.read_bytes()
    scrubbed = pattern.sub(
        lambda match: replacement_by_term[match.group().lower()], original
    )
    if scrubbed != original:
        path.write_bytes(scrubbed)
PY

# Stage the scrubbed allow-list before comparing destination paths; no commit is
# made until every integrity and secret check below has passed.
git -C "${mirror_root}" add --all --force

# Verify every sealed specification copied from the source index unchanged.
git -C "${repo_root}" ls-files -z -- experiments/ > "${experiment_list}"
git -C "${mirror_root}" ls-files -z -- experiments/ > "${mirror_experiment_list}"
if ! cmp -s "${experiment_list}" "${mirror_experiment_list}"; then
  printf 'sealed experiment file lists differ\n' >&2
  exit 1
fi

experiment_count=0
while IFS= read -r -d '' path; do
  if ! cmp -s "${repo_root}/${path}" "${mirror_root}/${path}"; then
    printf 'sealed experiment changed: %s\n' "${path}" >&2
    exit 1
  fi
  ((experiment_count += 1))
done < "${experiment_list}"
printf 'sealed experiment files byte-identical: %s\n' "${experiment_count}"

# Preserve executable bits on tracked scripts; cp -p above should make this a
# check rather than a repair, so a mode regression fails loudly.
while IFS= read -r -d '' path; do
  if [[ -x "${repo_root}/${path}" && ! -x "${mirror_root}/${path}" ]]; then
    printf 'script lost executable bit: %s\n' "${path}" >&2
    exit 1
  fi
done < <(git -C "${repo_root}" ls-files -z -- scripts/)

# Secret checks run on the staged tree before anything is committed.
legacy_hits="$(grep -RniI --exclude-dir=.git -- "${legacy_name}" "${mirror_root}" || true)"
legacy_hit_count=0
if [[ -n "${legacy_hits}" ]]; then
  legacy_hit_count="$(printf '%s\n' "${legacy_hits}" | wc -l | tr -d '[:space:]')"
fi
printf 'case-insensitive legacy-name grep hits: %s\n' "${legacy_hit_count}"
if [[ "${legacy_hit_count}" -ne 0 ]]; then
  printf '%s\n' "${legacy_hits}" >&2
  exit 1
fi

home_hits="$(grep -RnIF --exclude-dir=.git -- "${operator_home}" "${mirror_root}" || true)"
home_hit_count=0
if [[ -n "${home_hits}" ]]; then
  home_hit_count="$(printf '%s\n' "${home_hits}" | wc -l | tr -d '[:space:]')"
fi
printf 'operator-home grep hits: %s\n' "${home_hit_count}"
if [[ "${home_hit_count}" -ne 0 ]]; then
  printf '%s\n' "${home_hits}" >&2
  exit 1
fi

# Commit. Incremental mode adds one commit describing this publish; rebuild
# mode produces the single fresh commit. Either way the commit message is
# operator-supplied prose, never private history.
if [[ "${incremental}" -eq 1 && -z "$(git -C "${mirror_root}" diff --cached --name-status)" ]]; then
  printf 'no changes to publish; mirror already matches the source tree.\n'
  exit 0
fi
if [[ -z "${commit_message}" ]]; then
  if [[ "${incremental}" -eq 1 ]]; then
    commit_message="Mirror update $(date -u +%Y-%m-%d)"
  else
    commit_message="Initial public mirror"
  fi
fi
git -C "${mirror_root}" -c user.name='Grove public mirror' -c user.email='grove-public@localhost' commit -m "${commit_message}" >/dev/null

mirror_file_count="$(git -C "${mirror_root}" ls-files -z | python3 -c 'import sys; print(sum(1 for item in sys.stdin.buffer.read().split(b"\0") if item))')"
printf 'files in mirror commit: %s\n' "${mirror_file_count}"
if [[ "${mirror_file_count}" -ne "${copied_count}" ]]; then
  printf 'mirror file count differs from copied count\n' >&2
  exit 1
fi
if [[ "${incremental}" -eq 0 && -n "$(git -C "${mirror_root}" remote)" ]]; then
  printf 'fresh mirror unexpectedly has a remote\n' >&2
  exit 1
fi

log_output="$(git -C "${mirror_root}" log --oneline -5)"
commit_count="$(git -C "${mirror_root}" rev-list --count HEAD)"
printf 'git log --oneline (%s commit(s), last 5):\n%s\n' "${commit_count}" "${log_output}"
if [[ "${incremental}" -eq 0 && "${commit_count}" -ne 1 ]]; then
  printf 'expected exactly one mirror commit after rebuild\n' >&2
  exit 1
fi
if [[ -n "$(git -C "${mirror_root}" status --porcelain)" ]]; then
  printf 'mirror has uncommitted changes\n' >&2
  exit 1
fi

printf 'mirror ready: %s\n' "${mirror_root}"
printf 'publish with: git -C %s push origin main\n' "${mirror_root}"
