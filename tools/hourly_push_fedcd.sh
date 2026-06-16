#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/ds1home/aislab/Min/FedCD}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-add baseline logs per hour}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
LOG_FILE="${LOG_FILE:-/tmp/fedcd_hourly_push_fedcd.log}"
LOCK_DIR="${LOCK_DIR:-/tmp/fedcd_hourly_push.lock}"

ONCE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: hourly_push_fedcd.sh [options]

Options:
  --once                    Run one commit/push attempt and exit.
  --dry-run                 Print current git status without staging or pushing.
  --repo-dir PATH           Git repository directory. Default: /ds1home/aislab/Min/FedCD
  --remote NAME             Git remote name. Default: origin
  --branch NAME             Remote branch name. Default: main
  --interval-seconds N      Sleep interval for loop mode. Default: 3600
  --log-file PATH           Log file path. Default: /tmp/fedcd_hourly_push_fedcd.log
  -h, --help                Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --repo-dir)
      shift
      REPO_DIR="$1"
      ;;
    --remote)
      shift
      REMOTE="$1"
      ;;
    --branch)
      shift
      BRANCH="$1"
      ;;
    --interval-seconds)
      shift
      INTERVAL_SECONDS="$1"
      ;;
    --log-file)
      shift
      LOG_FILE="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  local line
  line="[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$LOG_FILE"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
    return 0
  fi

  log "Another hourly push process is already running; exiting."
  exit 0
}

validate_config() {
  if [[ ! "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SECONDS" -lt 1 ]]; then
    log "Invalid --interval-seconds value: $INTERVAL_SECONDS"
    exit 2
  fi

  if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Not a git repository: $REPO_DIR"
    exit 2
  fi
}

print_status() {
  git -C "$REPO_DIR" status --short --branch
}

run_once() {
  log "Checking $REPO_DIR"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry run only; no files will be staged, committed, or pushed."
    print_status
    return 0
  fi

  git -C "$REPO_DIR" add -A .

  if git -C "$REPO_DIR" diff --cached --quiet; then
    log "No changes to commit."
    return 0
  fi

  git -C "$REPO_DIR" commit -m "$COMMIT_MESSAGE"
  git -C "$REPO_DIR" push "$REMOTE" "HEAD:$BRANCH"
  log "Pushed committed changes to $REMOTE/$BRANCH."
}

validate_config
acquire_lock

while true; do
  if ! run_once; then
    log "Push attempt failed; will retry on the next interval."
  fi

  if [[ "$ONCE" -eq 1 ]]; then
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
