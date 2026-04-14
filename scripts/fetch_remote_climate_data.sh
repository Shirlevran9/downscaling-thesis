#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-shirlevran}"
JUMP_HOST="${JUMP_HOST:-${REMOTE_USER}@bava.cs.huji.ac.il}"
REMOTE_HOST="${REMOTE_HOST:-river}"
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
CONTROL_PATH="${HOME}/.ssh/codex-%r@%h:%p"
SSH_OPTS=(-J "$JUMP_HOST" -o ControlMaster=auto -o ControlPersist=10m -o "ControlPath=${CONTROL_PATH}")
EST_MBPS="${EST_MBPS:-20}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/data/raw/remote_sources"
CMIP_DEST="${DEST_DIR}/cmip6"
ERA5_DEST="${DEST_DIR}/era5"
ERA5_LAND_DEST="${DEST_DIR}/era5_land"
mkdir -p "$CMIP_DEST" "$ERA5_DEST" "$ERA5_LAND_DEST"

echo "[info] Project root: ${PROJECT_ROOT}"
echo "[info] Remote target: ${SSH_TARGET} via ${JUMP_HOST}"
echo "[info] Local destination: ${DEST_DIR}"
echo "[info] Transfer estimate speed: ${EST_MBPS} MB/s"

run_remote() {
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"
}

format_bytes() {
  local bytes="$1"
  python3 - <<PY
size = int(${bytes})
units = ["B", "KB", "MB", "GB", "TB"]
value = float(size)
unit = units[0]
for unit in units:
    if value < 1024 or unit == units[-1]:
        break
    value /= 1024
print(f"{value:.1f} {unit}")
PY
}

estimate_transfer_time() {
  local bytes="$1"
  local mbps="$2"
  python3 - <<PY
size = int(${bytes})
mbps = float(${mbps})
seconds = size / (mbps * 1024 * 1024) if mbps > 0 else 0
if seconds < 60:
    print(f"{seconds:.1f} sec")
else:
    print(f"{seconds/60:.1f} min")
PY
}

get_remote_file_size() {
  local remote_path="$1"
  run_remote "python3 - <<'PY'
from pathlib import Path
path = Path(${remote_path@Q})
print(path.stat().st_size)
PY"
}

copy_remote_file() {
  local remote_path="$1"
  local local_dir="$2"
  local size_bytes
  size_bytes="$(get_remote_file_size "$remote_path" | tail -n 1)"
  local pretty_size
  pretty_size="$(format_bytes "$size_bytes")"
  local eta
  eta="$(estimate_transfer_time "$size_bytes" "$EST_MBPS")"
  echo "[copy] ${remote_path}"
  echo "[copy] size=${pretty_size} (${size_bytes} bytes)"
  echo "[copy] estimated transfer time at ${EST_MBPS} MB/s: ${eta}"
  echo "[copy] ${remote_path}"
  scp "${SSH_OPTS[@]}" "${SSH_TARGET}:${remote_path}" "${local_dir}/"
}

find_first_remote_match() {
  local root="$1"
  local name_glob="$2"
  local required_fragment="$3"

  echo "[search] root=${root}" >&2
  echo "[search] pattern=${name_glob}" >&2
  if [[ -n "$required_fragment" ]]; then
    echo "[search] required fragment=${required_fragment}" >&2
  fi

  local cmd="find '$root' -type f -name '$name_glob' 2>/dev/null"
  if [[ -n "$required_fragment" ]]; then
    cmd+=" | grep -i '$required_fragment'"
  fi
  cmd+=" | head -n 1"

  run_remote "bash -lc $(
    printf '%q' "$cmd"
  )"
}

fetch_if_missing() {
  local label="$1"
  local local_path="$2"
  local local_dir="$3"
  local remote_root="$4"
  local name_glob="$5"
  local required_fragment="${6:-}"

  if [[ -f "$local_path" ]]; then
    echo "[skip] ${label} already exists locally: ${local_path}"
    return 0
  fi

  echo "[resolve] ${label} not found locally. Searching remote source..."
  local remote_match
  remote_match="$(find_first_remote_match "$remote_root" "$name_glob" "$required_fragment" | tail -n 1 || true)"

  if [[ -z "$remote_match" ]]; then
    echo "[miss] Could not find ${label} under ${remote_root}"
    return 1
  fi

  echo "[found] ${label}: ${remote_match}"
  copy_remote_file "$remote_match" "$local_dir"
}

CMIP_FILE="tas_day_CESM2-WACCM_historical_r1i1p1f1_gn_19900101-19991231.nc"
CMIP_LOCAL="${CMIP_DEST}/${CMIP_FILE}"

echo
echo "[step] Fetching minimal CMIP6 exploration file..."
if ! fetch_if_missing \
  "CMIP6 tas historical sample" \
  "$CMIP_LOCAL" \
  "$CMIP_DEST" \
  "/sci/labs/efratmorin/anton.gelman/work" \
  "$CMIP_FILE" \
  ""; then
  echo "[note] CMIP sample was not found in the searched Anton path."
  echo "[note] Trying a few narrower fallback CMIP directories from the brief..."
  if ! fetch_if_missing \
    "CMIP6 tas historical sample (fallback)" \
    "$CMIP_LOCAL" \
    "$CMIP_DEST" \
    "/sci/labs/efratmorin/anton.gelman/work/scenarios" \
    "$CMIP_FILE" \
    ""; then
    if ! fetch_if_missing \
      "CMIP6 tas historical sample (Andre fallback)" \
      "$CMIP_LOCAL" \
      "$CMIP_DEST" \
      "/sci/labs/assafhochman/andre.klif/data/hist" \
      "$CMIP_FILE" \
      ""; then
      echo "[note] CMIP sample was not found in the fallback paths either."
    fi
  fi
fi

echo
echo "[step] Fetching one ERA5 t2m file mentioned in the brief..."
if ! fetch_if_missing \
  "ERA5 t2m 1990 sample" \
  "${ERA5_DEST}/t2m_day_levels_era5_1990.nc" \
  "$ERA5_DEST" \
  "/sci/labs/assafhochman/andre.klif/data/hist/ERA5" \
  "*1990*.nc" \
  "t2m"; then
  echo "[note] ERA5 t2m 1990 sample was not found in Andre's documented path."
fi

echo
echo "[step] Attempting to locate an ERA5-Land 1990 file on shared storage..."
if ! fetch_if_missing \
  "ERA5-Land 1990 sample" \
  "${ERA5_LAND_DEST}/era5_land_1990.nc" \
  "$ERA5_LAND_DEST" \
  "/sci/labs/efratmorin/ronit" \
  "*1990*.nc" \
  "era5"; then
  echo "[note] ERA5-Land was not found under /sci/labs/efratmorin/ronit."
  if ! fetch_if_missing \
    "ERA5-Land 1990 sample (Morin fallback)" \
    "${ERA5_LAND_DEST}/era5_land_1990.nc" \
    "$ERA5_LAND_DEST" \
    "/sci/labs/efratmorin" \
    "*1990*.nc" \
    "land"; then
    true
  fi
fi
if [[ ! -f "${ERA5_LAND_DEST}/era5_land_1990.nc" ]]; then
  echo "[note] No shared ERA5-Land file was found through the documented cluster paths."
  echo "[note] The brief's explicit source for ERA5-Land is CDS, not a fixed shared filesystem path."
fi

echo
echo "[done] Remote fetch attempt completed."
echo "[done] Files copied into: ${DEST_DIR}"
