#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/vdb_to_sparse_samples.cpp"
BIN="$(mktemp /tmp/vdb_to_sparse_samples.XXXXXX)"

cleanup() {
  rm -f "${BIN}"
}
trap cleanup EXIT

if ! g++ -O3 -std=c++17 "${SRC}" -o "${BIN}" -lopenvdb -ltbb -lImath-3_1; then
  g++ -O3 -std=c++17 "${SRC}" -o "${BIN}" -lopenvdb -ltbb -lImath
fi
"${BIN}" "$@"
