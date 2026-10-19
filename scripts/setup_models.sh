#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

clone_if_missing() {
    local repo_url="$1"
    local destination="$2"
    if [[ -e "$destination" ]]; then
        printf 'Already exists, skipping: %s\n' "$destination"
    else
        git clone "$repo_url" "$destination"
    fi
}

clone_if_missing "https://github.com/EdgeIntelligenceLab/HART.git" "$MODELS_DIR/HART"
clone_if_missing "https://github.com/WANDS-HKUST/LIMU-BERT_Experience.git" "$MODELS_DIR/LIMU-BERT-X"

printf '\nModel repository paths:\n%s\n%s\n' \
    "$MODELS_DIR/HART" \
    "$MODELS_DIR/LIMU-BERT-X"
