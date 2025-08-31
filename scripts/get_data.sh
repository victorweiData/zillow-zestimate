#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "==> Downloading Zillow Prize data into $RAW_DIR ..."

# Download directly into raw dir
kaggle competitions download -c zillow-prize-1 -p "$RAW_DIR"

cd "$RAW_DIR"

# Unzip and remove the archive
unzip -o zillow-prize-1.zip
rm -f zillow-prize-1.zip

echo "✅ Done. Files are in $RAW_DIR"