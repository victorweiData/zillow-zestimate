#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="data/external/census/shp_ca_tracts_2016"
ZIP="$OUT_DIR/tl_2016_06_tract.zip"
URL="https://www2.census.gov/geo/tiger/TIGER2016/TRACT/tl_2016_06_tract.zip"

echo "==> Downloading TIGER/Line 2016 California Census Tracts (shapefile)…"
mkdir -p "$OUT_DIR"
curl -L "$URL" -o "$ZIP"
unzip -o "$ZIP" -d "$OUT_DIR"
echo "✅ Tracts shapefile ready at $OUT_DIR"
