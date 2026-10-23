#!/usr/bin/env bash
# Static checks: eslint + eslint-plugin-sonarjs, jscpd, knip.
# Numbers always print; individual tools may exit non-zero when findings exist.
set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/static"
mkdir -p "$OUT/logs"
cd "$ROOT"

echo "== 1/3 eslint (browser: docs/ | node: scripts/, tests/) ==" > "$OUT/eslint.log"
node_modules/.bin/eslint "docs/**/*.js" >> "$OUT/eslint.log" 2>&1
ESLINT=$?
if find scripts tests -name "*.js" 2>/dev/null | grep -q .; then
  echo "-- node scope --" >> "$OUT/eslint.log"
  node_modules/.bin/eslint --no-error-on-unmatched-pattern "scripts/**/*.js" "tests/**/*.js" >> "$OUT/eslint.log" 2>&1
  ESLINT=$?
else
  echo "-- node scope: no JS files yet, skipped --" >> "$OUT/eslint.log"
fi

echo "== 2/3 jscpd (duplication, threshold 1%) ==" > "$OUT/jscpd.log"
node_modules/.bin/jscpd "docs" "scripts" "tests" \
  --min-tokens 30 --threshold 1 --reporters consoleFull,html --output "$OUT/jscpd" --silent \
  >> "$OUT/jscpd.log" 2>&1
JSCPD=$?

echo "== 3/3 knip (dead code) ==" > "$OUT/knip.log"
node_modules/.bin/knip >> "$OUT/knip.log" 2>&1
KNIP=$?

echo; echo "===== summary ====="
printf '%-12s exit=%d%s\n' eslint "$ESLINT" "$(grep -cm1 'No files matching' "$OUT/eslint.log" >/dev/null && echo '' )"
printf '%-12s exit=%d\n' jscpd "$JSCPD"
printf '%-12s exit=%d\n' knip "$KNIP"
echo "duplication: $(grep -oE 'Found [0-9]+ exact clones with [0-9]+\([0-9.]+%\) duplicated lines' "$OUT/jscpd.log" | tail -1)"
echo
echo "logs: results/static/{eslint,jscpd,knip}.log ; jscpd HTML: results/static/jscpd/jscpd-report.html"
echo "--- eslint.log ---"; cat "$OUT/eslint.log"
echo "--- jscpd.log ---"; tail -8 "$OUT/jscpd.log"
echo "--- knip.log ---"; cat "$OUT/knip.log"
