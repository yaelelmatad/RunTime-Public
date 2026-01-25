#!/usr/bin/env bash
set -euo pipefail

# Generate the RunTime white paper PDF from Markdown via Pandoc.
#
# Usage:
#   bash generate_white_paper_pdf.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MD_IN="${ROOT_DIR}/Technical_Details.md"
PDF_OUT="${ROOT_DIR}/Technical_Details.pdf"
TEMPLATE="${ROOT_DIR}/Paper_Template.tex"

if [[ ! -f "${MD_IN}" ]]; then
  echo "ERROR: missing input markdown: ${MD_IN}" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: missing pandoc LaTeX template: ${TEMPLATE}" >&2
  echo "Hint: generate one via: pandoc -D latex > White_Paper_Template.tex" >&2
  exit 1
fi

echo "[generate_white_paper_pdf] building..."

echo "  input:    ${MD_IN}"
echo "  template: ${TEMPLATE}"
echo "  output:   ${PDF_OUT}"

pandoc "${MD_IN}" \
  -o "${PDF_OUT}" \
  --template="${TEMPLATE}" \
  --resource-path="${ROOT_DIR}"

echo "[generate_white_paper_pdf] done: ${PDF_OUT}"
