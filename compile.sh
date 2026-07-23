#!/bin/bash
# Compile main.tex for Machine Learning (Springer) submission
# Usage: ./compile.sh

export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$(dirname "$0")"

echo "=== 1/4 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== 2/4 bibtex ==="
bibtex main

echo "=== 3/4 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== 4/4 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex

echo "=== DONE ==="
grep "Warning.*undefined" main.log | wc -l | xargs echo "Undefined refs:"
grep "Output written" main.log | tail -1
