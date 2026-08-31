#!/bin/bash
set -e
shopt -s nullglob
zips=(*.zip)
if [ ${#zips[@]} -eq 0 ]; then echo "no zip found"; exit 1; fi
newest=$(ls -t *.zip | head -1)
echo "using: $newest"
unzip -o "$newest"
rm -f *.zip
git add -A
if git diff --cached --quiet; then
  echo "no changes - repo already matches this zip"
else
  git commit -m "${1:-update ($newest)}"
  git push
fi
