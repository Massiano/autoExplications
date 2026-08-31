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
git commit -m "${1:-update ($newest)}"
git push
