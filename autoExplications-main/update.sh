#!/bin/bash
set -e
unzip -o exp_server.zip
rm exp_server.zip
git add -A
git commit -m "${1:-update}"
git push
