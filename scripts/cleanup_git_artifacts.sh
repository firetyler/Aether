#!/usr/bin/env bash
set -e

echo "Removing tracked virtualenv and generated artifacts from Git index..."

git rm -r --cached --ignore-unmatch .venv venv env logs .vscode *.log ai_Model/*.pth || true

git add .gitignore

git commit -m "chore: remove tracked virtualenv and generated artifacts; respect .gitignore" || true

echo "Cleanup complete. If nothing was removed, review git status for remaining files."
