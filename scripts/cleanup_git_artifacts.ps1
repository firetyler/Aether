Param()

Write-Host "Removing tracked virtualenv and generated artifacts from Git index..."

git rm -r --cached --ignore-unmatch .venv venv env logs .vscode *.log ai_Model/*.pth

git add .gitignore

git commit -m "chore: remove tracked virtualenv and generated artifacts; respect .gitignore"

Write-Host "Cleanup complete. If nothing was removed, review git status for remaining files."
