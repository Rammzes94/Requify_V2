# 🧠 Basic Git Cheat Sheet

## 🔧 Initial Setup
```bash
git init                      # Initialize a new Git repository
git clone <url>              # Clone a repository
```

## 📥 Staging & Committing
```bash
git status                   # Check status of files
git add .                    # Stage all changes
git add <file>               # Stage a specific file
git reset <file>             # Unstage a file

git commit -m "message"      # Commit staged changes
git commit -am "message"     # Add + commit tracked files
```

## 🔄 Sync with Remote
```bash
git remote add origin <url>  # Add remote repository
git remote remove origin     # Remove remote repository
git remote -v                # Show remotes

git push -u origin main      # Push first time, set upstream
git push                     # Push changes
git pull                     # Pull latest changes
```

## 🕹️ Branching
```bash
git branch                   # List branches
git branch <name>            # Create a new branch
git checkout <name>          # Switch to a branch
git checkout -b <name>       # Create and switch to new branch
git merge <branch>           # Merge branch into current
```

## 🧽 Cleaning Up
```bash
git rm <file>                # Remove file and stage the deletion
git rm --cached <file>       # Remove from index (keep local)
git clean -fd                # Remove untracked files & dirs
```

## 🕰️ Undo & History
```bash
git log                      # Show commit history
git diff                     # Show unstaged changes
git restore <file>           # Restore file from last commit
git reset --hard HEAD        # Reset all to last commit
```
