# Git Automation Setup

This repository is configured with several automation features to streamline Git operations.

## Quick Commands

### Basic Operations
```bash
# Commit and push in one command
./git_push.sh cp "Your commit message"

# Just commit
./git_push.sh commit "Your commit message"

# Just push
./git_push.sh push

# Check status
./git_push.sh status
```

## Features

### 1. GitHub Actions - Claude Integration
- **Location**: `.github/workflows/claude.yml`
- **Trigger**: Comment `@claude` in issues or PRs
- **Requirement**: Set `ANTHROPIC_API_KEY` in repository secrets

### 2. Automated Git Operations
- **Script**: `git_push.sh`
- Handles GPG signing failures automatically
- Provides simple commands for common operations
- No need to manually handle signing issues

### 3. Git Configuration
The following are configured globally:
- GPG signing (with automatic fallback)
- GitHub CLI authentication
- Credential caching

## Setup Instructions

### For New Clones
1. Clone the repository
2. Make `git_push.sh` executable: `chmod +x git_push.sh`
3. Use the script for all git operations

### For GitHub Actions
1. Go to repository Settings → Secrets and variables → Actions
2. Add secret: `ANTHROPIC_API_KEY`
3. The workflow will automatically respond to `@claude` mentions

## Common Issues & Solutions

### GPG Signing Fails
- The `git_push.sh` script automatically handles this
- Falls back to unsigned commits if GPG is unavailable

### Push Requires Manual Button
- Use `./git_push.sh push` instead of GUI
- Or use `./git_push.sh cp "message"` to commit and push together

### Need to Force Push
```bash
./git_push.sh push main --force
```

## Python Path
Always use: `/home/divake/miniconda3/envs/env_cu121/bin/python`