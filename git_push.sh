#!/bin/bash
# Script to handle git operations with proper configuration

# Set GPG TTY for signing
export GPG_TTY=$(tty)

# Function to commit with automatic retry on GPG failure
git_commit() {
    local message="$1"
    
    # Try with GPG signing first
    if git commit -m "$message" 2>/dev/null; then
        echo "✓ Committed with GPG signature"
    else
        # If GPG fails, commit without signing
        echo "⚠️  GPG signing failed, committing without signature"
        git commit --no-gpg-sign -m "$message"
    fi
}

# Function to push with proper error handling
git_push() {
    local branch="${1:-main}"
    local force="${2:-}"
    
    if [ "$force" == "--force" ]; then
        git push --force-with-lease origin "$branch"
    else
        git push origin "$branch"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully pushed to origin/$branch"
    else
        echo "✗ Push failed"
        return 1
    fi
}

# Main operations
case "${1:-help}" in
    commit)
        shift
        git_commit "$@"
        ;;
    push)
        shift
        git_push "$@"
        ;;
    cp)  # commit and push
        message="${2:-Auto commit}"
        git add -A
        git_commit "$message"
        git_push
        ;;
    status)
        git status
        ;;
    help|*)
        echo "Usage: ./git_push.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  commit <message>    - Commit with automatic GPG handling"
        echo "  push [branch]       - Push to remote"
        echo "  cp <message>        - Add all, commit, and push"
        echo "  status              - Show git status"
        echo ""
        echo "Examples:"
        echo "  ./git_push.sh commit 'Fix: Updated scoring function'"
        echo "  ./git_push.sh push main"
        echo "  ./git_push.sh cp 'Feature: Added new dataset'"
        ;;
esac