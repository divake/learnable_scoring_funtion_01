#!/bin/bash
# Enhanced Git automation script that syncs with GUI tools like Cursor

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set GPG TTY for signing
export GPG_TTY=$(tty)

# Function to refresh GUI by touching .git/index
refresh_gui() {
    touch .git/index
    # Also fetch to update remote refs
    git fetch --quiet
}

# Function to commit with automatic retry on GPG failure
git_commit() {
    local message="$1"
    
    echo -e "${YELLOW}Committing: $message${NC}"
    
    # Try with GPG signing first
    if git commit -m "$message" 2>/dev/null; then
        echo -e "${GREEN}✓ Committed with GPG signature${NC}"
    else
        # If GPG fails, commit without signing
        echo -e "${YELLOW}⚠️  GPG signing failed, committing without signature${NC}"
        git commit --no-gpg-sign -m "$message"
    fi
}

# Function to push with proper error handling and GUI sync
git_push() {
    local branch="${1:-main}"
    local force="${2:-}"
    
    echo -e "${YELLOW}Pushing to origin/$branch...${NC}"
    
    if [ "$force" == "--force" ]; then
        git push --force-with-lease origin "$branch"
    else
        git push origin "$branch"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully pushed to origin/$branch${NC}"
        # Refresh GUI after successful push
        refresh_gui
        echo -e "${GREEN}✓ GUI refreshed${NC}"
        
        # Pull to ensure local is fully synced
        git pull --rebase --quiet
        echo -e "${GREEN}✓ Local branch synced${NC}"
    else
        echo -e "${RED}✗ Push failed${NC}"
        return 1
    fi
}

# Function for complete automation (add, commit, push, sync)
auto_commit_push() {
    local message="${1:-Auto commit}"
    
    echo -e "${YELLOW}Starting automated git workflow...${NC}"
    echo ""
    
    # Check if there are changes
    if [ -z "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}No changes to commit${NC}"
        return 0
    fi
    
    # Show status
    echo -e "${YELLOW}Current changes:${NC}"
    git status --short
    echo ""
    
    # Add all changes
    git add -A
    echo -e "${GREEN}✓ Added all changes${NC}"
    
    # Commit
    git_commit "$message"
    
    # Push
    git_push
    
    # Final sync
    echo ""
    echo -e "${GREEN}✓ Complete! All changes committed and pushed.${NC}"
    echo -e "${GREEN}✓ GUI should now be in sync.${NC}"
}

# Function to sync with remote (pull + push)
sync_remote() {
    echo -e "${YELLOW}Syncing with remote...${NC}"
    
    # Fetch latest
    git fetch origin
    
    # Pull with rebase
    git pull --rebase origin main
    
    # Push any local commits
    git push origin main
    
    # Refresh GUI
    refresh_gui
    
    echo -e "${GREEN}✓ Synced with remote${NC}"
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
    cp|auto)  # commit and push with full automation
        shift
        auto_commit_push "$@"
        ;;
    sync)
        sync_remote
        ;;
    status)
        git status
        echo ""
        echo -e "${YELLOW}Remote status:${NC}"
        git remote show origin
        ;;
    refresh)
        refresh_gui
        echo -e "${GREEN}✓ GUI refreshed${NC}"
        ;;
    help|*)
        echo "Enhanced Git Automation Script"
        echo ""
        echo "Usage: ./auto_git.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  auto <message>      - Full automation: add, commit, push, sync"
        echo "  cp <message>        - Same as 'auto'"
        echo "  commit <message>    - Just commit with GPG handling"
        echo "  push [branch]       - Just push to remote"
        echo "  sync                - Sync with remote (pull + push)"
        echo "  refresh             - Refresh GUI (Cursor/VSCode)"
        echo "  status              - Show detailed status"
        echo ""
        echo "Examples:"
        echo "  ./auto_git.sh auto 'Fix: Updated scoring function'"
        echo "  ./auto_git.sh sync"
        echo "  ./auto_git.sh refresh"
        echo ""
        echo "This script automatically handles:"
        echo "  - GPG signing failures"
        echo "  - GUI synchronization (Cursor/VSCode)"
        echo "  - Remote syncing"
        ;;
esac