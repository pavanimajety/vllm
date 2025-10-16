#!/bin/bash
# Script to update integration branch when WIP PRs are updated

set -e

echo "Updating integration branch with latest PR changes..."

# Store current branch
CURRENT_BRANCH=$(git branch --show-current)

# Update main
echo "Updating main..."
git fetch origin main:main

# Fetch latest changes from both PRs
echo "Fetching latest PR #25103..."
git fetch https://github.com/therealnaveenkamal/vllm.git mla_attn:wip-pr-25103 --force

echo "Fetching latest PR #25954..."
git fetch https://github.com/neuralmagic/vllm.git split-attention-cache-update:wip-pr-25954 --force

# Rebuild integration-base
echo "Rebuilding integration-base..."
git checkout integration-base
git reset --hard main
git merge wip-pr-25103 --no-edit
git merge wip-pr-25954 --no-edit

# Rebase feature branch
echo "Rebasing my-feature-branch..."
git checkout my-feature-branch
git rebase integration-base

echo ""
echo "✓ Update complete!"
echo ""
echo "Current branch: $(git branch --show-current)"
echo "Integration-base updated with latest changes from both PRs"
echo "Your feature branch rebased on top"
echo ""
echo "If there are conflicts, resolve them and run: git rebase --continue"

