#!/usr/bin/env bash
# Point the official BatteryLife code at one version of the processed data.
#
# Usage: scripts/link_data.sh v4|v12
#
# Creates two relative symlinks (neither is tracked by git):
#   dataset/<version>/seen_unseen_labels -> upstream/dataset/seen_unseen_labels
#   upstream/dataset/Life labels         -> dataset/<version>/Life labels
# Training is then run from upstream/ with --root_path ../dataset/<version>.

set -euo pipefail

version="${1:-}"
if [[ "$version" != "v4" && "$version" != "v12" ]]; then
    echo "Usage: $0 v4|v12" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/dataset/$version"
labels_link="$repo_root/upstream/dataset/Life labels"

if [[ ! -d "$data_dir/Life labels" ]]; then
    echo "Missing $data_dir/Life labels. Download and unzip the data first." >&2
    exit 1
fi

if [[ -e "$labels_link" && ! -L "$labels_link" ]]; then
    echo "$labels_link exists and is not a symlink. Remove it first." >&2
    exit 1
fi

ln -sfn "../../upstream/dataset/seen_unseen_labels" "$data_dir/seen_unseen_labels"
ln -sfn "../../dataset/$version/Life labels" "$labels_link"

echo "Linked BatteryLife data $version"
echo "  $data_dir/seen_unseen_labels -> $(readlink "$data_dir/seen_unseen_labels")"
echo "  $labels_link -> $(readlink "$labels_link")"
