#!/usr/bin/env bash
# Point the official BatteryLife code at one version of the processed data.
#
# Usage: scripts/link_data.sh v4|v12
#
# Creates relative symlinks (none are tracked by git):
#   dataset/<version>/seen_unseen_labels -> upstream/dataset/seen_unseen_labels
#   upstream/dataset/Life labels/<file>  -> dataset/<version>/Life labels/<file>
#
# models/Dummy.py treats every json in upstream/dataset/Life labels as a
# dataset, so only the label files of the datasets used in the paper are
# linked. This leaves out e.g. NA-coin_labels.json (v4), which has no data
# in the release, and the datasets added after v4 (SDU, Stanford_2).
# Training is run from upstream/ with --root_path ../dataset/<version>.

set -euo pipefail

version="${1:-}"
if [[ "$version" != "v4" && "$version" != "v12" ]]; then
    echo "Usage: $0 v4|v12" >&2
    exit 1
fi

paper_label_files=(
    CALB_labels.json CALCE_labels.json HNEI_labels.json HUST_labels.json
    ISU-ILCC_labels.json MATR_labels.json NA-ion_labels.json RWTH_labels.json
    SNL_labels.json Stanford_labels.json Tongji_labels.json UL-PUR_labels.json
    XJTU_labels.json ZN-coin_labels.json total_MICH_labels.json
)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/dataset/$version"
labels_src="$data_dir/Life labels"
labels_dst="$repo_root/upstream/dataset/Life labels"

if [[ ! -d "$labels_src" ]]; then
    echo "Missing $labels_src. Download and unzip the data first." >&2
    exit 1
fi

# Seen/unseen labels inside the data folder.
ln -sfn "../../upstream/dataset/seen_unseen_labels" "$data_dir/seen_unseen_labels"

# Fresh folder of per-file links for the life labels.
if [[ -L "$labels_dst" ]]; then
    rm "$labels_dst"
elif [[ -d "$labels_dst" ]]; then
    find "$labels_dst" -mindepth 1 -maxdepth 1 -type l -delete
    if [[ -n "$(ls -A "$labels_dst")" ]]; then
        echo "$labels_dst contains real files. Remove them first." >&2
        exit 1
    fi
fi
mkdir -p "$labels_dst"

for f in "${paper_label_files[@]}"; do
    if [[ ! -f "$labels_src/$f" ]]; then
        echo "Missing $labels_src/$f" >&2
        exit 1
    fi
    ln -s "../../../dataset/$version/Life labels/$f" "$labels_dst/$f"
done

skipped=$(comm -23 <(ls "$labels_src" | grep '\.json$' | sort) <(printf '%s\n' "${paper_label_files[@]}" | sort) | tr '\n' ' ')

echo "Linked BatteryLife data $version"
echo "  $data_dir/seen_unseen_labels -> $(readlink "$data_dir/seen_unseen_labels")"
echo "  $labels_dst: ${#paper_label_files[@]} label files"
echo "  not linked (not used in the paper): ${skipped:-none}"
