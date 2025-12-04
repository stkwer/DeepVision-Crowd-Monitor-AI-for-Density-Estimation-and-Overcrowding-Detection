#!/usr/bin/env python3
"""
Rename preprocessed images like __results___100_1.png -> IMG_100.png

Behavior:
- Scans the given preprocessed folder for image files (.png/.jpg/.jpeg)
- Extracts numeric tokens from filename and selects the largest number as the original image id
  (e.g. from '__results___100_1.png' -> selects 100 -> target 'IMG_100.png')
- Preserves the original file extension
- If multiple preprocessed files would map to the same target name, appends a counter suffix
  (IMG_100.png, IMG_100_1.png, IMG_100_2.png, ...)
- By default performs a dry-run. Use --apply to actually perform renames.
- Saves a CSV mapping file (rename_mapping.csv) listing original and new paths.

Usage examples:
  # dry-run (safe): shows planned renames but does not change files
  python rename_preprocessed_to_img_names.py --preproc-folder "C:\\Users\\ajayv\\OneDrive\\Desktop\\CrowdCounting\\preprocessed"

  # actually apply renames
  python rename_preprocessed_to_img_names.py --preproc-folder "C:\\Users\\ajayv\\OneDrive\\Desktop\\CrowdCounting\\preprocessed" --apply

  # copy instead of rename
  python rename_preprocessed_to_img_names.py --preproc-folder "..." --apply --copy

"""

import os
import re
import argparse
import csv
import shutil
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def choose_number_from_tokens(tokens):
    """Choose the numeric token that most likely corresponds to the original image ID.
    Heuristic: choose the numeric token with the largest integer value (works for names like
    __results___100_1.png where tokens are [100,1] -> pick 100).
    """
    if not tokens:
        return None
    nums = [int(t.lstrip('0') or '0') for t in tokens]
    # return the token that has the maximum numeric value
    max_idx = max(range(len(nums)), key=lambda i: nums[i])
    return tokens[max_idx]


def plan_and_execute(preproc_folder, apply_changes=False, copy_instead=False, mapping_csv_path=None):
    preproc_folder = Path(preproc_folder)
    if not preproc_folder.is_dir():
        raise ValueError(f"Preprocessed folder not found: {preproc_folder}")

    files = [p for p in preproc_folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    print(f"Found {len(files)} image files in {preproc_folder}")

    planned = []  # tuples of (orig_path, target_path)
    target_names_count = {}

    for p in sorted(files):
        name = p.stem
        ext = p.suffix.lower()
        tokens = re.findall(r"\d+", p.name)
        chosen = choose_number_from_tokens(tokens)
        if chosen is None:
            # fallback: move to 'unnamed' plan (leave unchanged)
            print(f"No numeric token found in filename '{p.name}' -> SKIPPING")
            continue

        base_name = f"IMG_{int(chosen)}"  # normalize (remove leading zeros)
        target = base_name + ext

        # avoid collisions: if target already planned or exists, add counter suffix
        cnt = target_names_count.get(target, 0)
        final_target = target if cnt == 0 else f"{base_name}_{cnt}{ext}"
        # increment counter for future collisions
        target_names_count[target] = cnt + 1

        final_target_path = preproc_folder / final_target

        # if file already exists on disk (not just planned), also ensure uniqueness
        i = 1
        while final_target_path.exists() and final_target_path != p:
            final_target = f"{base_name}_{cnt + i}{ext}"
            final_target_path = preproc_folder / final_target
            i += 1

        planned.append((str(p), str(final_target_path)))

    # show summary
    if not planned:
        print("No files to rename.")
        return planned

    print("Planned renames (showing up to 200):")
    for orig, tgt in planned[:200]:
        print(f"  {os.path.basename(orig)}  ->  {os.path.basename(tgt)}")

    # write mapping csv
    if mapping_csv_path is None:
        mapping_csv_path = preproc_folder / 'rename_mapping.csv'
    else:
        mapping_csv_path = Path(mapping_csv_path)

    with open(mapping_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original_path', 'new_path', 'applied'])
        for orig, tgt in planned:
            writer.writerow([orig, tgt, 'no'])

    if not apply_changes:
        print(f"Dry-run complete. No files have been changed. To apply changes use --apply")
        print(f"Mapping written to: {mapping_csv_path}")
        return planned

    # perform rename or copy
    for orig, tgt in planned:
        try:
            if copy_instead:
                shutil.copy2(orig, tgt)
            else:
                os.replace(orig, tgt)  # atomic rename on many platforms
            # update CSV 'applied' column by appending a new mapping file (simple)
        except Exception as e:
            print(f"Failed to move {orig} -> {tgt}: {e}")

    # rewrite mapping with applied flag
    with open(mapping_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original_path', 'new_path', 'applied'])
        for orig, tgt in planned:
            applied_flag = 'yes' if Path(tgt).exists() else 'no'
            writer.writerow([orig, tgt, applied_flag])

    print(f"Applied {len(planned)} rename/copy operations. Mapping written to: {mapping_csv_path}")
    return planned


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename preprocessed images like __results___100_1.png -> IMG_100.png')
    parser.add_argument('--preproc-folder', required=True, help='Preprocessed images folder')
    parser.add_argument('--apply', action='store_true', help='Actually perform the renames (default: dry-run)')
    parser.add_argument('--copy', dest='copy', action='store_true', help='Copy files instead of renaming (keeps originals)')
    parser.add_argument('--mapping-csv', default=None, help='Path to write mapping CSV (default: rename_mapping.csv in preproc folder)')

    args = parser.parse_args()
    plan_and_execute(args.preproc_folder, apply_changes=args.apply, copy_instead=args.copy, mapping_csv_path=args.mapping_csv)
