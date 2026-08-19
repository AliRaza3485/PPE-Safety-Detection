"""
Remove the 'person' class (class_id 2) from all label files.
Keeps 'head' (0) and 'helmet' (1) unchanged — no remapping needed
since they are already the first two class IDs.
"""

from pathlib import Path

DATA_ROOT = Path("data/raw")
SPLITS = ["train", "valid", "test"]
PERSON_CLASS_ID = "2"


def clean_labels():
    total_files_modified = 0
    total_lines_removed = 0
    total_files_now_empty = 0

    for split in SPLITS:
        labels_dir = DATA_ROOT / split / "labels"
        if not labels_dir.exists():
            print(f"Skipping {split} — labels dir not found")
            continue

        label_files = list(labels_dir.glob("*.txt"))
        print(f"\nProcessing {split}: {len(label_files)} label files")

        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()

            kept_lines = []
            removed_here = 0
            for line in lines:
                class_id = line.split()[0]
                if class_id == PERSON_CLASS_ID:
                    removed_here += 1
                else:
                    kept_lines.append(line)

            if removed_here > 0:
                with open(label_file, "w") as f:
                    f.writelines(kept_lines)
                total_files_modified += 1
                total_lines_removed += removed_here

                if len(kept_lines) == 0:
                    total_files_now_empty += 1

        print(f"  Done with {split}")

    print(f"\n=== Summary ===")
    print(f"Files modified: {total_files_modified}")
    print(f"'person' instances removed: {total_lines_removed}")
    print(f"Files now empty (no boxes left): {total_files_now_empty}")
    if total_files_now_empty > 0:
        print(
            f"  Note: these images now have no labels — consider removing them separately."
        )


def update_data_yaml():
    yaml_path = DATA_ROOT / "data.yaml"
    with open(yaml_path, "r") as f:
        content = f.read()

    content = content.replace(
        "nc: 3\nnames: ['head', 'helmet', 'person']", "nc: 2\nnames: ['head', 'helmet']"
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    print("\ndata.yaml updated: nc=2, names=['head', 'helmet']")


if __name__ == "__main__":
    clean_labels()
    update_data_yaml()
