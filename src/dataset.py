    #!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def count_in_split(split_dir: Path):
    counts = {}
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        cnt = sum(1 for f in class_dir.rglob('*') if f.suffix.lower() in EXTS)
        counts[class_dir.name] = cnt
    return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Path to Split Dataset (contains Train/Validation/Test)')
    args = parser.parse_args()
    root = Path(args.data_root)
    assert root.exists(), f"Path not found: {root}"

    total_counts = defaultdict(int)
    for split in ['Train','Validation','Test']:
        p = root / split
        if not p.exists():
            print(f"Warning: {p} missing.")
            continue
        counts = count_in_split(p)
        print(f"\n=== {split} ===")
        for cls, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{cls:20s}: {c}")
            total_counts[cls] += c

    print("\n=== Totals across splits ===")
    for cls, c in sorted(total_counts.items(), key=lambda x: -x[1]):
        print(f"{cls:20s}: {c}")

    # bar chart for totals
    classes = [c for c,_ in sorted(total_counts.items(), key=lambda x:-x[1])]
    values = [total_counts[c] for c in classes]
    plt.figure(figsize=(8,4))
    plt.bar(classes, values)
    plt.title('Total image counts per class')
    plt.ylabel('Num images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out = (root.parent / 'experiments' / 'class_counts.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"\nSaved bar chart to {out}")

if __name__ == '__main__':
    main()