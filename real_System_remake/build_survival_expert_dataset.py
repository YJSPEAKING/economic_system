import argparse
import csv
import os
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "human_collect_web_data"


def read_csv_rows(path, has_header=True):
    encoding = "utf-8-sig" if has_header else "utf-8"
    with open(path, newline="", encoding=encoding) as file:
        if has_header:
            return list(csv.DictReader(file))
        return list(csv.reader(file))


def write_dict_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plain_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def resolve_path(raw_path, data_dir):
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = data_dir / path.name
    return candidate if candidate.exists() else path


def collect_good_episodes(data_dir, min_survival_days):
    good = {}
    summary_files = sorted(data_dir.glob("*_summary.csv"))
    for summary_path in summary_files:
        for row in read_csv_rows(summary_path, has_header=True):
            try:
                survival_days = float(row.get("survival_days", "0") or 0)
            except ValueError:
                survival_days = 0
            if survival_days <= min_survival_days:
                continue

            meta_path = resolve_path(row.get("meta_output_path"), data_dir)
            expert_path = resolve_path(row.get("expert_output_path"), data_dir)
            episode = str(row.get("episode", ""))
            if not meta_path or not expert_path or not meta_path.exists() or not expert_path.exists():
                continue

            key = (str(meta_path), str(expert_path))
            good.setdefault(key, set()).add(episode)
    return good, summary_files


def build_dataset(data_dir, output_prefix, min_survival_days):
    good, summary_files = collect_good_episodes(data_dir, min_survival_days)
    selected_meta_rows = []
    selected_expert_rows = []
    skipped_pairs = []

    for (meta_path_str, expert_path_str), episodes in good.items():
        meta_path = Path(meta_path_str)
        expert_path = Path(expert_path_str)
        meta_rows = read_csv_rows(meta_path, has_header=True)
        expert_rows = read_csv_rows(expert_path, has_header=False)
        if len(meta_rows) != len(expert_rows):
            skipped_pairs.append((meta_path.name, expert_path.name, len(meta_rows), len(expert_rows)))
            continue

        for meta_row, expert_row in zip(meta_rows, expert_rows):
            if str(meta_row.get("episode", "")) in episodes:
                selected_meta_rows.append(meta_row)
                selected_expert_rows.append(expert_row)

    expert_output = data_dir / f"{output_prefix}_expert.csv"
    meta_output = data_dir / f"{output_prefix}_meta.csv"

    if selected_meta_rows:
        write_plain_rows(expert_output, selected_expert_rows)
        write_dict_rows(meta_output, selected_meta_rows, list(selected_meta_rows[0].keys()))

    return {
        "summary_files": len(summary_files),
        "good_episode_groups": sum(len(episodes) for episodes in good.values()),
        "selected_rows": len(selected_expert_rows),
        "expert_output": expert_output,
        "meta_output": meta_output,
        "skipped_pairs": skipped_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter human expert rows from episodes whose survival_days is above a threshold."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Folder containing *_expert.csv, *_meta.csv and *_summary.csv.")
    parser.add_argument("--min-survival-days", type=float, default=90, help="Keep episodes with survival_days greater than this value.")
    parser.add_argument("--output-prefix", default="survival_gt90_human_expert", help="Output filename prefix.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    result = build_dataset(data_dir, args.output_prefix, args.min_survival_days)

    print(f"Data folder: {data_dir}")
    print(f"Summary files found: {result['summary_files']}")
    print(f"Eligible episodes: {result['good_episode_groups']}")
    print(f"Selected expert rows: {result['selected_rows']}")
    if result["selected_rows"]:
        print(f"Expert output: {result['expert_output']}")
        print(f"Meta output: {result['meta_output']}")
    else:
        print("No output written. Make sure participants click end/next episode or the episode finishes so *_summary.csv is created.")
    if result["skipped_pairs"]:
        print("Skipped mismatched file pairs:")
        for meta_name, expert_name, meta_count, expert_count in result["skipped_pairs"]:
            print(f"  {meta_name} / {expert_name}: meta rows={meta_count}, expert rows={expert_count}")


if __name__ == "__main__":
    main()
