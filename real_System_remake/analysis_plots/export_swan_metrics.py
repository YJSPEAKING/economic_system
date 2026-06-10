from pathlib import Path
import csv
import json
import re
import sys

from swanlab.data.porter.datastore import DataStore
from swanlab.proto.v0 import BaseModel


METRIC_KEYS = {
    "survival": "\u6bcf\u767e\u56de\u5408/\u5b58\u6d3b\u5929\u6570",
    "production": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u751f\u4ea7\u4f01\u4e1a1",
    "consumption": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u6d88\u8d39\u4f01\u4e1a1",
    "bank": "\u6bcf\u767e\u56de\u5408/\u7d2f\u8ba1\u5956\u52b1/\u94f6\u884c",
}


def seed_from_config(run_dir: Path):
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return ""
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"random_seed:\s*([0-9]+)", text)
    return match.group(1) if match else ""


def git_from_meta(run_dir: Path):
    meta_path = run_dir / "files" / "swanlab-metadata.json"
    if not meta_path.exists():
        return ""
    try:
        return json.dumps(
            json.loads(meta_path.read_text(encoding="utf-8", errors="ignore")).get("git_info"),
            ensure_ascii=False,
        )
    except Exception:
        return ""


def export_run(run_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {metric_id: {} for metric_id in METRIC_KEYS}
    description = ""
    backup_path = run_dir / "backup.swanlab"

    datastore = DataStore()
    datastore.open_for_scan(str(backup_path))
    try:
        while True:
            record = datastore.scan()
            if record is None:
                break
            try:
                data = BaseModel.from_record(record).model_dump()
            except Exception:
                continue

            desc = data.get("description") or data.get("notes")
            if isinstance(desc, str) and desc:
                description = desc

            config = data.get("config")
            if isinstance(config, dict):
                config_desc = config.get("description") or config.get("notes")
                if isinstance(config_desc, str) and config_desc:
                    description = config_desc

            key = data.get("key")
            if key not in METRIC_KEYS.values():
                continue

            metric = data.get("metric")
            value = metric.get("data") if isinstance(metric, dict) else metric
            try:
                step = int(data.get("step"))
                value = float(value)
            except Exception:
                continue

            for metric_id, metric_key in METRIC_KEYS.items():
                if key == metric_key:
                    metrics[metric_id][step] = value
                    break
    finally:
        try:
            datastore.close()
        except Exception:
            pass

    seed = seed_from_config(run_dir)
    csv_path = out_dir / f"{run_dir.name}_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "seed", "metric_id", "step", "value"])
        for metric_id, values in metrics.items():
            for step in sorted(values):
                writer.writerow([run_dir.name, seed, metric_id, step, values[step]])

    meta_path = out_dir / f"{run_dir.name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "seed": seed,
                "description": description,
                "git_info": git_from_meta(run_dir),
                "metric_counts": {metric_id: len(values) for metric_id, values in metrics.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(csv_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: export_swan_metrics.py <run_dir> <out_dir>")
    export_run(Path(sys.argv[1]), Path(sys.argv[2]))
