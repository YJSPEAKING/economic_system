import csv
import glob
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import requests


BASE = "http://127.0.0.1:8501"
PASSWORD = "123"
TARGET_SUCCESS = int(os.environ.get("CODEX_BATCH_TARGET_SUCCESS", "1700"))
MAX_ATTEMPTS = int(os.environ.get("CODEX_BATCH_MAX_ATTEMPTS", str(TARGET_SUCCESS + 25)))
DATA_DIR = os.path.join("real_System_remake", "human_collect_web_data")
PROGRESS_LOG = os.environ.get(
    "CODEX_BATCH_PROGRESS_LOG",
    os.path.join(os.getcwd(), "codex_humanlike_batch_runner_progress.log"),
)


def log(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def read_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def load_success_profiles():
    profiles = []
    for meta_path in sorted(glob.glob(os.path.join(DATA_DIR, "*meta.csv"))):
        expert_path = meta_path[:-8] + "expert.csv"
        if not os.path.exists(expert_path):
            continue
        with open(meta_path, encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row.get("participant_id", ""), row.get("episode", ""))].append(row)
        for key, group_rows in grouped.items():
            try:
                max_day = max(int(float(row.get("day", 0))) for row in group_rows)
            except Exception:
                max_day = len(group_rows)
            if len(group_rows) >= 90 or max_day >= 90:
                group_rows.sort(key=lambda row: int(float(row.get("day", 0) or 0)))
                profiles.append(
                    {
                        "source": os.path.basename(meta_path),
                        "key": key,
                        "rows": group_rows,
                        "max_day": max_day,
                    }
                )
    if not profiles:
        raise RuntimeError("没有找到可学习的存活超过90天的 expert/meta 数据。")
    return profiles


def post(session, path, payload, timeout=30, retries=4):
    last_error = None
    for attempt in range(retries):
        try:
            response = session.post(BASE + path, json=payload, timeout=timeout)
            data = response.json()
            if response.status_code >= 400 or data.get("error"):
                raise RuntimeError(f"{path} failed status={response.status_code}: {data}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(0.35 + attempt * 0.5)
    raise RuntimeError(f"{path} failed after retries: {last_error}")


def clamp(value, low, high):
    return max(low, min(high, value))


def rounded(value):
    return round(float(value), 4)


def rounded_inside_limit(value, low, high):
    value = clamp(float(value), low, high)
    if value >= high:
        value = math.floor((high - 1e-9) * 10000.0) / 10000.0
    elif value <= low:
        value = 0.0 if abs(low) < 1e-12 else math.ceil((low + 1e-9) * 10000.0) / 10000.0
    else:
        value = round(value, 4)
        if value > high:
            value = math.floor((high - 1e-9) * 10000.0) / 10000.0
        if value < low:
            value = math.ceil((low + 1e-9) * 10000.0) / 10000.0
    return round(float(value), 4)


def module_items(state, title_contains):
    for module in state.get("dashboard", {}).get("modules", []):
        if title_contains in module.get("title", ""):
            return {item.get("label"): float(item.get("raw") or 0.0) for item in module.get("items", [])}
    return {}


def limit_value(limits, index, fallback):
    try:
        low = float(limits[index].get("min", 0.0))
        high = float(limits[index].get("max", fallback))
    except Exception:
        low, high = 0.0, float(fallback)
    return low, high


def bounded(state, values):
    limits = state.get("limits") or []
    out = []
    for index, value in enumerate(values):
        low, high = limit_value(limits, index, value)
        out.append(rounded_inside_limit(value, low, high))
    return out


def guide_for_day(profile, day):
    rows = profile["rows"]
    if not rows:
        return None
    index = max(0, min(len(rows) - 1, day - 1))
    return rows[index]


def decide(state, memory, rng):
    day = int(state.get("day") or 1)
    defaults = [float(value or 0.0) for value in (state.get("defaults") or [0, 5, 5, 8])]
    limits = state.get("limits") or []
    loan_info = module_items(state, "贷款")
    material_info = module_items(state, "原料数量")
    price_info = module_items(state, "定价")

    cash = loan_info.get("当前现金", 0.0)
    due = loan_info.get("今天还款", 0.0)
    debt = loan_info.get("总欠款", 0.0)
    got_a = material_info.get("甲公司-A", defaults[1])
    got_b = material_info.get("甲公司-B", defaults[2])
    cons_a = material_info.get("乙公司-A", 0.0)
    third_a = price_info.get("A第三方市场定价", 100.0) or 100.0
    sold = price_info.get("产品A昨天售出", 0.0)
    produced = price_info.get("产品A昨天产出", 0.0)
    stock = price_info.get("产品A当前库存", 0.0)

    a_base = max(0.01, defaults[1])
    b_base = max(0.01, defaults[2])
    price_base = max(0.01, defaults[3])
    sell_ratio = sold / max(produced, 1e-6)
    stock_ratio = stock / max(produced, 1.0)
    material_balance = min(got_a, got_b) / max(got_a, got_b, 1e-6)

    guide = guide_for_day(memory["profile"], day)
    guide_a = read_float(guide, "human_原料A采购需求", a_base) if guide else a_base
    guide_b = read_float(guide, "human_原料B采购需求", b_base) if guide else b_base
    guide_price = read_float(guide, "human_产品A销售价格", price_base) if guide else price_base

    if "scale_bias" not in memory:
        memory["scale_bias"] = rng.uniform(0.94, 1.06)
    if "price_bias" not in memory:
        memory["price_bias"] = rng.uniform(-0.010, 0.012)

    guide_qty = max(4.4, (guide_a + guide_b) / 2.0) * memory["scale_bias"]
    current_qty = (a_base + b_base) / 2.0
    target_qty = 0.72 * guide_qty + 0.28 * current_qty

    if sell_ratio >= 0.84 and stock_ratio <= 2.1 and cash > max(40.0, due * 1.45):
        target_qty += rng.uniform(0.00, 0.16)
    if sell_ratio < 0.58 or stock_ratio > 2.9:
        target_qty -= rng.uniform(0.08, 0.26)
    if cash < max(30.0, due * 1.25):
        target_qty -= rng.uniform(0.04, 0.16)
    if material_balance < 0.86:
        target_qty -= rng.uniform(0.03, 0.10)
    if day % rng.choice([8, 11, 13, 17]) == 0:
        target_qty += rng.uniform(-0.22, 0.24)
    if day > 25:
        target_qty = max(target_qty, 5.0 + min(3.0, (day - 25) * 0.045))
    target_qty = clamp(target_qty + rng.gauss(0.0, 0.07), 4.5, 13.2)

    mismatch = rng.gauss(0.0, 0.010)
    a_need = target_qty * (1.0 + mismatch)
    b_need = target_qty * (1.0 - mismatch)
    a_low, a_high = limit_value(limits, 1, a_need)
    b_low, b_high = limit_value(limits, 2, b_need)
    common_low = max(a_low, b_low)
    common_high = min(a_high, b_high, 14.0)
    common = clamp((a_need + b_need) / 2.0, common_low, common_high)
    a_need = clamp(common * (1.0 + mismatch * 0.35), a_low, a_high)
    b_need = clamp(common * (1.0 - mismatch * 0.35), b_low, b_high)

    loan = 0.0
    loan_low, loan_high = limit_value(limits, 0, cash)
    guide_loan = read_float(guide, "human_申请贷款金额", 0.0) if guide else 0.0
    if cash > 0 and due > cash * 0.72:
        loan = min(loan_high, max(0.0, due * rng.uniform(1.06, 1.24) - cash))
    elif guide_loan > 0 and rng.random() < 0.08:
        loan = min(loan_high, guide_loan * rng.uniform(0.35, 0.85))
    elif cash > 0 and cash < 35 and debt < cash * 1.8 and rng.random() < 0.16:
        loan = min(loan_high, cash * rng.uniform(0.02, 0.08))
    loan = clamp(loan, loan_low, loan_high)

    adaptive_price = price_base * (1.045 + memory["price_bias"] + rng.gauss(0.0, 0.012))
    if sell_ratio >= 0.86 and stock_ratio <= 2.0:
        adaptive_price *= rng.uniform(1.004, 1.020)
    if sell_ratio < 0.58 or stock_ratio > 2.8:
        adaptive_price *= rng.uniform(0.94, 0.985)
    if cons_a < max(1.0, sold * 0.45):
        adaptive_price *= rng.uniform(0.985, 1.000)
    if day > 60:
        adaptive_price *= rng.uniform(0.982, 1.006)

    price = 0.35 * guide_price * rng.uniform(0.985, 1.020) + 0.65 * adaptive_price
    price_low, price_high = limit_value(limits, 3, price)
    third_cap = third_a * (rng.uniform(0.82, 0.94) if price_base <= 78 else rng.uniform(0.82, 0.90))
    if target_qty < 6.0 and day > 45:
        third_cap = min(third_cap, 84.0)
    price = clamp(price, price_low, min(price_high, third_cap, 96.0))
    if stock_ratio > 2.0 and day % rng.choice([13, 17, 19]) == 0:
        price = max(price_low, price * rng.uniform(0.985, 0.998))

    return bounded(state, [loan, a_need, b_need, price])


def main():
    profiles = load_success_profiles()
    source_files = len({profile["source"] for profile in profiles})
    participant_id = "codex_humanlike1700_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    participant_info = {
        "age_group": "20-40岁",
        "education": "硕士及以上",
        "gender": "男",
        "econ_background": "比较熟悉",
        "strategy_experience": "经常接触",
    }
    rng = random.Random(20260602)
    session = requests.Session()
    log(f"loaded_profiles={len(profiles)} source_meta_files={source_files}")
    log(f"participant_id={participant_id}")
    data = post(
        session,
        "/api/start",
        {
            "participant_id": participant_id,
            "password": PASSWORD,
            "participant_info": participant_info,
        },
    )
    session_id = data["session_id"]
    output_path = data.get("output_path")
    meta_output_path = data.get("meta_output_path")
    summary_output_path = data.get("summary_output_path")
    log(f"session_id={session_id}")
    log(f"expert_output={output_path}")
    log(f"meta_output={meta_output_path}")
    log(f"summary_output={summary_output_path}")

    successes = 0
    attempts = 0
    started_at = time.perf_counter()
    try:
        while successes < TARGET_SUCCESS and attempts < MAX_ATTEMPTS:
            attempts += 1
            profile = rng.choice(profiles)
            memory = {
                "profile": profile,
                "profile_name": f"{profile['source']}:{profile['key']}",
            }
            state = data["state"]
            episode_last_values = None
            while True:
                values = decide(state, memory, rng)
                episode_last_values = values
                data = post(session, "/api/step", {"session_id": session_id, "values": values}, timeout=60)
                if data.get("done"):
                    survival = int(data.get("survival_days") or 0)
                    rows = int(data.get("rows_saved") or 0)
                    ok = survival > 90
                    if ok:
                        successes += 1
                    elapsed = time.perf_counter() - started_at
                    log(
                        "episode=%d survival=%d success=%s successes=%d rows=%d elapsed_min=%.2f profile=%s last_values=%s"
                        % (
                            attempts,
                            survival,
                            ok,
                            successes,
                            rows,
                            elapsed / 60.0,
                            memory["profile_name"],
                            episode_last_values,
                        )
                    )
                    break
                state = data["state"]
            if successes < TARGET_SUCCESS and attempts < MAX_ATTEMPTS:
                data = post(session, "/api/next_episode", {"session_id": session_id})
        if successes < TARGET_SUCCESS:
            raise RuntimeError(f"only {successes}/{TARGET_SUCCESS} successes after {attempts} attempts")
    finally:
        try:
            end = post(session, "/api/end", {"session_id": session_id})
            log("ended_stats=" + json.dumps(end.get("stats", {}), ensure_ascii=False))
        except Exception as exc:
            log(f"end_failed={exc}")
    log(f"completed successes={successes} attempts={attempts}")


if __name__ == "__main__":
    main()
