import csv
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Number
from urllib.parse import urlparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from real_System_remake.human_collect_production1 import (
    ACTION_DISPLAY_NAMES,
    DEFAULT_AUTO_POLICY,
    DEFAULT_SEED,
    HumanProductionCollector,
    PARTICIPANT_INFO_FIELDS,
    PARTICIPANT_META_FIELDS,
    display_rows,
    format_number,
    raw_state_value,
    risk_tag,
    row_change_tag,
)


HOST = "0.0.0.0"
PORT = 8501
SERVER_AUTO_POLICY = os.environ.get("HUMAN_COLLECT_POLICY", DEFAULT_AUTO_POLICY)
ACCESS_PASSWORD = os.environ.get("HUMAN_COLLECT_PASSWORD", "")
WEB_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_collect_web_data")
BLOCK_MIN_HUMAN_DAYS = 8
DYNAMIC_SKIP_MAX_DAYS = 15
DYNAMIC_REL_CHANGE = 0.25
DYNAMIC_PRICE_REL_CHANGE = 0.15
DYNAMIC_ABS_CHANGE = {
    "Cash": 50.0,
    "Product A inventory": 10.0,
    "Total debt": 50.0,
    "Payment due today": 20.0,
    "Raw Material A reference quantity": 2.0,
    "Raw Material B reference quantity": 2.0,
    "Product A sale price": 1.0,
}
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
PARTICIPANT_INFO_OPTIONS = {
    "age_group": ["20岁以下", "20-40岁", "40岁以上"],
    "education": ["高中及以下", "大专/本科", "硕士及以上"],
    "gender": ["男", "女"],
    "econ_background": ["几乎没有", "学过一点", "比较熟悉"],
    "strategy_experience": ["几乎没有", "偶尔接触", "经常接触"],
}
PARTICIPANT_INFO_LABELS = {
    "age_group": "an age group",
    "education": "an education level",
    "gender": "a gender",
    "econ_background": "an economics/management background level",
    "strategy_experience": "a business/strategy game experience level",
}


def safe_name(value):
    value = (value or "anonymous").strip()
    value = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value)
    return value[:40] or "anonymous"


def session_paths(participant_id, session_id):
    os.makedirs(WEB_DATA_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{safe_name(participant_id)}_{stamp}_{session_id[:8]}"
    return (
        os.path.join(WEB_DATA_DIR, f"{prefix}_expert.csv"),
        os.path.join(WEB_DATA_DIR, f"{prefix}_meta.csv"),
        os.path.join(WEB_DATA_DIR, f"{prefix}_summary.csv"),
    )


def normalize_participant_info(payload):
    raw_info = payload.get("participant_info") or {}
    normalized = {}
    for key, _ in PARTICIPANT_INFO_FIELDS:
        value = str(raw_info.get(key, "")).strip()
        options = PARTICIPANT_INFO_OPTIONS.get(key, [])
        if value not in options:
            raise ValueError(f"Please select {PARTICIPANT_INFO_LABELS.get(key, 'an option')}.")
        normalized[key] = value
    return normalized


def default_human_values(state):
    return [
        max(0.0, raw_state_value(state, 7)),
        max(0.0, raw_state_value(state, 11)),
        max(0.0, raw_state_value(state, 12)),
        max(0.01, raw_state_value(state, 6)),
    ]


def action_hints(state):
    cash = raw_state_value(state, 0)
    k_base = raw_state_value(state, 11)
    l_base = raw_state_value(state, 12)
    price_base = raw_state_value(state, 6)
    return [
        f"Cash: {format_number(cash)}. Loan range: 0-{format_number(cash)}.",
        f"Use A and B together. Previous reference: A {format_number(k_base)}, B {format_number(l_base)}. Excess A may be wasted.",
        f"Use A and B together. Previous reference: B {format_number(l_base)}, A {format_number(k_base)}. Excess B may be wasted.",
        "Balance cost and profit with Company B's affordability and stable cooperation.",
    ]


def input_number(value):
    value = float(value)
    return round(value, 4)


def action_limits(state):
    cash = max(0.0, raw_state_value(state, 0))
    k_base = raw_state_value(state, 11)
    l_base = raw_state_value(state, 12)
    price_base = raw_state_value(state, 6)

    def quantity_limits(base):
        if base <= 0:
            return {"min": 0.0, "max": 10.0}
        return {"min": max(0.0, base * 0.5), "max": max(0.0, base * 1.5)}

    price_min = max(0.0, price_base * 0.5)
    price_max = max(price_min, price_base * 1.5)
    return [
        {"min": 0.0, "max": cash},
        quantity_limits(k_base),
        quantity_limits(l_base),
        {"min": price_min, "max": price_max},
    ]


def quantity_to_action(value, base, name):
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    if base <= 0:
        action = value / 10 - 0.5
        if not -0.5 <= action <= 0.5:
            raise ValueError(f"The current reference for {name} is 0. Enter a value between 0 and 10.")
        return action
    action = value / base - 1
    if not -0.5 <= action <= 0.5:
        raise ValueError(f"{name} must be between {format_number(base * 0.5)} and {format_number(base * 1.5)}.")
    return action


def human_to_model_action(state, values):
    if len(values) != 4:
        raise ValueError("Please enter all four action values.")
    if any(value < 0 for value in values):
        raise ValueError("Please do not enter negative values.")
    cash = raw_state_value(state, 0)
    k_base = raw_state_value(state, 11)
    l_base = raw_state_value(state, 12)
    price_base = raw_state_value(state, 6)
    loan, k_need, l_need, price = values

    if cash <= 0:
        if loan > 0:
            raise ValueError("Current cash is 0, so the loan request must be 0.")
        loan_action = -0.5
    else:
        if loan > cash:
            raise ValueError(f"The loan request cannot exceed current cash ({format_number(cash)}).")
        loan_action = loan / cash - 0.5

    k_action = quantity_to_action(k_need, k_base, "Raw Material A purchase demand")
    l_action = quantity_to_action(l_need, l_base, "Raw Material B purchase demand")
    if price_base <= 0:
        raise ValueError("The current price reference is invalid, so the price action cannot be submitted.")
    price_action = price / price_base - 1
    if not -0.5 <= price_action <= 0.5:
        raise ValueError(f"The sale price must be between {format_number(price_base * 0.5)} and {format_number(price_base * 1.5)}.")
    return [loan_action, k_action, l_action, price_action]


def clamp_human_values_for_state(state, values):
    if len(values) != 4:
        raise ValueError("Please enter all four action values.")
    limits = action_limits(state)
    clamped = []
    for value, limit in zip(values, limits):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Each action value must be a valid number.")
        min_value = float(limit.get("min", 0.0))
        max_value = limit.get("max")
        value = max(min_value, value)
        if max_value is not None:
            value = min(float(max_value), value)
        clamped.append(value)
    return clamped


def raw_or_zero(state, index):
    try:
        return raw_state_value(state, index)
    except Exception:
        return 0.0


def agent_raw_value(full_state, agent_key, index):
    if not full_state or agent_key not in full_state:
        return None
    return raw_or_zero(full_state[agent_key], index)


def signed_number(value):
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{format_number(value)}"


def production_metrics(state):
    k_local_pair = (raw_or_zero(state, 13), raw_or_zero(state, 14))
    l_local_pair = (raw_or_zero(state, 15), raw_or_zero(state, 16))
    k_third_pair = (raw_or_zero(state, 17), raw_or_zero(state, 18))
    l_third_pair = (raw_or_zero(state, 19), raw_or_zero(state, 20))
    k_pairs = [k_local_pair, k_third_pair]
    l_pairs = [l_local_pair, l_third_pair]
    purchase_spend = sum(price * num for price, num in k_pairs + l_pairs)
    revenue = raw_or_zero(state, 3) * raw_or_zero(state, 5)
    return {
        "k_local_pair": k_local_pair,
        "l_local_pair": l_local_pair,
        "k_third_pair": k_third_pair,
        "l_third_pair": l_third_pair,
        "k_total": k_local_pair[1] + k_third_pair[1],
        "l_total": l_local_pair[1] + l_third_pair[1],
        "purchase_spend": purchase_spend,
        "revenue": revenue,
        "net": revenue - purchase_spend,
    }


def consumption_purchase_metrics(state):
    k_local_pair = (raw_or_zero(state, 21), raw_or_zero(state, 22))
    l_local_pair = (raw_or_zero(state, 23), raw_or_zero(state, 24))
    k_third_pair = (raw_or_zero(state, 25), raw_or_zero(state, 26))
    l_third_pair = (raw_or_zero(state, 27), raw_or_zero(state, 28))
    return {
        "k_total": k_local_pair[1] + k_third_pair[1],
        "l_total": l_local_pair[1] + l_third_pair[1],
        "k_local_pair": k_local_pair,
        "l_local_pair": l_local_pair,
        "k_third_pair": k_third_pair,
        "l_third_pair": l_third_pair,
    }


def profit_line_charts(collector):
    snapshots = getattr(collector, "state_history", [])
    if not snapshots:
        return []

    def cash_of(snapshot, agent_key):
        full_state = snapshot.get("full_state", {})
        return agent_raw_value(full_state, agent_key, 0)

    charts = []
    for agent_key, title in (("production1", "Company A Daily Net Profit"), ("consumption1", "Company B Daily Net Profit")):
        points = []
        previous_cash = None
        for snapshot in snapshots:
            cash = cash_of(snapshot, agent_key)
            if cash is None:
                continue
            if previous_cash is None:
                previous_cash = cash
                continue
            profit = cash - previous_cash
            points.append({"day": len(points) + 1, "value": round(profit, 4)})
            previous_cash = cash
        charts.append({"title": title, "points": points})
    return charts


def purchase_breakdown_text(title, local_pair, third_pair):
    local_price, local_num = local_pair
    third_price, third_num = third_pair
    total_num = local_num + third_num
    total_spend = local_price * local_num + third_price * third_num
    return (
        f"{title}: regular market quantity {format_number(local_num)} at unit price {format_number(local_price)}; "
        f"third-party market quantity {format_number(third_num)} at listed price {format_number(third_price)}; "
        f"total quantity {format_number(total_num)}, total cost {format_number(total_spend)}."
    )


def business_warning_lines(state, prod, cons):
    warnings = []
    previous_price = raw_or_zero(state, 5)
    third_a_price = cons["k_third_pair"][0] or raw_or_zero(state, 30)
    third_a_num = cons["k_third_pair"][1]
    sold = raw_or_zero(state, 3)
    stock = raw_or_zero(state, 1)

    if third_a_price > 0 and previous_price > third_a_price:
        warnings.append(
            f"Yesterday, Product A was cheaper in the third-party market (our price: {format_number(previous_price)}; "
            f"third-party price: {format_number(third_a_price)}). Because the system buys from the lower-priced source first, "
            "Company B may prioritize the third-party market, which may reduce our Product A sales."
        )
    if third_a_num > 0 and sold <= 1e-9 and stock > 0:
        warnings.append(
            f"Yesterday, Company B bought {format_number(third_a_num)} units of Product A from the third-party market, "
            "while our Product A sales were 0. This usually means that external supply met Company B's demand and "
            "our Product A did not sell."
        )

    a_total = prod["k_total"]
    b_total = prod["l_total"]
    if max(a_total, b_total) > 0:
        imbalance = abs(a_total - b_total)
        if imbalance >= 1 and imbalance / max(a_total, b_total) >= 0.1:
            more_name, less_name = ("Raw Material A", "Raw Material B") if a_total > b_total else ("Raw Material B", "Raw Material A")
            warnings.append(
                f"Yesterday, the amounts of Raw Materials A and B that we actually purchased were unbalanced "
                f"({more_name} exceeded {less_name} by {format_number(imbalance)}). Product A output is limited by the smaller "
                f"amount of {less_name}. Excess {more_name} cannot be stored overnight, so it does not contribute to output."
            )

    shortages = []
    for name, need, got in (
        ("Raw Material A", raw_or_zero(state, 11), a_total),
        ("Raw Material B", raw_or_zero(state, 12), b_total),
    ):
        shortage = need - got
        if need > 0 and shortage >= 1 and got < need * 0.8:
            shortages.append(f"{name} was short by {format_number(shortage)}")
    if shortages:
        warnings.append(
            "Yesterday, actual raw-material purchases were below the requested quantities (" + "; ".join(shortages) +
            "). Cash constraints or market allocation may have caused the shortfall. Product A output will be based on the quantities actually purchased."
        )

    debt_due = raw_or_zero(state, 9) + raw_or_zero(state, 10)
    cash = raw_or_zero(state, 0)
    if debt_due > cash:
        warnings.append(
            f"Today's payment of {format_number(debt_due)} exceeds current cash of {format_number(cash)}. "
            "If today's loan and operating cash flow do not cover the payment, the episode may end."
        )
    return warnings[:4]


def production_flow_detail_html():
    return """
      <div class="production-flow-detail" aria-label="Product production and market flow diagram">
        <div class="flow-market-row">
          <div class="flow-node flow-market">
            <strong>Regular Market</strong>
            <span>Holds Raw Materials A and B carried into the market from the previous day</span>
            <small>Raw Material A comes from Company A's Product A; Raw Material B comes from Company B's Product B</small>
          </div>
          <div class="flow-node flow-third-market">
            <strong>Third-Party Market</strong>
            <span>Provides supplementary Raw Materials A and B</span>
            <small>Companies may purchase additional materials here when regular-market supply is insufficient or its price is unfavorable</small>
          </div>
        </div>

        <div class="flow-lanes">
          <div class="flow-lane">
            <div class="flow-lane-title">Company A Production Cycle</div>
            <div class="flow-node">
              <strong>Raw Material A + Raw Material B</strong>
              <span>Company A purchases from the regular and third-party markets</span>
              <small>Raw materials are purchased and used on the same day and cannot be stored overnight.</small>
            </div>
            <div class="flow-arrow"><span>Inputs to Company A</span></div>
            <div class="flow-node flow-company-a">
              <strong>Company A</strong>
              <span>Produces using Raw Materials A and B</span>
              <small>Product A output = 2.5 × min(Raw Material A purchased, Raw Material B purchased).</small>
            </div>
            <div class="flow-arrow"><span>Produced today</span></div>
            <div class="flow-node flow-product-a">
              <strong>Product A</strong>
              <span>Enters the regular market the next day as Raw Material A</span>
            </div>
          </div>

          <div class="flow-lane">
            <div class="flow-lane-title">Company B Production Cycle</div>
            <div class="flow-node">
              <strong>Raw Material A + Raw Material B</strong>
              <span>Company B also purchases from the regular and third-party markets</span>
              <small>Company B's actions are determined automatically by the system.</small>
            </div>
            <div class="flow-arrow"><span>Inputs to Company B</span></div>
            <div class="flow-node flow-company-b">
              <strong>Company B</strong>
              <span>Produces using Raw Materials A and B</span>
            </div>
            <div class="flow-arrow"><span>Produced today</span></div>
            <div class="flow-node flow-product-b">
              <strong>Product B</strong>
              <span>Enters the regular market the next day as Raw Material B</span>
            </div>
          </div>
        </div>
      </div>
    """


def dashboard_payload(collector, state, day, previous_state=None, full_state=None, previous_full_state=None):
    prod = production_metrics(state)
    cons = consumption_purchase_metrics(state)
    cash = raw_or_zero(state, 0)
    previous_cash = raw_or_zero(previous_state, 0) if previous_state is not None else None
    cash_delta = cash - previous_cash if previous_cash is not None else None
    actual_loan = raw_or_zero(state, 8)
    consumption_cash = agent_raw_value(full_state, "consumption1", 0)
    previous_consumption_cash = agent_raw_value(previous_full_state, "consumption1", 0)
    consumption_cash_delta = (
        consumption_cash - previous_consumption_cash
        if consumption_cash is not None and previous_consumption_cash is not None
        else None
    )

    if previous_state is None:
        summary_lines = [
            "This is the first day, so there is no net-profit record from the previous day.",
            f"Company A (your company) currently has {format_number(cash)} in cash. Decide how much to borrow, how much of Raw Materials A and B to purchase, and the sale price of Product A."
        ]
        summary_warnings = []
    else:
        summary_lines = [
            f"Yesterday, Company A (your company) had a net profit of {signed_number(cash_delta)}. Current cash is {format_number(cash)}.",
            (
                f"Company B's net profit yesterday was {signed_number(consumption_cash_delta)}. If its position continues to worsen, its ability to purchase Product A may be affected."
                if consumption_cash_delta is not None
                else "Company B's net profit for yesterday is not available yet. Check again on the next day to see whether its position improves or worsens."
            ),
        ]
        summary_warnings = business_warning_lines(state, prod, cons)

    modules = [
        {
            "title": "Loan Information",
            "unit": "Amount",
            "items": [
                {"label": "Current Cash", "value": format_number(cash), "raw": cash},
                {"label": "Payment Due Today", "value": format_number(raw_or_zero(state, 9) + raw_or_zero(state, 10)), "raw": raw_or_zero(state, 9) + raw_or_zero(state, 10)},
                {"label": "Total Debt", "value": format_number(raw_or_zero(state, 2)), "raw": raw_or_zero(state, 2)},
                {"label": "Loan Received Yesterday", "value": format_number(actual_loan), "raw": actual_loan},
            ],
        },
        {
            "title": "Raw Materials Purchased by Both Companies Yesterday",
            "items": [
                {"label": "Company A - A", "value": format_number(prod["k_total"]), "raw": prod["k_total"]},
                {"label": "Company A - B", "value": format_number(prod["l_total"]), "raw": prod["l_total"]},
                {"label": "Company B - A", "value": format_number(cons["k_total"]), "raw": cons["k_total"]},
                {"label": "Company B - B", "value": format_number(cons["l_total"]), "raw": cons["l_total"]},
            ],
        },
        {
            "title": "Pricing and Sales Information",
            "items": [
                {"label": "A Regular Market Price", "value": format_number(raw_or_zero(state, 29)), "raw": raw_or_zero(state, 29)},
                {"label": "A Third-Party Market Price", "value": format_number(raw_or_zero(state, 30)), "raw": raw_or_zero(state, 30)},
                {"label": "B Regular Market Price", "value": format_number(raw_or_zero(state, 31)), "raw": raw_or_zero(state, 31)},
                {"label": "B Third-Party Market Price", "value": format_number(raw_or_zero(state, 32)), "raw": raw_or_zero(state, 32)},
                {"label": "Product A Sold Yesterday", "value": format_number(raw_or_zero(state, 3)), "raw": raw_or_zero(state, 3)},
                {"label": "Product A Produced Yesterday", "value": format_number(raw_or_zero(state, 4)), "raw": raw_or_zero(state, 4)},
                {"label": "Current Product A Inventory", "value": format_number(raw_or_zero(state, 1)), "raw": raw_or_zero(state, 1)},
            ],
        },
    ]

    charts = []

    details = [
        {
            "title": "Product Production and Market Flow",
            "html": production_flow_detail_html(),
        },
        {
            "title": "Market Purchasing Rules",
            "lines": [
                "The regular market and the third-party market may offer the same type of goods at the same time. The third-party market can be understood as a government macroeconomic regulator.",
                "The third-party market has a fixed price of 100 and sufficient supply, preventing a complete shortage of goods.",
                "The system purchases from the lower-priced source first. If the regular market cannot meet demand, it purchases the remainder from the next source.",
                "When the available quantity of a good is insufficient, the system allocates it in proportion to demand."
            ],
        },
        {
            "title": "Six Simulation Stages (Completed Automatically)",
            "lines": [
                "P1 Enterprise decision stage: Company A determines its loan request, purchase demands for Raw Materials A and B, and Product A price.",
                "P2 Bank decision stage: The bank determines available credit based on Company A, Company B, and market conditions.",
                "P3 Loan disbursement stage: The bank disburses loans to the companies. Loan principal is due after five days by default, and interest accrues and is paid each day during the loan term.",
                "P4 Goods trading stage: Companies purchase raw materials and sell products in the market.",
                "P5 Goods production stage: Company A uses the purchased Raw Materials A and B to produce Product A for sale the next day.",
                "P6 Settlement stage: The system settles repayments, interest, cash, and bankruptcy status, then advances to the next day.",
            ],
        },
        {
            "title": "Previous-Day Transaction Details",
            "lines": [
                purchase_breakdown_text("Company A purchased Raw Material A", prod["k_local_pair"], prod["k_third_pair"]),
                purchase_breakdown_text("Company A purchased Raw Material B", prod["l_local_pair"], prod["l_third_pair"]),
                purchase_breakdown_text("Company B purchased Raw Material A", cons["k_local_pair"], cons["k_third_pair"]),
                purchase_breakdown_text("Company B purchased Raw Material B", cons["l_local_pair"], cons["l_third_pair"]),
            ],
        },
    ]

    return {
        "summary": {"title": f"Day {day} Business Summary", "lines": summary_lines, "warnings": summary_warnings},
        "modules": modules,
        "lineCharts": profit_line_charts(collector),
        "charts": charts,
        "details": details,
    }


def serialize_state(collector, state=None, day=None, readonly=False,
                    previous_state_override=None, full_state_override=None,
                    previous_full_state_override=None):
    if state is None:
        state = collector.current_state()
    if day is None:
        day = collector.env.day
    full_state = full_state_override if full_state_override is not None else collector.state
    previous_state = None
    previous_full_state = None
    if previous_state_override is not None:
        previous_state = previous_state_override
        previous_full_state = previous_full_state_override
    elif not readonly and collector.history:
        previous_state = collector.history[-1]["state"]
        previous_full_state = collector.history[-1].get("full_state")
    rows = []
    for group, name, value, key in display_rows(state, day=day):
        tags = list(row_change_tag(key, value, previous_state) or risk_tag(key, value, state))
        rows.append(
            {
                "group": group,
                "name": name,
                "value": format_number(value),
                "change": change_text(key, value, previous_state),
                "tags": tags,
            }
        )
    return {
        "day": day,
        "rows": rows,
        "defaults": [input_number(value) for value in default_human_values(state)],
        "hints": action_hints(state),
        "limits": action_limits(state),
        "dashboard": dashboard_payload(
            collector=collector,
            state=state,
            day=day,
            previous_state=previous_state,
            full_state=full_state,
            previous_full_state=previous_full_state,
        ),
    }


def change_text(key, value, previous_state):
    if previous_state is None:
        return "-"
    if not isinstance(value, Number):
        return "-"
    previous = {row[3]: row[2] for row in display_rows(previous_state)}
    if key not in previous:
        return "-"
    if not isinstance(previous[key], Number):
        return "-"
    delta = value - previous[key]
    if abs(delta) < 1e-9:
        return "No change"
    sign = "+" if delta > 0 else ""
    return f"{sign}{format_number(delta)}"


def session_stats(session):
    elapsed = time.perf_counter() - session["started_at"]
    durations = session["durations"]
    count = len(durations)
    avg = sum(durations) / count if count else 0.0
    throughput = count / (elapsed / 60) if elapsed > 0 else 0.0
    return {
        "elapsed_seconds": round(elapsed, 3),
        "submitted": count,
        "avg_seconds": round(avg, 3),
        "throughput_per_minute": round(throughput, 3),
    }


def write_episode_summary(session, status, survival_days=None):
    durations = list(session.get("episode_durations", []))
    if not durations:
        return
    summary_path = session["summary_output_path"]
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    exists = os.path.exists(summary_path)
    total_seconds = sum(durations)
    count = len(durations)
    avg_seconds = total_seconds / count if count else 0.0
    throughput = count / (total_seconds / 60) if total_seconds > 0 else 0.0
    wall_seconds = time.perf_counter() - session.get("episode_started_at", session["started_at"])
    wall_throughput = count / (wall_seconds / 60) if wall_seconds > 0 else 0.0
    collector = session["collector"]

    columns = [
        "participant_id",
        *[f"participant_{key}" for key, _ in PARTICIPANT_META_FIELDS],
        "seed",
        "episode",
        "status",
        "survival_days",
        "human_decision_days",
        "episode_decision_seconds",
        "avg_seconds_per_decision",
        "decisions_per_minute",
        "episode_wall_seconds",
        "wall_decisions_per_minute",
        "rows_saved_total",
        "auto_policy",
        "expert_output_path",
        "meta_output_path",
    ]
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "participant_id": session["participant_id"],
            **{
                f"participant_{key}": session.get("participant_info", {}).get(key, "")
                for key, _ in PARTICIPANT_META_FIELDS
            },
            "seed": collector.seed,
            "episode": collector.env.episode,
            "status": status,
            "survival_days": survival_days if survival_days is not None else collector.env.day,
            "human_decision_days": count,
            "episode_decision_seconds": round(total_seconds, 3),
            "avg_seconds_per_decision": round(avg_seconds, 3),
            "decisions_per_minute": round(throughput, 3),
            "episode_wall_seconds": round(wall_seconds, 3),
            "wall_decisions_per_minute": round(wall_throughput, 3),
            "rows_saved_total": collector.rows_saved,
            "auto_policy": session["auto_policy"],
            "expert_output_path": session["output_path"],
            "meta_output_path": session["meta_output_path"],
        })
    session["episode_summary_written"] = True


def decision_metrics(state):
    return {
        "Cash": raw_state_value(state, 0),
        "Product A inventory": raw_state_value(state, 1),
        "Total debt": raw_state_value(state, 2),
        "Payment due today": raw_state_value(state, 9) + raw_state_value(state, 10),
        "Raw Material A reference quantity": raw_state_value(state, 11),
        "Raw Material B reference quantity": raw_state_value(state, 12),
        "Product A sale price": raw_state_value(state, 6),
    }


def metric_change_reasons(anchor_state, current_state):
    old = decision_metrics(anchor_state)
    new = decision_metrics(current_state)
    reasons = []
    for name, old_value in old.items():
        new_value = new[name]
        diff = new_value - old_value
        abs_diff = abs(diff)
        if abs_diff <= 1e-9:
            continue
        abs_threshold = DYNAMIC_ABS_CHANGE.get(name, 1.0)
        rel_threshold = DYNAMIC_PRICE_REL_CHANGE if "price" in name.lower() else DYNAMIC_REL_CHANGE
        base = max(abs(old_value), 1.0)
        rel_change = abs_diff / base
        if abs_diff >= abs_threshold or rel_change >= rel_threshold:
            direction = "increased" if diff > 0 else "decreased"
            reasons.append(
                f"{name} {direction} by {format_number(abs_diff)} "
                f"(from {format_number(old_value)} to {format_number(new_value)})"
            )
    return reasons


def block_status(session):
    collector = session["collector"]
    day = int(collector.env.day)
    manual_days = int(session.get("human_days_in_block", 0))
    remaining = max(0, BLOCK_MIN_HUMAN_DAYS - manual_days)
    can_skip = remaining == 0 and day < collector.env.lim_day - 1
    last_skip_reason = session.get("last_skip_reason")
    return {
        "day": day,
        "block_title": "Continuous Human Decision Stage",
        "block_start": session.get("block_start_day", day),
        "block_reason": "First collect a short continuous segment of human decisions, then let the system advance automatically until business conditions change significantly.",
        "manual_days_in_block": manual_days,
        "min_human_days": BLOCK_MIN_HUMAN_DAYS,
        "remaining_before_skip": remaining,
        "can_skip": can_skip,
        "next_block_start": None,
        "next_block_title": "Significant State-Change Point",
        "next_block_reason": last_skip_reason or "The system then advances to the next significant change in cash, debt, inventory, purchase references, or price and returns control to you.",
        "dynamic_skip_max_days": DYNAMIC_SKIP_MAX_DAYS,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Company A Human Expert Data Collection</title>
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    main { max-width: 1420px; margin: 0 auto; padding: 22px; }
    .panel { background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
    .intro { text-align: center; padding: 60px 28px; }
    .intro h1 { margin: 0 0 18px; font-size: 30px; }
    .intro p { max-width: 760px; margin: 12px auto; line-height: 1.8; font-size: 16px; }
    .flow { max-width: 900px; margin: 28px auto 0; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; align-items: stretch; }
    .flow-step { border: 1px solid #cbd5df; background: #f8fbff; border-radius: 8px; padding: 14px 12px; text-align: center; line-height: 1.55; }
    .flow-step strong { display: block; color: #18324a; margin-bottom: 4px; }
    .role-box { max-width: 900px; margin: 20px auto 0; border: 1px solid #cfe0f5; background: #f3f8ff; border-radius: 8px; padding: 14px; line-height: 1.7; text-align: left; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    input { height: 34px; border: 1px solid #cbd5df; border-radius: 6px; padding: 0 10px; font-size: 15px; }
    button { height: 36px; border: 0; border-radius: 6px; padding: 0 14px; background: #1f6feb; color: white; cursor: pointer; font-size: 14px; }
    button.secondary { background: #596b7d; }
    button.danger { background: #b42318; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { border-bottom: 1px solid #e5eaf0; padding: 10px 8px; text-align: center; }
    th { background: #eef3f8; font-weight: 600; }
    tr.risk_high td, tr.risk_medium td, tr.change_bad td { background: #e2f6e8; }
    tr.change_good td { background: #ffe6e6; }
    td { white-space: pre-line; line-height: 1.55; }
    .work-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(380px, 420px); gap: 16px; align-items: stretch; }
    .info-column { min-width: 0; }
    .info-column .panel:last-child, .decision-panel { margin-bottom: 0; }
    .decision-panel { display: flex; flex-direction: column; align-self: stretch; }
    .decision-title { margin-bottom: 12px; }
    .decision-title h2 { margin: 0; font-size: 18px; color: #18324a; }
    .decision-title p { margin: 6px 0 0; color: #596b7d; font-size: 13px; line-height: 1.55; }
    .actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .action-card { border: 1px solid #dde3ea; border-radius: 8px; padding: 12px; }
    .action-card label { display: block; font-weight: 600; margin-bottom: 8px; }
    .action-card input { width: calc(100% - 22px); text-align: center; }
    .hint { color: #546579; font-size: 13px; line-height: 1.5; margin-top: 8px; min-height: 38px; }
    .adjust { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
    .adjust button { height: 28px; padding: 0 8px; background: #eef3f8; color: #18324a; border: 1px solid #ccd6e0; }
    .decision-submit { margin-top: 14px; display: grid; gap: 8px; }
    .decision-submit #submitBtn { width: 100%; height: 42px; font-size: 15px; }
    .decision-secondary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .decision-secondary button { width: 100%; padding: 0 8px; }
    .status { color: #546579; margin-left: 10px; }
    .block-info { margin-top: 14px; padding: 12px 14px; background: #f3f8ff; border: 1px solid #cfe0f5; border-radius: 8px; line-height: 1.65; }
    .block-info strong { color: #18324a; }
    .block-info .muted { color: #596b7d; }
    .summary-card { border-left: 4px solid #18324a; background: #f8fbff; padding: 14px 16px; border-radius: 6px; line-height: 1.75; }
    .summary-card h3, .chart-card h3, .module-card h3 { margin: 0 0 10px; color: #18324a; font-size: 17px; }
    .summary-card p { margin: 6px 0; }
    .summary-warning { color: #b42318; font-weight: 600; }
    .line-chart-grid { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }
    .line-chart { border: 1px solid #d9e1ea; border-radius: 8px; background: #fff; padding: 10px; }
    .line-chart-title { font-weight: 700; color: #344054; margin-bottom: 6px; font-size: 14px; }
    .line-chart svg { width: 100%; height: 178px; display: block; }
    .axis-label { fill: #667085; font-size: 10px; }
    .line-path { fill: none; stroke: #4b7bec; stroke-width: 2.5; }
    .line-dot { fill: #4b7bec; pointer-events: none; }
    .line-dot-hit { fill: transparent; cursor: pointer; }
    .line-tooltip {
      position: fixed;
      z-index: 1000;
      display: none;
      pointer-events: none;
      background: rgba(24, 50, 74, 0.94);
      color: #fff;
      border-radius: 6px;
      padding: 7px 9px;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18);
      white-space: nowrap;
    }
    .line-tooltip.show { display: block; }
    .module-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }
    .module-card { border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; background: #ffffff; }
    .vertical-bars { display: flex; gap: 6px; align-items: end; min-height: 250px; padding-top: 8px; overflow: hidden; }
    .vertical-bar-item { display: grid; grid-template-rows: 24px 110px 104px; gap: 6px; text-align: center; flex: 1 1 0; min-width: 0; }
    .vertical-value { font-size: 12px; font-weight: 700; color: #1f2933; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .vertical-track { height: 110px; display: flex; align-items: end; justify-content: center; background: #f3f6fa; border-radius: 6px; overflow: hidden; }
    .vertical-fill { width: 58%; background: #4b7bec; border-radius: 6px 6px 0 0; min-height: 2px; }
    .vertical-label { font-size: 11px; color: #536273; line-height: 1.2; word-break: keep-all; overflow-wrap: anywhere; }
    .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 16px; }
    .chart-card { border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; background: #fff; }
    .bar-row { display: grid; grid-template-columns: 128px 1fr 72px; gap: 10px; align-items: center; margin: 9px 0; }
    .bar-label { color: #344054; font-size: 13px; }
    .bar-track { height: 18px; background: #eef3f8; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; background: #4b7bec; border-radius: 4px; min-width: 2px; }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; color: #1f2933; font-size: 13px; }
    details { border: 1px solid #dde3ea; border-radius: 8px; padding: 12px 14px; background: #fff; margin-top: 10px; }
    summary { cursor: pointer; color: #18324a; font-weight: 700; }
    details ol { margin: 10px 0 0 22px; padding: 0; line-height: 1.75; }
    .detail-section { margin-top: 14px; }
    .detail-section h4 { margin: 0 0 6px; color: #344054; }
    .production-flow-detail { margin-top: 10px; border: 1px solid #d9e1ea; border-radius: 8px; background: #f8fbff; padding: 12px; }
    .flow-market-row, .flow-lanes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .flow-market-row { margin-bottom: 12px; }
    .flow-lane { display: grid; grid-template-columns: 1fr; gap: 8px; border: 1px solid #d9e1ea; border-radius: 8px; background: #ffffff; padding: 12px; }
    .flow-lane-title { text-align: center; color: #18324a; font-weight: 700; }
    .flow-node { border: 1px solid #d9e1ea; border-radius: 8px; background: #fff; padding: 10px 12px; min-height: 86px; display: grid; align-content: center; gap: 4px; text-align: center; }
    .flow-node strong { color: #18324a; font-size: 15px; }
    .flow-node span { color: #344054; line-height: 1.45; }
    .flow-node small { color: #596b7d; line-height: 1.45; }
    .flow-market { background: #eaf4ff; border-color: #b9d9ff; }
    .flow-third-market { background: #fff0ee; border-color: #ffc9c2; }
    .flow-company-a { background: #fff7e6; border-color: #f3c66d; }
    .flow-company-b { background: #ecf9f0; border-color: #b9dfbf; }
    .flow-product-a { background: #eef6ff; border-color: #b9d9ff; }
    .flow-product-b { background: #effaf2; border-color: #b9dfbf; }
    .flow-arrow { position: relative; display: grid; place-items: center; min-height: 36px; color: #42556c; text-align: center; font-size: 12px; line-height: 1.35; z-index: 1; }
    .flow-arrow::after { content: ""; position: absolute; top: 4px; bottom: 4px; left: 50%; width: 2px; background: #8aa1b5; transform: translateX(-50%); z-index: -1; }
    .flow-arrow::before { content: ""; position: absolute; bottom: 4px; left: 50%; width: 9px; height: 9px; border-top: 2px solid #8aa1b5; border-right: 2px solid #8aa1b5; transform: translateX(-50%) rotate(135deg); z-index: -1; }
    .flow-arrow span { display: inline-block; background: #ffffff; padding: 2px 5px; border-radius: 4px; }
    .intake-grid { max-width: 980px; margin: 24px auto 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; text-align: left; }
    .intake-grid label { display: grid; gap: 6px; color: #344054; font-size: 14px; }
    .intake-grid input, .intake-grid select { width: 100%; box-sizing: border-box; height: 38px; border: 1px solid #cfd8e3; border-radius: 6px; padding: 6px 10px; background: #fff; font: inherit; }
    .start-area { margin-top: 18px; display: flex; justify-content: center; }
    .start-area button { min-width: 160px; height: 40px; }
    .consent-box { max-width: 980px; margin: 16px auto 0; padding: 14px 16px; border: 1px solid #cfd8e3; border-radius: 6px; background: #f7faff; color: #344054; text-align: left; font-size: 14px; line-height: 1.65; }
    .consent-box p { width: 100%; max-width: none; box-sizing: border-box; margin: 0 0 8px; overflow-wrap: break-word; font-size: 14px; }
    .consent-box .collection-period { color: #12263f; }
    .consent-check { display: flex; align-items: flex-start; gap: 9px; font-weight: 600; color: #12263f; cursor: pointer; }
    .consent-check input { width: 17px; height: 17px; margin-top: 3px; flex: 0 0 auto; }
    .hidden { display: none; }
    .busy { position: fixed; inset: 0; background: rgba(255,255,255,.72); display: none; align-items: center; justify-content: center; z-index: 20; }
    .busy.show { display: flex; }
    .spinner { width: 42px; height: 42px; border: 5px solid #c9d7e6; border-top-color: #1f6feb; border-radius: 50%; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 1100px) { .work-layout { grid-template-columns: 1fr; } }
    @media (max-width: 920px) { .flow, .module-grid, .chart-grid, .line-chart-grid, .intake-grid, .flow-market-row, .flow-lanes { grid-template-columns: 1fr; } }
    @media (max-width: 620px) { .actions, .decision-secondary { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section id="intro" class="panel intro">
      <h1>Task Description</h1>
      <p>This system includes Company A, Company B, a bank, and a third-party market.<br>You are the manager of Company A. Use the daily business information to make decisions, keep the company stable, and help it survive longer.</p>
      <div class="flow" aria-label="Daily process">
        <div class="flow-step"><strong>1. Review Today's Status</strong>You will see cash, debt, inventory, prices, and the previous day's business results.</div>
        <div class="flow-step"><strong>2. Make Business Decisions</strong>Enter the loan request, purchase demands for Raw Materials A and B, and the Product A sale price.</div>
        <div class="flow-step"><strong>3. Let the System Run and Advance</strong>After submission, the system completes trading, production, and settlement automatically. The episode ends if cash is insufficient to repay debt.</div>
      </div>
      <div class="intake-grid">
        <label>Participant ID (optional; do not enter your name)
          <input id="participant" value="anonymous" />
        </label>
        <label>Age Group
          <select id="ageGroup">
            <option value="">Please select</option>
            <option value="20岁以下">Under 20</option>
            <option value="20-40岁">20-40</option>
            <option value="40岁以上">Over 40</option>
          </select>
        </label>
        <label>Education Level
          <select id="education">
            <option value="">Please select</option>
            <option value="高中及以下">High school or below</option>
            <option value="大专/本科">Associate or bachelor's degree</option>
            <option value="硕士及以上">Master's degree or above</option>
          </select>
        </label>
        <label>Gender
          <select id="gender">
            <option value="">Please select</option>
            <option value="男">Male</option>
            <option value="女">Female</option>
          </select>
        </label>
        <label>Economics/Management Background
          <select id="econBackground">
            <option value="">Please select</option>
            <option value="几乎没有">Little or none</option>
            <option value="学过一点">Some study</option>
            <option value="比较熟悉">Fairly familiar</option>
          </select>
        </label>
        <label>Business/Strategy Game Experience
          <select id="strategyExperience">
            <option value="">Please select</option>
            <option value="几乎没有">Little or none</option>
            <option value="偶尔接触">Occasional</option>
            <option value="经常接触">Frequent</option>
          </select>
        </label>
        <label>Access Password (enter 123)
          <input id="accessPassword" type="password" />
        </label>
      </div>
      <div class="consent-box" aria-labelledby="consentTitle">
        <p id="consentTitle"><strong>Participant Information and Informed Consent</strong></p>
        <p class="collection-period"><strong>Collection Period: May 21, 00:00 to June 21, 00:00</strong></p>
        <p>This experiment studies business decisions in a simulated economic environment. The system will record basic information, daily decisions, decision time, and simulation outcomes for academic research, model training, and statistical analysis. Use an anonymous ID and do not enter your name, contact details, or other directly identifying information. Participation is entirely voluntary, and you may select “End Collection” at any time to withdraw.</p>
        <label class="consent-check">
          <input id="consent" type="checkbox" />
          <span>I have read the information above, voluntarily agree to participate, and consent to the recording and use of my data within the stated scope.</span>
        </label>
      </div>
      <div class="start-area">
        <button id="startBtn" disabled>Start Collection</button>
      </div>
    </section>

    <section id="app" class="hidden">
      <div class="work-layout">
        <div class="info-column">
          <div class="panel">
            <div class="row">
              <strong id="dayTitle">Day -</strong>
              <span id="status" class="status"></span>
            </div>
          </div>

          <div class="panel">
            <div id="infoModules" class="module-grid"></div>
          </div>

          <div class="panel">
            <div id="charts" class="chart-grid"></div>
            <div id="explainDetails"></div>
          </div>

          <div class="panel">
            <div id="summaryNews" class="summary-card"></div>
          </div>
        </div>

        <aside class="panel decision-panel">
          <div class="decision-title">
            <h2>Today's Decisions</h2>
            <p>Enter A and B first, then review the loan and price.</p>
          </div>
          <div class="actions" id="actions"></div>
          <div class="decision-submit">
            <button id="submitBtn">Submit Actions and Continue to the Next Day</button>
            <div class="decision-secondary">
              <button id="prevBtn" class="secondary" disabled>View Previous Day</button>
              <button id="skipBtn" class="secondary" disabled>Finish This Segment and Skip Ahead</button>
              <button id="nextEpisodeBtn" class="secondary" disabled>Start Next Episode</button>
              <button id="endBtn" class="danger">End Collection</button>
            </div>
          </div>
          <div id="blockInfo" class="block-info"></div>
        </aside>
      </div>
    </section>
  </main>
  <div id="busy" class="busy"><div class="spinner"></div></div>

<script>
const actionNames = ["Loan Request Amount", "Raw Material A Purchase Demand", "Raw Material B Purchase Demand", "Product A Sale Price"];
const participantFields = [
  ["ageGroup", "age_group", "an age group"],
  ["education", "education", "an education level"],
  ["gender", "gender", "a gender"],
  ["econBackground", "econ_background", "an economics/management background level"],
  ["strategyExperience", "strategy_experience", "a business/strategy game experience level"],
];
let sessionId = null;
let current = null;
let previousSnapshot = null;
let viewingPrevious = false;
let draftValues = null;
let block = null;

function $(id) { return document.getElementById(id); }
function busy(show) { $("busy").classList.toggle("show", show); }
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatChartNumber(value) {
  const number = Number(value) || 0;
  if (Math.abs(number) >= 10 || Number.isInteger(number)) return String(Math.round(number));
  return number.toFixed(2).replace(/\.?0+$/, "");
}

function formatTooltipNumber(value) {
  return (Number(value) || 0).toFixed(2).replace(/\.?0+$/, "");
}

function niceChartStep(rawStep, maxValue) {
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 1;
  const exponent = Math.floor(Math.log10(rawStep));
  const scale = Math.pow(10, exponent);
  const base = rawStep / scale;
  let niceBase = 10;
  if (base <= 1) niceBase = 1;
  else if (base <= 2) niceBase = 2;
  else if (base <= 5) niceBase = 5;
  let step = niceBase * scale;
  if (maxValue >= 10 && step < 10) step = 10;
  return step;
}

function axisFromZero(maxValue) {
  const rawMax = Math.max(1, Number(maxValue) || 0);
  const step = niceChartStep(rawMax / 5, rawMax);
  const maxY = Math.max(step, Math.ceil(rawMax / step) * step);
  const ticks = [];
  for (let value = 0; value <= maxY + step * 0.001; value += step) {
    ticks.push(Number(value.toFixed(8)));
  }
  return {maxY, ticks};
}

function lineTooltipEl() {
  let tooltip = $("lineTooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "lineTooltip";
    tooltip.className = "line-tooltip";
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function showLineTooltip(event, day, value) {
  const tooltip = lineTooltipEl();
  tooltip.innerHTML = `Day: ${escapeHtml(day)}<br>Net Profit: ${escapeHtml(formatTooltipNumber(value))}`;
  const offset = 12;
  tooltip.style.left = `${event.clientX + offset}px`;
  tooltip.style.top = `${event.clientY + offset}px`;
  tooltip.classList.add("show");
}

function hideLineTooltip() {
  const tooltip = $("lineTooltip");
  if (tooltip) tooltip.classList.remove("show");
}

async function api(path, payload) {
  busy(true);
  try {
    const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload || {})});
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Request failed.");
    return data;
  } finally {
    busy(false);
  }
}

function renderState(data, readonly=false) {
  $("dayTitle").textContent = `Company A Information Available on Day ${data.day}${readonly ? " (Read-Only View)" : ""}`;
  renderDashboard(data.dashboard || {});
}

function renderDashboard(dashboard) {
  const summary = dashboard.summary || {title: "Business Summary", lines: []};
  const summaryLines = summary.lines || [];
  $("summaryNews").innerHTML = `
    <h3>${escapeHtml(summary.title)}</h3>
    ${summaryLines[0] ? `<p>${escapeHtml(summaryLines[0])}</p>` : ""}
    ${(summary.warnings || []).map(line => `<p class="summary-warning">${escapeHtml(line)}</p>`).join("")}
    ${summaryLines.slice(1).map(line => `<p>${escapeHtml(line)}</p>`).join("")}
    ${renderLineCharts(dashboard.lineCharts || [])}
  `;

  $("infoModules").innerHTML = (dashboard.modules || []).map(module => `
    <section class="module-card">
      <h3>${escapeHtml(module.title)}</h3>
      ${renderVerticalBars(module.items || [])}
    </section>
  `).join("");

  $("charts").innerHTML = (dashboard.charts || []).map(chart => renderChart(chart)).join("");
  $("charts").classList.toggle("hidden", !(dashboard.charts || []).length);
  $("explainDetails").innerHTML = renderCombinedDetails(dashboard.details || []);
}

function renderLineCharts(charts) {
  if (!charts.length) return "";
  return `<div class="line-chart-grid">${charts.map(chart => renderLineChart(chart)).join("")}</div>`;
}

function renderLineChart(chart) {
  const points = chart.points || [];
  const width = 320;
  const height = 178;
  const left = 48;
  const right = 30;
  const top = 28;
  const bottom = 42;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const values = points.map(point => Number(point.value) || 0);
  const days = points.map(point => Number(point.day) || 1);
  let minY = 0;
  let {maxY, ticks: yTicks} = axisFromZero(Math.max(0, ...values));
  const ySpan = Math.max(1, maxY - minY);
  const minX = 0;
  const maxX = Math.max(1, ...days);
  const xSpan = Math.max(1, maxX - minX);
  const xTicks = [0, ...Array.from(new Set(days)).filter(day => day > 0).sort((a, b) => a - b)];
  const xOf = day => left + (day - minX) / xSpan * plotW;
  const yOf = value => top + (maxY - value) / ySpan * plotH;
  const coords = points.map(point => {
    const day = Number(point.day) || minX;
    const value = Number(point.value) || 0;
    const x = xOf(day);
    const y = yOf(value);
    return {x, y, value, day};
  });
  const path = coords.map(point => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  const zeroY = yOf(0);
  return `
    <section class="line-chart">
      <div class="line-chart-title">${escapeHtml(chart.title)}</div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(chart.title)}">
        <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#cfd8e3" />
        <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#cfd8e3" />
        ${yTicks.map(value => {
          const y = yOf(value);
          return `
            <line x1="${left}" y1="${y.toFixed(1)}" x2="${width - right}" y2="${y.toFixed(1)}" stroke="#eef2f6" />
            <text x="${left - 6}" y="${(y + 3).toFixed(1)}" class="axis-label" text-anchor="end">${escapeHtml(formatChartNumber(value))}</text>
          `;
        }).join("")}
        <line x1="${left}" y1="${zeroY.toFixed(1)}" x2="${width - right}" y2="${zeroY.toFixed(1)}" stroke="#98a2b3" stroke-width="1.4" />
        ${xTicks.map(day => {
          const x = xOf(day);
          return `
            <line x1="${x.toFixed(1)}" y1="${height - bottom}" x2="${x.toFixed(1)}" y2="${height - bottom + 4}" stroke="#98a2b3" />
            <text x="${x.toFixed(1)}" y="${height - 18}" class="axis-label" text-anchor="middle">${escapeHtml(day)}</text>
          `;
        }).join("")}
        <text x="${left - 6}" y="12" class="axis-label" text-anchor="end">Net Profit</text>
        <text x="${width - right + 6}" y="${height - bottom + 4}" class="axis-label">Day</text>
        ${coords.length > 1 ? `<polyline class="line-path" points="${path}" />` : ""}
        ${coords.map(point => `
          <g onmousemove="showLineTooltip(event, ${point.day}, ${point.value})" onmouseleave="hideLineTooltip()">
            <circle class="line-dot-hit" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="9"></circle>
            <circle class="line-dot" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="3"></circle>
          </g>
        `).join("")}
      </svg>
    </section>
  `;
}

function renderVerticalBars(items) {
  const max = Math.max(1, ...items.map(item => Math.abs(Number(item.raw ?? item.value) || 0)));
  return `
    <div class="vertical-bars">
      ${items.map(item => {
        const value = Number(item.raw ?? item.value) || 0;
        const height = Math.max(2, Math.round(Math.abs(value) / max * 100));
        return `
          <div class="vertical-bar-item">
            <div class="vertical-value">${escapeHtml(item.value)}</div>
            <div class="vertical-track"><div class="vertical-fill" style="height:${height}%"></div></div>
            <div class="vertical-label">${escapeHtml(item.label)}</div>
          </div>`;
      }).join("")}
    </div>
  `;
}

function renderCombinedDetails(details) {
  if (!details.length) return "";
  return `
    <details>
      <summary>More Rules and Transaction Details</summary>
      ${details.map(detail => `
        <section class="detail-section">
          <h4>${escapeHtml(detail.title)}</h4>
          ${detail.html ? detail.html : `
            <ol>
              ${(detail.lines || []).map(line => `<li>${escapeHtml(line)}</li>`).join("")}
            </ol>
          `}
        </section>
      `).join("")}
    </details>
  `;
}

function renderChart(chart) {
  const bars = chart.bars || [];
  const max = Math.max(1, ...bars.map(bar => Math.abs(Number(bar.value) || 0)));
  return `
    <section class="chart-card">
      <h3>${escapeHtml(chart.title)}</h3>
      ${bars.map(bar => {
        const value = Number(bar.value) || 0;
        const width = Math.max(2, Math.round(Math.abs(value) / max * 100));
        return `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(bar.label)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="bar-value">${escapeHtml(value.toFixed(2).replace(/\.?0+$/, ""))}</div>
          </div>`;
      }).join("")}
    </section>
  `;
}

function renderBlockInfo(status) {
  block = status;
  if (!status) {
    $("blockInfo").innerHTML = "";
    $("skipBtn").disabled = true;
    $("skipBtn").textContent = "Finish This Segment and Skip Ahead";
    return;
  }
  const skipText = status.can_skip
    ? "Available now"
    : `${status.remaining_before_skip} more human decision days required`;
  $("blockInfo").innerHTML = `
    <div><strong>Current Progress:</strong> Day ${status.day}, ${status.block_title} (${status.manual_days_in_block}/${status.min_human_days} days complete).</div>
    <div class="muted"><strong>Skip:</strong> ${skipText}. ${status.next_block_reason}</div>
  `;
  $("skipBtn").disabled = !status.can_skip;
  $("skipBtn").textContent = status.can_skip
    ? "Skip to the Next Significant State Change"
    : `${status.remaining_before_skip} More Days Before Skipping`;
}

function renderActions(defaults, hints, limits, disabled=false) {
  const box = $("actions");
  box.innerHTML = "";
  const actionLayoutOrder = [1, 2, 0, 3];
  actionLayoutOrder.forEach(i => {
    const name = actionNames[i];
    const limit = limits && limits[i] ? limits[i] : {min: 0, max: ""};
    const maxAttr = Number.isFinite(Number(limit.max)) ? `max="${limit.max}"` : "";
    const card = document.createElement("div");
    card.className = "action-card";
    card.innerHTML = `
      <label>${name}</label>
      <input id="action${i}" type="number" step="0.01" min="${limit.min}" ${maxAttr} value="${defaults[i] || 0}" onblur="clampInput(${i})" ${disabled ? "disabled" : ""}/>
      <div class="hint">${hints[i] || ""}</div>
      <div class="adjust">
        <button type="button" onclick="adjust(${i},0.9)" ${disabled ? "disabled" : ""}>Decrease 10%</button>
        <button type="button" onclick="adjust(${i},1.1)" ${disabled ? "disabled" : ""}>Increase 10%</button>
        <button type="button" onclick="add(${i},-1)" ${disabled ? "disabled" : ""}>-1</button>
        <button type="button" onclick="add(${i},1)" ${disabled ? "disabled" : ""}>+1</button>
        <button type="button" onclick="setPreset(${i}, ${Number(limit.min || 0)})" ${disabled ? "disabled" : ""}>Minimum</button>
        <button type="button" onclick="setPreset(${i}, ${Number(defaults[i] || 0)})" ${disabled ? "disabled" : ""}>Default</button>
        <button type="button" onclick="setPreset(${i}, ${Number(limit.max || 0)})" ${disabled ? "disabled" : ""}>Maximum</button>
      </div>`;
    box.appendChild(card);
  });
}

function values() {
  return [0,1,2,3].map(i => {
    const el = $("action" + i);
    const value = Number(el.value);
    const min = Number(el.min || 0);
    const max = Number(el.max);
    if (!Number.isFinite(value)) throw new Error(`Please enter ${actionNames[i]}.`);
    if (value < min || (Number.isFinite(max) && value > max)) {
      const maxText = Number.isFinite(max) ? max : "no upper limit";
      throw new Error(`${actionNames[i]} must be between ${min} and ${maxText}.`);
    }
    return value;
  });
}
function formatInputValue(value) { return Number(value).toFixed(4).replace(/\.?0+$/, ""); }
function bounded(i, value) {
  const el = $("action" + i);
  const min = Number(el.min || 0);
  const max = Number(el.max);
  let next = Math.max(min, Number(value || 0));
  if (Number.isFinite(max)) next = Math.min(max, next);
  return next;
}
function clampInput(i) { const el = $("action" + i); el.value = formatInputValue(bounded(i, Number(el.value || 0))); }
function adjust(i, factor) { const el = $("action" + i); el.value = formatInputValue(bounded(i, Number(el.value || 0) * factor)); }
function add(i, delta) { const el = $("action" + i); el.value = formatInputValue(bounded(i, Number(el.value || 0) + delta)); }
function setPreset(i, value) { const el = $("action" + i); el.value = formatInputValue(bounded(i, value)); }

function participantInfo() {
  const info = {};
  for (const [elementId, key, label] of participantFields) {
    const value = $(elementId).value;
    if (!value) throw new Error(`Please select ${label}.`);
    info[key] = value;
  }
  return info;
}

async function start() {
  if (!$("consent").checked) throw new Error("Please read the participant information and confirm your informed consent first.");
  const data = await api("/api/start", {
    participant_id: $("participant").value,
    password: $("accessPassword").value,
    participant_info: participantInfo(),
    consent: true,
  });
  sessionId = data.session_id;
  current = data.state;
  block = data.block;
  previousSnapshot = null;
  $("intro").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("status").textContent = "The environment is ready. Please enter today's decisions.";
  renderState(current);
  renderActions(current.defaults, current.hints, current.limits);
  renderBlockInfo(data.block);
}

async function submitStep() {
  const data = await api("/api/step", {session_id: sessionId, values: values()});
  previousSnapshot = data.previous;
  current = data.state;
  block = data.block;
  $("prevBtn").disabled = !previousSnapshot;
  if (data.done) {
    $("submitBtn").disabled = true;
    $("skipBtn").disabled = true;
    $("nextEpisodeBtn").disabled = false;
    $("status").textContent = `This episode has ended after ${data.survival_days} survival days. ${data.rows_saved} expert records have been saved.`;
  } else {
    $("status").textContent = `Day ${current.day} has begun. ${data.rows_saved} expert records have been saved.`;
    renderState(current);
    renderActions(current.defaults, current.hints, current.limits);
    renderBlockInfo(data.block);
  }
}

async function skipToNextBlock() {
  const data = await api("/api/skip_to_next_block", {session_id: sessionId, values: values()});
  previousSnapshot = null;
  viewingPrevious = false;
  current = data.state;
  block = data.block;
  $("prevBtn").disabled = true;
  $("prevBtn").textContent = "View Previous Day";
  if (data.done) {
    $("submitBtn").disabled = true;
    $("skipBtn").disabled = true;
    $("nextEpisodeBtn").disabled = false;
    $("status").textContent = `The episode ended during automatic advancement after ${data.survival_days} survival days. ${data.rows_saved} expert records have been saved.`;
  } else {
    $("submitBtn").disabled = false;
    $("status").textContent = `The system advanced automatically by ${data.auto_days} days. It is now Day ${current.day}; data collection resumes at: ${data.block.block_title}.`;
  }
  renderState(current);
  renderActions(current.defaults, current.hints, current.limits, data.done);
  renderBlockInfo(data.block);
  if (data.done) $("skipBtn").disabled = true;
}

function togglePrevious() {
  if (!previousSnapshot) return;
  if (!viewingPrevious) {
    draftValues = values();
    viewingPrevious = true;
    $("prevBtn").textContent = "Return to Today";
    $("submitBtn").disabled = true;
    $("skipBtn").disabled = true;
    renderState(previousSnapshot, true);
    renderActions(previousSnapshot.human_values, previousSnapshot.hints, previousSnapshot.limits, true);
    $("status").textContent = "You are viewing the previous day's record. This view is read-only.";
  } else {
    viewingPrevious = false;
    $("prevBtn").textContent = "View Previous Day";
    $("submitBtn").disabled = false;
    renderBlockInfo(block);
    renderState(current);
    renderActions(draftValues || current.defaults, current.hints, current.limits);
    $("status").textContent = "You have returned to today. Please continue entering today's decisions.";
  }
}

async function nextEpisode() {
  const data = await api("/api/next_episode", {session_id: sessionId});
  current = data.state;
  block = data.block;
  previousSnapshot = null;
  viewingPrevious = false;
  $("prevBtn").disabled = true;
  $("prevBtn").textContent = "View Previous Day";
  $("submitBtn").disabled = false;
  $("skipBtn").disabled = true;
  $("nextEpisodeBtn").disabled = true;
  $("status").textContent = "A new episode has started. Please enter today's decisions.";
  renderState(current);
  renderActions(current.defaults, current.hints, current.limits);
  renderBlockInfo(data.block);
}

async function endSession() {
  await api("/api/end", {session_id: sessionId});
  $("submitBtn").disabled = true;
  $("skipBtn").disabled = true;
  $("nextEpisodeBtn").disabled = true;
  $("prevBtn").disabled = true;
  $("endBtn").disabled = true;
  $("status").textContent = "Data collection has ended. The data have been saved on the server.";
}

$("startBtn").onclick = () => start().catch(e => alert(e.message));
$("consent").onchange = () => { $("startBtn").disabled = !$("consent").checked; };
$("submitBtn").onclick = () => submitStep().catch(e => alert(e.message));
$("skipBtn").onclick = () => skipToNextBlock().catch(e => alert(e.message));
$("prevBtn").onclick = togglePrevious;
$("nextEpisodeBtn").onclick = () => nextEpisode().catch(e => alert(e.message));
$("endBtn").onclick = () => endSession().catch(e => alert(e.message));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path != "/":
            self.send_error(404)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            payload = self._read_json()
            if path == "/api/start":
                self._send_json(api_start(payload))
            elif path == "/api/step":
                self._send_json(api_step(payload))
            elif path == "/api/skip_to_next_block":
                self._send_json(api_skip_to_next_block(payload))
            elif path == "/api/next_episode":
                self._send_json(api_next_episode(payload))
            elif path == "/api/end":
                self._send_json(api_end(payload))
            else:
                self._send_json({"error": "Unknown endpoint."}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.address_string(), fmt % args))


def get_session(session_id):
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise ValueError("The experiment session does not exist or has ended. Please start again.")
    return session


def api_start(payload):
    if ACCESS_PASSWORD and payload.get("password", "") != ACCESS_PASSWORD:
        raise ValueError("The access password is incorrect.")
    if payload.get("consent") is not True:
        raise ValueError("Please read the participant information and confirm your informed consent first.")
    participant_id = payload.get("participant_id", "anonymous")
    participant_info = normalize_participant_info(payload)
    participant_info["consent"] = "yes"
    participant_info["consent_timestamp"] = datetime.now(timezone.utc).isoformat()
    auto_policy = payload.get("auto_policy", SERVER_AUTO_POLICY)
    if auto_policy not in {"td3", "fixed"}:
        raise ValueError("auto_policy must be either td3 or fixed.")
    session_id = uuid.uuid4().hex
    output_path, meta_output_path, summary_output_path = session_paths(participant_id, session_id)
    collector = HumanProductionCollector(
        seed=DEFAULT_SEED,
        output_path=output_path,
        meta_output_path=meta_output_path,
        auto_policy=auto_policy,
        participant_id=participant_id,
        participant_info=participant_info,
    )
    collector.start_episode()
    session = {
        "collector": collector,
        "lock": threading.Lock(),
        "started_at": time.perf_counter(),
        "decision_started_at": time.perf_counter(),
        "durations": [],
        "episode_durations": [],
        "participant_id": participant_id,
        "participant_info": participant_info,
        "auto_policy": auto_policy,
        "output_path": output_path,
        "meta_output_path": meta_output_path,
        "summary_output_path": summary_output_path,
        "episode_started_at": time.perf_counter(),
        "episode_summary_written": False,
        "block_start_day": collector.env.day,
        "human_days_in_block": 0,
        "last_skip_reason": None,
    }
    with SESSIONS_LOCK:
        SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "state": serialize_state(collector),
        "block": block_status(session),
        "output_path": output_path,
        "meta_output_path": meta_output_path,
        "summary_output_path": summary_output_path,
    }


def api_step(payload):
    session = get_session(payload.get("session_id"))
    with session["lock"]:
        collector = session["collector"]
        values = [float(value) for value in payload.get("values", [])]
        state_before = collector.current_state().copy()
        day_before = collector.env.day
        model_action = human_to_model_action(state_before, values)
        now = time.perf_counter()
        decision_seconds = now - session["decision_started_at"]
        session["durations"].append(decision_seconds)
        session["episode_durations"].append(decision_seconds)
        done, _ = collector.step(model_action, values, decision_seconds)
        session["human_days_in_block"] = int(session.get("human_days_in_block", 0)) + 1
        full_state_before = collector.history[-1].get("full_state") if collector.history else None
        previous = serialize_state(
            collector,
            state_before,
            day_before,
            readonly=True,
            full_state_override=full_state_before,
        )
        previous["human_values"] = [input_number(value) for value in values]
        session["decision_started_at"] = time.perf_counter()
        result = {
            "done": done,
            "previous": previous,
            "rows_saved": collector.rows_saved,
            "stats": session_stats(session),
            "block": block_status(session),
        }
        if done:
            result["survival_days"] = collector.env.day
            result["state"] = serialize_state(
                collector,
                state_before,
                day_before,
                readonly=True,
                full_state_override=full_state_before,
            )
            write_episode_summary(session, "episode_done", survival_days=collector.env.day)
        else:
            result["state"] = serialize_state(collector)
        return result


def api_skip_to_next_block(payload):
    session = get_session(payload.get("session_id"))
    with session["lock"]:
        collector = session["collector"]
        status = block_status(session)
        if not status["can_skip"]:
            raise ValueError(f"You must complete {status['remaining_before_skip']} more human decision days in this segment before skipping.")

        anchor_state = collector.current_state().copy()
        requested_values = [float(value) for value in payload.get("values", [])]
        skip_values = clamp_human_values_for_state(anchor_state, requested_values)
        auto_days = 0
        last_auto = None
        done = False
        trigger_reasons = []

        while auto_days < DYNAMIC_SKIP_MAX_DAYS:
            current_state = collector.current_state().copy()
            current_values = clamp_human_values_for_state(current_state, skip_values)
            model_action = human_to_model_action(current_state, current_values)
            last_auto = collector.advance_with_target_action(model_action)
            last_auto["human_values"] = [input_number(value) for value in current_values]
            auto_days += 1
            done = bool(last_auto["done"])
            if done:
                break
            trigger_reasons = metric_change_reasons(anchor_state, collector.current_state())
            if trigger_reasons:
                break

        if not done:
            session["block_start_day"] = collector.env.day
            session["human_days_in_block"] = 0
            session["decision_started_at"] = time.perf_counter()
            if trigger_reasons:
                session["last_skip_reason"] = "; ".join(trigger_reasons[:3]) + ". A new human decision is therefore required."
            else:
                session["last_skip_reason"] = f"The system advanced automatically by {auto_days} days. Even without a sharp change in any single measure, business decisions must be reviewed periodically."

        previous_state = last_auto["state"] if last_auto is not None else None
        previous_full_state = last_auto.get("full_state") if last_auto is not None else None
        result = {
            "done": done,
            "auto_days": auto_days,
            "trigger_reasons": trigger_reasons,
            "rows_saved": collector.rows_saved,
            "stats": session_stats(session),
            "block": block_status(session),
        }
        if done:
            result["survival_days"] = collector.env.day
            result["state"] = serialize_state(collector)
            write_episode_summary(session, "episode_done_during_auto_skip", survival_days=collector.env.day)
        else:
            result["state"] = serialize_state(
                collector,
                previous_state_override=previous_state,
                previous_full_state_override=previous_full_state,
            )
        return result


def api_next_episode(payload):
    session = get_session(payload.get("session_id"))
    with session["lock"]:
        collector = session["collector"]
        if session.get("episode_durations") and not session.get("episode_summary_written"):
            write_episode_summary(session, "next_episode_before_done", survival_days=collector.env.day)
        collector.start_episode()
        session["decision_started_at"] = time.perf_counter()
        session["episode_started_at"] = time.perf_counter()
        session["episode_durations"] = []
        session["episode_summary_written"] = False
        session["block_start_day"] = collector.env.day
        session["human_days_in_block"] = 0
        session["last_skip_reason"] = None
        return {"state": serialize_state(collector), "block": block_status(session)}


def api_end(payload):
    session_id = payload.get("session_id")
    session = get_session(session_id)
    with session["lock"]:
        if session.get("episode_durations") and not session.get("episode_summary_written"):
            write_episode_summary(session, "ended_by_user", survival_days=session["collector"].env.day)
        stats = session_stats(session)
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)
    return {"ok": True, "stats": stats}


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Web experiment system started: http://127.0.0.1:{PORT}")
    print(f"For LAN access, use this computer's IP address, for example: http://<computer-IP>:{PORT}")
    print(f"Non-human agent policy: {SERVER_AUTO_POLICY}")
    print("Access password: " + ("enabled" if ACCESS_PASSWORD else "disabled"))
    print(f"Data directory: {WEB_DATA_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
