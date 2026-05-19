import json
import os
import re
import sys
import threading
import time
import uuid
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
DECISION_BLOCKS = [
    {
        "start": 1,
        "title": "开局连续决策",
        "reason": "开局阶段决定贷款、采购和定价的初始方向，会影响企业后续能否稳定运转。",
    },
    {
        "start": 15,
        "title": "早中期经营片段",
        "reason": "企业已经经历了一段自动经营，现金、债务和库存可能发生变化，需要重新判断采购和价格。",
    },
    {
        "start": 40,
        "title": "中期经营片段",
        "reason": "中期阶段的还款压力、产品库存和市场价格更复杂，需要观察人在压力较高时如何调整经营。",
    },
    {
        "start": 70,
        "title": "后期经营片段",
        "reason": "如果企业能存活到后期，这里用于观察长期经营后的保守或激进决策。",
    },
]
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


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
    )


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
        f"当前现金：{format_number(cash)}；申请贷款金额可填0到{format_number(cash)}",
        f"原料K与原料L配套生产产品K，少的一种会卡住产量；当前K计划 {format_number(k_base)}，可填 {format_number(k_base * 0.5 if k_base else 0)} 到 {format_number(k_base * 1.5 if k_base else 10)}",
        f"原料L与原料K配套生产产品K，少的一种会卡住产量；当前L计划 {format_number(l_base)}，可填 {format_number(l_base * 0.5 if l_base else 0)} 到 {format_number(l_base * 1.5 if l_base else 10)}",
        f"这是产品K的出售价格；当前预设价格 {format_number(price_base)}，可填 {format_number(price_base * 0.5)} 到 {format_number(price_base * 1.5)}",
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
        raise ValueError(f"{name}不能为负数。")
    if base <= 0:
        action = value / 10 - 0.5
        if not -0.5 <= action <= 0.5:
            raise ValueError(f"{name}当前为0，请填写0到10之间的数。")
        return action
    action = value / base - 1
    if not -0.5 <= action <= 0.5:
        raise ValueError(f"{name}只能在 {format_number(base * 0.5)} 到 {format_number(base * 1.5)} 之间。")
    return action


def human_to_model_action(state, values):
    if len(values) != 4:
        raise ValueError("请填写四个动作数值。")
    if any(value < 0 for value in values):
        raise ValueError("请不要输入负数。")
    cash = raw_state_value(state, 0)
    k_base = raw_state_value(state, 11)
    l_base = raw_state_value(state, 12)
    price_base = raw_state_value(state, 6)
    loan, k_need, l_need, price = values

    if cash <= 0:
        if loan > 0:
            raise ValueError("当前现金为0，申请贷款金额只能填写0。")
        loan_action = -0.5
    else:
        if loan > cash:
            raise ValueError(f"申请贷款金额不能超过当前现金 {format_number(cash)}。")
        loan_action = loan / cash - 0.5

    k_action = quantity_to_action(k_need, k_base, "K采购需求")
    l_action = quantity_to_action(l_need, l_base, "L采购需求")
    if price_base <= 0:
        raise ValueError("当前价格基准异常，不能提交价格动作。")
    price_action = price / price_base - 1
    if not -0.5 <= price_action <= 0.5:
        raise ValueError(f"销售价格只能在 {format_number(price_base * 0.5)} 到 {format_number(price_base * 1.5)} 之间。")
    return [loan_action, k_action, l_action, price_action]


def serialize_state(collector, state=None, day=None, readonly=False, previous_state_override=None):
    if state is None:
        state = collector.current_state()
    if day is None:
        day = collector.env.day
    previous_state = None
    if previous_state_override is not None:
        previous_state = previous_state_override
    elif not readonly and collector.history:
        previous_state = collector.history[-1]["state"]
    rows = []
    for group, name, value, key in display_rows(state, day=day):
        tags = list(risk_tag(key, value, state) or row_change_tag(key, value, previous_state))
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
        return "无变化"
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


def current_block(session):
    index = session.get("block_index", 0)
    index = max(0, min(index, len(DECISION_BLOCKS) - 1))
    return DECISION_BLOCKS[index]


def next_block_after(day):
    for index, block in enumerate(DECISION_BLOCKS):
        if block["start"] > day:
            return index, block
    return None, None


def block_status(session):
    collector = session["collector"]
    day = int(collector.env.day)
    manual_days = int(session.get("human_days_in_block", 0))
    remaining = max(0, BLOCK_MIN_HUMAN_DAYS - manual_days)
    next_index, next_block = next_block_after(day)
    can_skip = remaining == 0 and next_block is not None
    block = current_block(session)
    return {
        "day": day,
        "block_index": session.get("block_index", 0),
        "block_title": block["title"],
        "block_start": session.get("block_start_day", block["start"]),
        "block_reason": block["reason"],
        "manual_days_in_block": manual_days,
        "min_human_days": BLOCK_MIN_HUMAN_DAYS,
        "remaining_before_skip": remaining,
        "can_skip": can_skip,
        "next_block_start": next_block["start"] if next_block else None,
        "next_block_title": next_block["title"] if next_block else None,
        "next_block_reason": next_block["reason"] if next_block else "当前已经没有预设的下一段，可以继续手动决策或结束采集。",
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>生产企业人类专家数据采集</title>
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    header { background: #18324a; color: white; padding: 18px 28px; }
    main { max-width: 1180px; margin: 0 auto; padding: 22px; }
    .panel { background: white; border: 1px solid #dde3ea; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
    .intro { text-align: center; padding: 60px 28px; }
    .intro h1 { margin: 0 0 18px; font-size: 30px; }
    .intro p { max-width: 760px; margin: 12px auto; line-height: 1.8; font-size: 16px; }
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
    .actions { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; }
    .action-card { border: 1px solid #dde3ea; border-radius: 8px; padding: 12px; }
    .action-card label { display: block; font-weight: 600; margin-bottom: 8px; }
    .action-card input { width: calc(100% - 22px); text-align: center; }
    .hint { color: #546579; font-size: 13px; line-height: 1.5; margin-top: 8px; min-height: 38px; }
    .adjust { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
    .adjust button { height: 28px; padding: 0 8px; background: #eef3f8; color: #18324a; border: 1px solid #ccd6e0; }
    .status { color: #546579; margin-left: 10px; }
    .block-info { margin-top: 14px; padding: 12px 14px; background: #f3f8ff; border: 1px solid #cfe0f5; border-radius: 8px; line-height: 1.65; }
    .block-info strong { color: #18324a; }
    .block-info .muted { color: #596b7d; }
    .hidden { display: none; }
    .busy { position: fixed; inset: 0; background: rgba(255,255,255,.72); display: none; align-items: center; justify-content: center; z-index: 20; }
    .busy.show { display: flex; }
    .spinner { width: 42px; height: 42px; border: 5px solid #c9d7e6; border-top-color: #1f6feb; border-radius: 50%; animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header><strong>生产企业人类专家数据采集</strong></header>
  <main>
    <section id="intro" class="panel intro">
      <h1>任务目标</h1>
      <p>你将扮演“生产企业”的经营决策者。每天系统会展示企业现金、债务、还款压力、采购需求和市场价格等关键信息。</p>
      <p>请填写今天希望申请的贷款金额、原料K采购需求、原料L采购需求和产品K销售价格。目标不是某一天赚最多，而是尽量让企业活得更久、经营更稳定。</p>
      <div class="row" style="justify-content:center;margin-top:24px">
        <label>参与者编号（可选） <input id="participant" value="anonymous" /></label>
        <label>访问口令（如有） <input id="accessPassword" type="password" /></label>
        <button id="startBtn">开始采集</button>
      </div>
    </section>

    <section id="app" class="hidden">
      <div class="panel">
        <div class="row">
          <strong id="dayTitle">第 - 天</strong>
          <span id="status" class="status"></span>
        </div>
        <div id="blockInfo" class="block-info"></div>
      </div>

      <div class="panel">
        <table>
          <thead><tr><th>信息</th><th>参考数值</th><th>相对上一天</th></tr></thead>
          <tbody id="stateRows"></tbody>
        </table>
      </div>

      <div class="panel">
        <div class="actions" id="actions"></div>
      </div>

      <div class="panel">
        <div class="row">
          <button id="prevBtn" class="secondary" disabled>查看上一天</button>
          <button id="submitBtn">提交动作并进入下一天</button>
          <button id="skipBtn" class="secondary" disabled>完成本段，跳到下一段</button>
          <button id="nextEpisodeBtn" class="secondary" disabled>开始下一回合</button>
          <button id="endBtn" class="danger">结束采集</button>
        </div>
      </div>
    </section>
  </main>
  <div id="busy" class="busy"><div class="spinner"></div></div>

<script>
const actionNames = ["申请贷款金额", "原料K采购需求", "原料L采购需求", "产品K销售价格"];
let sessionId = null;
let current = null;
let previousSnapshot = null;
let viewingPrevious = false;
let draftValues = null;
let block = null;

function $(id) { return document.getElementById(id); }
function busy(show) { $("busy").classList.toggle("show", show); }

async function api(path, payload) {
  busy(true);
  try {
    const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload || {})});
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "请求失败");
    return data;
  } finally {
    busy(false);
  }
}

function renderState(data, readonly=false) {
  $("dayTitle").textContent = `生产企业第 ${data.day} 天可见的信息${readonly ? "（只读查看）" : ""}`;
  const tbody = $("stateRows");
  tbody.innerHTML = "";
  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    (row.tags || []).forEach(tag => tr.classList.add(tag));
    tr.innerHTML = `<td>${row.name}</td><td>${row.value}</td><td>${row.change}</td>`;
    tbody.appendChild(tr);
  });
}

function renderBlockInfo(status) {
  block = status;
  if (!status) {
    $("blockInfo").innerHTML = "";
    $("skipBtn").disabled = true;
    $("skipBtn").textContent = "完成本段，跳到下一段";
    return;
  }
  const skipText = status.can_skip
    ? `本段已完成 ${status.manual_days_in_block} 天，可以跳到第 ${status.next_block_start} 天继续决策`
    : (status.next_block_start
        ? `还需要完成 ${status.remaining_before_skip} 天人工决策后，才能跳到第 ${status.next_block_start} 天`
        : "当前已经没有预设的下一段");
  $("blockInfo").innerHTML = `
    <div><strong>当前进度：</strong>第 ${status.day} 天，${status.block_title}。本段已完成 ${status.manual_days_in_block}/${status.min_human_days} 天人工决策。</div>
    <div class="muted"><strong>跳转提示：</strong>${skipText}。</div>
    <div class="muted"><strong>下一段为什么需要决策：</strong>${status.next_block_reason}</div>
  `;
  $("skipBtn").disabled = !status.can_skip;
  $("skipBtn").textContent = status.can_skip
    ? `跳到第 ${status.next_block_start} 天`
    : (status.next_block_start ? `还差 ${status.remaining_before_skip} 天可跳过` : "没有下一段");
}

function renderActions(defaults, hints, limits, disabled=false) {
  const box = $("actions");
  box.innerHTML = "";
  actionNames.forEach((name, i) => {
    const limit = limits && limits[i] ? limits[i] : {min: 0, max: ""};
    const maxAttr = Number.isFinite(Number(limit.max)) ? `max="${limit.max}"` : "";
    const card = document.createElement("div");
    card.className = "action-card";
    card.innerHTML = `
      <label>${name}</label>
      <input id="action${i}" type="number" step="0.01" min="${limit.min}" ${maxAttr} value="${defaults[i] || 0}" onblur="clampInput(${i})" ${disabled ? "disabled" : ""}/>
      <div class="hint">${hints[i] || ""}</div>
      <div class="adjust">
        <button type="button" onclick="adjust(${i},0.9)" ${disabled ? "disabled" : ""}>减少10%</button>
        <button type="button" onclick="adjust(${i},1.1)" ${disabled ? "disabled" : ""}>增加10%</button>
        <button type="button" onclick="add(${i},-1)" ${disabled ? "disabled" : ""}>-1</button>
        <button type="button" onclick="add(${i},1)" ${disabled ? "disabled" : ""}>+1</button>
        <button type="button" onclick="setZero(${i})" ${disabled ? "disabled" : ""}>设为0</button>
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
    if (!Number.isFinite(value)) throw new Error(`请填写${actionNames[i]}。`);
    if (value < min || (Number.isFinite(max) && value > max)) {
      const maxText = Number.isFinite(max) ? max : "不限";
      throw new Error(`${actionNames[i]}只能填写 ${min} 到 ${maxText} 之间的数。`);
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
function setZero(i) { const el = $("action" + i); el.value = formatInputValue(bounded(i, 0)); }

async function start() {
  const data = await api("/api/start", {participant_id: $("participant").value, password: $("accessPassword").value});
  sessionId = data.session_id;
  current = data.state;
  block = data.block;
  previousSnapshot = null;
  $("intro").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("status").textContent = "环境已初始化，请填写今天的决策。";
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
    $("status").textContent = `本回合结束，存活 ${data.survival_days} 天；已保存 ${data.rows_saved} 条专家数据。`;
  } else {
    $("status").textContent = `已进入第 ${current.day} 天；已保存 ${data.rows_saved} 条专家数据。`;
    renderState(current);
    renderActions(current.defaults, current.hints, current.limits);
    renderBlockInfo(data.block);
  }
}

async function skipToNextBlock() {
  const data = await api("/api/skip_to_next_block", {session_id: sessionId});
  previousSnapshot = null;
  viewingPrevious = false;
  current = data.state;
  block = data.block;
  $("prevBtn").disabled = true;
  $("prevBtn").textContent = "查看上一天";
  if (data.done) {
    $("submitBtn").disabled = true;
    $("skipBtn").disabled = true;
    $("nextEpisodeBtn").disabled = false;
    $("status").textContent = `自动推进过程中本回合结束，存活 ${data.survival_days} 天；已保存 ${data.rows_saved} 条专家数据。`;
  } else {
    $("submitBtn").disabled = false;
    $("status").textContent = `系统已自动推进 ${data.auto_days} 天，现在到第 ${current.day} 天，开始采集：${data.block.block_title}。`;
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
    $("prevBtn").textContent = "返回今天";
    $("submitBtn").disabled = true;
    $("skipBtn").disabled = true;
    renderState(previousSnapshot, true);
    renderActions(previousSnapshot.human_values, previousSnapshot.hints, previousSnapshot.limits, true);
    $("status").textContent = "正在查看上一天记录：这里只能查看，不能修改。";
  } else {
    viewingPrevious = false;
    $("prevBtn").textContent = "查看上一天";
    $("submitBtn").disabled = false;
    renderBlockInfo(block);
    renderState(current);
    renderActions(draftValues || current.defaults, current.hints, current.limits);
    $("status").textContent = "已返回今天，请继续填写今天的决策。";
  }
}

async function nextEpisode() {
  const data = await api("/api/next_episode", {session_id: sessionId});
  current = data.state;
  block = data.block;
  previousSnapshot = null;
  viewingPrevious = false;
  $("prevBtn").disabled = true;
  $("prevBtn").textContent = "查看上一天";
  $("submitBtn").disabled = false;
  $("skipBtn").disabled = true;
  $("nextEpisodeBtn").disabled = true;
  $("status").textContent = "新回合已开始，请填写今天的决策。";
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
  $("status").textContent = "采集已结束，数据已保存在服务器。";
}

$("startBtn").onclick = () => start().catch(e => alert(e.message));
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
                self._send_json({"error": "未知接口"}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.address_string(), fmt % args))


def get_session(session_id):
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise ValueError("实验会话不存在或已结束，请重新开始。")
    return session


def api_start(payload):
    if ACCESS_PASSWORD and payload.get("password", "") != ACCESS_PASSWORD:
        raise ValueError("访问口令不正确。")
    participant_id = payload.get("participant_id", "anonymous")
    auto_policy = payload.get("auto_policy", SERVER_AUTO_POLICY)
    if auto_policy not in {"td3", "fixed"}:
        raise ValueError("auto_policy 只能是 td3 或 fixed。")
    session_id = uuid.uuid4().hex
    output_path, meta_output_path = session_paths(participant_id, session_id)
    collector = HumanProductionCollector(
        seed=DEFAULT_SEED,
        output_path=output_path,
        meta_output_path=meta_output_path,
        auto_policy=auto_policy,
        participant_id=participant_id,
    )
    collector.start_episode()
    session = {
        "collector": collector,
        "lock": threading.Lock(),
        "started_at": time.perf_counter(),
        "decision_started_at": time.perf_counter(),
        "durations": [],
        "participant_id": participant_id,
        "auto_policy": auto_policy,
        "output_path": output_path,
        "meta_output_path": meta_output_path,
        "block_index": 0,
        "block_start_day": DECISION_BLOCKS[0]["start"],
        "human_days_in_block": 0,
    }
    with SESSIONS_LOCK:
        SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "state": serialize_state(collector),
        "block": block_status(session),
        "output_path": output_path,
        "meta_output_path": meta_output_path,
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
        done, _ = collector.step(model_action, values, decision_seconds)
        session["human_days_in_block"] = int(session.get("human_days_in_block", 0)) + 1
        previous = serialize_state(collector, state_before, day_before, readonly=True)
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
            result["state"] = serialize_state(collector, state_before, day_before, readonly=True)
        else:
            result["state"] = serialize_state(collector)
        return result


def api_skip_to_next_block(payload):
    session = get_session(payload.get("session_id"))
    with session["lock"]:
        collector = session["collector"]
        status = block_status(session)
        if not status["can_skip"]:
            if status["next_block_start"] is None:
                raise ValueError("当前已经没有预设的下一段，可以继续人工决策或结束采集。")
            raise ValueError(f"本段还需要完成 {status['remaining_before_skip']} 天人工决策后才能跳过。")

        target_index, target_block = next_block_after(collector.env.day)
        target_day = int(target_block["start"])
        auto_days = 0
        last_auto = None
        done = False

        while collector.env.day < target_day:
            last_auto = collector.auto_step()
            auto_days += 1
            done = bool(last_auto["done"])
            if done:
                break

        if not done:
            session["block_index"] = target_index
            session["block_start_day"] = target_day
            session["human_days_in_block"] = 0
            session["decision_started_at"] = time.perf_counter()

        previous_state = last_auto["state"] if last_auto is not None else None
        result = {
            "done": done,
            "auto_days": auto_days,
            "rows_saved": collector.rows_saved,
            "stats": session_stats(session),
            "block": block_status(session),
        }
        if done:
            result["survival_days"] = collector.env.day
            result["state"] = serialize_state(collector)
        else:
            result["state"] = serialize_state(collector, previous_state_override=previous_state)
        return result


def api_next_episode(payload):
    session = get_session(payload.get("session_id"))
    with session["lock"]:
        collector = session["collector"]
        collector.start_episode()
        session["decision_started_at"] = time.perf_counter()
        session["block_index"] = 0
        session["block_start_day"] = DECISION_BLOCKS[0]["start"]
        session["human_days_in_block"] = 0
        return {"state": serialize_state(collector), "block": block_status(session)}


def api_end(payload):
    session_id = payload.get("session_id")
    session = get_session(session_id)
    with session["lock"]:
        stats = session_stats(session)
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)
    return {"ok": True, "stats": stats}


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"网页实验系统已启动：http://127.0.0.1:{PORT}")
    print(f"局域网访问请使用本机IP，例如：http://<你的电脑IP>:{PORT}")
    print(f"非人工主体策略：{SERVER_AUTO_POLICY}")
    print("访问口令：" + ("已启用" if ACCESS_PASSWORD else "未启用"))
    print(f"数据保存目录：{WEB_DATA_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
