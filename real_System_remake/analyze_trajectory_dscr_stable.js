const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const readline = require("readline");

const BASE_DIR = __dirname;
const TRAJECTORY_DIR = path.join(BASE_DIR, "trajectory_logs");
const OUTPUT_DIR = path.join(
  BASE_DIR,
  "analysis_plots",
  "trajectory_dscr_stable_analysis"
);
const STABLE_EPISODES = 1000;
const FINAL_EPISODES = 100;
const DSCR_DIVISOR = 2.0;

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function quantile(sorted, probability) {
  if (sorted.length === 0) return NaN;
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function summarize(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  const count = finite.length;
  if (count === 0) {
    return {
      count: 0,
      mean: NaN,
      median: NaN,
      p10: NaN,
      p25: NaN,
      p75: NaN,
      p90: NaN,
      min: NaN,
      max: NaN,
      below1Pct: NaN,
      nearNarrowPct: NaN,
      nearWidePct: NaN,
    };
  }
  const sum = finite.reduce((total, value) => total + value, 0);
  const percentage = (predicate) =>
    (finite.filter(predicate).length / count) * 100;
  return {
    count,
    mean: sum / count,
    median: quantile(finite, 0.5),
    p10: quantile(finite, 0.1),
    p25: quantile(finite, 0.25),
    p75: quantile(finite, 0.75),
    p90: quantile(finite, 0.9),
    min: finite[0],
    max: finite[count - 1],
    below1Pct: percentage((value) => value < 1),
    nearNarrowPct: percentage((value) => value >= 0.8 && value <= 1.2),
    nearWidePct: percentage((value) => value >= 0.5 && value <= 1.5),
  };
}

function summarizeIqrFiltered(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) {
    return {
      originalCount: 0,
      retainedCount: 0,
      removedCount: 0,
      q1: NaN,
      q3: NaN,
      iqr: NaN,
      lowerFence: NaN,
      upperFence: NaN,
      lowerWhisker: NaN,
      upperWhisker: NaN,
      mean: NaN,
      median: NaN,
    };
  }
  const q1 = quantile(finite, 0.25);
  const q3 = quantile(finite, 0.75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;
  const retained = finite.filter(
    (value) => value >= lowerFence && value <= upperFence
  );
  return {
    originalCount: finite.length,
    retainedCount: retained.length,
    removedCount: finite.length - retained.length,
    q1,
    q3,
    iqr,
    lowerFence,
    upperFence,
    lowerWhisker: retained[0],
    upperWhisker: retained[retained.length - 1],
    mean: retained.reduce((total, value) => total + value, 0) / retained.length,
    median: quantile(retained, 0.5),
  };
}

async function readHeader(filePath) {
  const stream = fs.createReadStream(filePath, {
    encoding: "utf8",
    start: 0,
    end: 32767,
  });
  let text = "";
  for await (const chunk of stream) {
    text += chunk;
    const newline = text.indexOf("\n");
    if (newline >= 0) {
      return text.slice(0, newline).replace(/^\uFEFF/, "").replace(/\r$/, "");
    }
  }
  throw new Error(`CSV header not found: ${filePath}`);
}

async function readLastDataLine(filePath) {
  const handle = await fsp.open(filePath, "r");
  try {
    const { size } = await handle.stat();
    let position = size;
    let suffix = "";
    const chunkSize = 65536;
    while (position > 0) {
      const length = Math.min(chunkSize, position);
      position -= length;
      const buffer = Buffer.alloc(length);
      await handle.read(buffer, 0, length, position);
      suffix = buffer.toString("utf8") + suffix;
      const lines = suffix.split(/\r?\n/).filter((line) => line.length > 0);
      if (lines.length >= 2 || position === 0) {
        return lines[lines.length - 1];
      }
    }
  } finally {
    await handle.close();
  }
  throw new Error(`CSV contains no data rows: ${filePath}`);
}

async function lineAtOrAfter(handle, position, size) {
  const chunkSize = Math.min(262144, size - position);
  if (chunkSize <= 0) return null;
  const buffer = Buffer.alloc(chunkSize);
  const { bytesRead } = await handle.read(buffer, 0, chunkSize, position);
  const text = buffer.subarray(0, bytesRead).toString("utf8");
  const firstNewline = position === 0 ? -1 : text.indexOf("\n");
  const startInText = firstNewline + 1;
  const secondNewline = text.indexOf("\n", startInText);
  if (secondNewline < 0) return null;
  const line = text.slice(startInText, secondNewline).replace(/\r$/, "");
  const episodeText = line.slice(0, line.indexOf(","));
  const episode = Number.parseInt(episodeText, 10);
  if (!Number.isFinite(episode)) return null;
  return {
    episode,
    lineStart: position + startInText,
    lineEnd: position + secondNewline + 1,
  };
}

async function findApproximateOffset(filePath, targetEpisode) {
  const handle = await fsp.open(filePath, "r");
  try {
    const { size } = await handle.stat();
    let low = 0;
    let high = size;
    for (let iteration = 0; iteration < 36 && high - low > 262144; iteration += 1) {
      const middle = Math.floor((low + high) / 2);
      const info = await lineAtOrAfter(handle, middle, size);
      if (!info) {
        high = middle;
      } else if (info.episode < targetEpisode) {
        low = info.lineEnd;
      } else {
        high = info.lineStart;
      }
    }
    return Math.max(0, high - 2 * 1024 * 1024);
  } finally {
    await handle.close();
  }
}

function createWindow(startEpisode) {
  return {
    startEpisode,
    validDaily: [],
    episodeAverage: [],
    adjustedDaily: [],
    adjustedEpisodeAverage: [],
    episodeCount: 0,
    validEpisodeCount: 0,
  };
}

function finalizeEpisode(episodeState, windows) {
  if (episodeState.episode === null) return;
  for (const window of windows) {
    if (episodeState.episode < window.startEpisode) continue;
    window.episodeCount += 1;
    if (episodeState.finalCount > 0 && Number.isFinite(episodeState.finalAverage)) {
      window.validEpisodeCount += 1;
      window.episodeAverage.push(episodeState.finalAverage);
    }
    if (episodeState.adjustedCount > 0) {
      window.adjustedEpisodeAverage.push(
        episodeState.adjustedSum / episodeState.adjustedCount
      );
    }
  }
}

async function analyzeFile(filePath, algorithm, seed) {
  const header = await readHeader(filePath);
  const columns = header.split(",");
  const required = {
    episode: columns.indexOf("episode"),
    day: columns.indexOf("day"),
    dscr: columns.indexOf("raw_dscr"),
    dscrCount: columns.indexOf("raw_dscr_count"),
    dscrAverage: columns.indexOf("raw_dscr_avg"),
  };
  for (const [name, index] of Object.entries(required)) {
    if (index < 0) throw new Error(`Missing ${name} column in ${filePath}`);
  }

  const lastLine = await readLastDataLine(filePath);
  const lastEpisode = Number.parseInt(lastLine.slice(0, lastLine.indexOf(",")), 10);
  if (!Number.isFinite(lastEpisode)) {
    throw new Error(`Cannot read final episode from ${filePath}`);
  }

  const stableStart = Math.max(0, lastEpisode - STABLE_EPISODES + 1);
  const finalStart = Math.max(0, lastEpisode - FINAL_EPISODES + 1);
  const windows = [createWindow(stableStart), createWindow(finalStart)];
  const startOffset = await findApproximateOffset(filePath, stableStart);
  const stream = fs.createReadStream(filePath, {
    encoding: "utf8",
    start: startOffset,
  });
  const lines = readline.createInterface({
    input: stream,
    crlfDelay: Infinity,
  });

  let firstLine = true;
  let episodeState = {
    episode: null,
    previousCount: 0,
    finalCount: 0,
    finalAverage: NaN,
    adjustedSum: 0,
    adjustedCount: 0,
  };

  for await (const line of lines) {
    if (!line) continue;
    if (firstLine && startOffset > 0) {
      firstLine = false;
      continue;
    }
    firstLine = false;
    if (line.startsWith("episode,")) continue;
    const fields = line.split(",");
    const episode = Number.parseInt(fields[required.episode], 10);
    if (!Number.isFinite(episode) || episode < stableStart) continue;

    if (episodeState.episode !== episode) {
      finalizeEpisode(episodeState, windows);
      episodeState = {
        episode,
        previousCount: 0,
        finalCount: 0,
        finalAverage: NaN,
        adjustedSum: 0,
        adjustedCount: 0,
      };
    }

    const day = Number.parseInt(fields[required.day], 10);
    const dscr = Number.parseFloat(fields[required.dscr]);
    const count = Number.parseInt(fields[required.dscrCount], 10);
    const average = Number.parseFloat(fields[required.dscrAverage]);
    if (
      Number.isFinite(day) &&
      day >= 5 &&
      Number.isFinite(count) &&
      count > episodeState.previousCount &&
      Number.isFinite(dscr)
    ) {
      const adjustedDscr = dscr / DSCR_DIVISOR;
      for (const window of windows) {
        if (episode >= window.startEpisode) {
          window.validDaily.push(dscr);
          if (Number.isFinite(adjustedDscr)) {
            window.adjustedDaily.push(adjustedDscr);
          }
        }
      }
      if (Number.isFinite(adjustedDscr)) {
        episodeState.adjustedSum += adjustedDscr;
        episodeState.adjustedCount += 1;
      }
    }
    if (Number.isFinite(count)) {
      episodeState.previousCount = count;
      episodeState.finalCount = count;
    }
    if (Number.isFinite(average)) episodeState.finalAverage = average;
  }
  finalizeEpisode(episodeState, windows);

  return windows.map((window, index) => ({
    algorithm,
    seed,
    file: path.relative(BASE_DIR, filePath),
    finalEpisode: lastEpisode,
    window: index === 0 ? `last_${STABLE_EPISODES}_episodes` : `last_${FINAL_EPISODES}_episodes`,
    startEpisode: window.startEpisode,
    endEpisode: lastEpisode,
    episodesObserved: window.episodeCount,
    validEpisodes: window.validEpisodeCount,
    daily: summarize(window.validDaily),
    episodeAverage: summarize(window.episodeAverage),
    adjustedDaily: summarize(window.adjustedDaily),
    adjustedEpisodeAverage: summarize(window.adjustedEpisodeAverage),
    adjustedDailyIqr: summarizeIqrFiltered(window.adjustedDaily),
    adjustedEpisodeAverageIqr: summarizeIqrFiltered(window.adjustedEpisodeAverage),
  }));
}

function flattenResult(result) {
  const row = {
    algorithm: result.algorithm,
    seed: result.seed,
    window: result.window,
    start_episode: result.startEpisode,
    end_episode: result.endEpisode,
    episodes_observed: result.episodesObserved,
    valid_episodes: result.validEpisodes,
  };
  for (const [prefix, summary] of [
    ["daily", result.daily],
    ["episode_avg", result.episodeAverage],
    ["adjusted_daily", result.adjustedDaily],
    ["adjusted_episode_avg", result.adjustedEpisodeAverage],
    ["method2_iqr_daily", result.adjustedDailyIqr],
    ["method2_iqr_episode_avg", result.adjustedEpisodeAverageIqr],
  ]) {
    for (const [key, value] of Object.entries(summary)) {
      row[`${prefix}_${key}`] = value;
    }
  }
  return row;
}

async function writeCsv(filePath, rows) {
  if (rows.length === 0) return;
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.map(csvEscape).join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ];
  await fsp.writeFile(filePath, `${lines.join("\n")}\n`, "utf8");
}

async function main() {
  await fsp.mkdir(OUTPUT_DIR, { recursive: true });
  const algorithms = (await fsp.readdir(TRAJECTORY_DIR, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  const results = [];
  for (const algorithm of algorithms) {
    const algorithmDir = path.join(TRAJECTORY_DIR, algorithm);
    const seedDirs = (await fsp.readdir(algorithmDir, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && entry.name.startsWith("seed_"))
      .sort((a, b) => Number(a.name.slice(5)) - Number(b.name.slice(5)));
    for (const seedDir of seedDirs) {
      const seed = Number(seedDir.name.slice(5));
      const filePath = path.join(
        algorithmDir,
        seedDir.name,
        "production1_daily_trajectory.csv"
      );
      process.stdout.write(`Analyzing ${algorithm} seed ${seed}...\n`);
      results.push(...(await analyzeFile(filePath, algorithm, seed)));
    }
  }

  const rows = results.map(flattenResult);
  await writeCsv(path.join(OUTPUT_DIR, "production_dscr_by_seed.csv"), rows);

  const td3Ranking = rows
    .filter(
      (row) => row.algorithm === "TD3" && row.window === `last_${STABLE_EPISODES}_episodes`
    )
    .sort(
      (a, b) =>
        b.adjusted_daily_nearNarrowPct - a.adjusted_daily_nearNarrowPct ||
        b.adjusted_episode_avg_nearNarrowPct -
          a.adjusted_episode_avg_nearNarrowPct
    );
  await writeCsv(path.join(OUTPUT_DIR, "td3_near_one_ranking.csv"), td3Ranking);

  const method = [
    "Production-enterprise DSCR trajectory analysis",
    "",
    "Source: production1_daily_trajectory.csv for every algorithm and seed.",
    "Valid daily observation: day >= 5 and raw_dscr_count increases relative to the previous row in the same episode.",
    "DSCR value: raw_dscr saved by the environment at the original pre-repayment calculation point.",
    `Stable window: the last ${STABLE_EPISODES} episodes in each trajectory file.`,
    `Final window: the last ${FINAL_EPISODES} episodes in each trajectory file.`,
    "Near 1 (narrow): 0.8 <= DSCR <= 1.2.",
    "Near 1 (wide): 0.5 <= DSCR <= 1.5.",
    "Two distributions are reported: valid daily DSCR and each episode's final raw_dscr_avg.",
    `Method 1 formula: DSCR = money / (${DSCR_DIVISOR} * (should_payback + iDebt)).`,
    `Method 1 daily DSCR is reconstructed as raw_dscr / ${DSCR_DIVISOR}.`,
    "Method 1 episode averages are recalculated from adjusted valid daily observations.",
    "Method 2 applies the 1.5*IQR rule to Method 1 values: Q1 and Q3 use linear interpolation, fences are Q1-1.5*IQR and Q3+1.5*IQR, and whiskers are the minimum and maximum retained observations.",
  ].join("\n");
  await fsp.writeFile(path.join(OUTPUT_DIR, "method.txt"), `${method}\n`, "utf8");
  process.stdout.write(`Results written to ${OUTPUT_DIR}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
