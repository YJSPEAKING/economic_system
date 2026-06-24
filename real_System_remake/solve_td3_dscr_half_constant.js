const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const readline = require("readline");

const BASE_DIR = __dirname;
const TD3_DIR = path.join(BASE_DIR, "trajectory_logs", "TD3");
const OUTPUT_DIR = path.join(
  BASE_DIR,
  "analysis_plots",
  "trajectory_dscr_stable_analysis"
);
const STABLE_EPISODES = 1000;
const CURRENT_CONSTANT = 2.5;

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
  const episode = Number.parseInt(line.slice(0, line.indexOf(",")), 10);
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

async function readSeedData(seed) {
  const filePath = path.join(
    TD3_DIR,
    `seed_${seed}`,
    "production1_daily_trajectory.csv"
  );
  const header = await readHeader(filePath);
  const columns = header.split(",");
  const required = {
    episode: columns.indexOf("episode"),
    day: columns.indexOf("day"),
    dscr: columns.indexOf("raw_dscr"),
    dscrCount: columns.indexOf("raw_dscr_count"),
    shouldPayback: columns.indexOf("raw_should_payback"),
    interestDue: columns.indexOf("raw_iDebt"),
  };
  for (const [name, index] of Object.entries(required)) {
    if (index < 0) throw new Error(`Missing ${name} in ${filePath}`);
  }

  const lastLine = await readLastDataLine(filePath);
  const lastEpisode = Number.parseInt(lastLine.slice(0, lastLine.indexOf(",")), 10);
  const startEpisode = Math.max(0, lastEpisode - STABLE_EPISODES + 1);
  const startOffset = await findApproximateOffset(filePath, startEpisode);
  const episodes = new Map();

  const input = fs.createReadStream(filePath, {
    encoding: "utf8",
    start: startOffset,
  });
  const lines = readline.createInterface({ input, crlfDelay: Infinity });
  let firstLine = true;
  let currentEpisode = null;
  let previousCount = 0;

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
    if (!Number.isFinite(episode) || episode < startEpisode) continue;

    if (episode !== currentEpisode) {
      currentEpisode = episode;
      previousCount = 0;
    }

    const day = Number.parseInt(fields[required.day], 10);
    const count = Number.parseInt(fields[required.dscrCount], 10);
    const rawDscr = Number.parseFloat(fields[required.dscr]);
    const due =
      Number.parseFloat(fields[required.shouldPayback]) +
      Number.parseFloat(fields[required.interestDue]);

    if (
      day >= 5 &&
      Number.isFinite(count) &&
      count > previousCount &&
      Number.isFinite(rawDscr) &&
      Number.isFinite(due) &&
      due > 0
    ) {
      if (!episodes.has(episode)) episodes.set(episode, []);
      episodes.get(episode).push([rawDscr, due]);
    }
    if (Number.isFinite(count)) previousCount = count;
  }

  return {
    seed,
    startEpisode,
    endEpisode: lastEpisode,
    episodes: [...episodes.values()],
  };
}

function stableEpisodeMean(seedData, constant) {
  let episodeTotal = 0;
  let validEpisodes = 0;
  for (const observations of seedData.episodes) {
    if (observations.length === 0) continue;
    let dailyTotal = 0;
    for (const [rawDscr, due] of observations) {
      dailyTotal += (rawDscr * due) / (due + constant);
    }
    episodeTotal += dailyTotal / observations.length;
    validEpisodes += 1;
  }
  return episodeTotal / validEpisodes;
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function quantile(sorted, probability) {
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function dailySummary(seedData, constant) {
  const values = [];
  for (const observations of seedData.episodes) {
    for (const [rawDscr, due] of observations) {
      values.push((rawDscr * due) / (due + constant));
    }
  }
  values.sort((a, b) => a - b);
  return {
    mean: mean(values),
    median: quantile(values, 0.5),
    p10: quantile(values, 0.1),
    p90: quantile(values, 0.9),
  };
}

function solveConstant(objective, target, initialLow = CURRENT_CONSTANT) {
  let low = initialLow;
  let high = Math.max(5, initialLow * 2);
  while (objective(high) > target && high < 1e9) high *= 2;
  for (let iteration = 0; iteration < 80; iteration += 1) {
    const middle = (low + high) / 2;
    if (objective(middle) > target) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return (low + high) / 2;
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filePath, rows) {
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.map(csvEscape).join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ];
  await fsp.writeFile(filePath, `${lines.join("\n")}\n`, "utf8");
}

async function main() {
  const seedDirs = (await fsp.readdir(TD3_DIR, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("seed_"))
    .sort((a, b) => Number(a.name.slice(5)) - Number(b.name.slice(5)));

  const seedData = [];
  for (const seedDir of seedDirs) {
    const seed = Number(seedDir.name.slice(5));
    process.stdout.write(`Reading TD3 seed ${seed}...\n`);
    seedData.push(await readSeedData(seed));
  }

  const currentValues = seedData.map((data) =>
    stableEpisodeMean(data, CURRENT_CONSTANT)
  );
  const currentGroupMean = mean(currentValues);
  const targetGroupMean = currentGroupMean / 2;
  const groupObjective = (constant) =>
    mean(seedData.map((data) => stableEpisodeMean(data, constant)));
  const sharedConstant = solveConstant(groupObjective, targetGroupMean);

  const rows = seedData.map((data, index) => {
    const currentValue = currentValues[index];
    const targetValue = currentValue / 2;
    const seedConstant = solveConstant(
      (constant) => stableEpisodeMean(data, constant),
      targetValue
    );
    const sharedValue = stableEpisodeMean(data, sharedConstant);
    const sharedDaily = dailySummary(data, sharedConstant);
    return {
      seed: data.seed,
      start_episode: data.startEpisode,
      end_episode: data.endEpisode,
      current_constant: CURRENT_CONSTANT,
      current_stable_episode_mean_dscr: currentValue,
      target_half_dscr: targetValue,
      shared_constant: sharedConstant,
      dscr_with_shared_constant: sharedValue,
      shared_to_current_ratio: sharedValue / currentValue,
      shared_daily_mean: sharedDaily.mean,
      shared_daily_median: sharedDaily.median,
      shared_daily_p10: sharedDaily.p10,
      shared_daily_p90: sharedDaily.p90,
      seed_specific_half_constant: seedConstant,
    };
  });

  await fsp.mkdir(OUTPUT_DIR, { recursive: true });
  const outputPath = path.join(OUTPUT_DIR, "td3_dscr_half_constant.csv");
  await writeCsv(outputPath, rows);
  const summary = {
    definition:
      "A shared constant is solved so that the arithmetic mean of the eight TD3 seed-level stable episode-average DSCR values equals half of its value under constant 2.5.",
    stable_window_episodes: STABLE_EPISODES,
    current_constant: CURRENT_CONSTANT,
    current_group_mean_dscr: currentGroupMean,
    target_group_mean_dscr: targetGroupMean,
    solved_shared_constant: sharedConstant,
    achieved_group_mean_dscr: groupObjective(sharedConstant),
  };
  await fsp.writeFile(
    path.join(OUTPUT_DIR, "td3_dscr_half_constant_summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8"
  );
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  process.stdout.write(`Results written to ${outputPath}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
