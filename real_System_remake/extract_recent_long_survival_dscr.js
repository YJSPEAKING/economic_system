const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const readline = require("readline");

const BASE_DIR = __dirname;
function argumentValue(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length
    ? process.argv[index + 1]
    : fallback;
}

const ALGORITHM = argumentValue("--algorithm", "GAIL+TD3");
const SEED = Number(argumentValue("--seed", "739"));
const DSCR_DIVISOR = Number(argumentValue("--dscr-divisor", "1"));
const MIN_DSCR = Number(argumentValue("--min-dscr", "-Infinity"));
if (
  !Number.isFinite(SEED) ||
  !Number.isFinite(DSCR_DIVISOR) ||
  DSCR_DIVISOR <= 0 ||
  Number.isNaN(MIN_DSCR)
) {
  throw new Error("Seed and DSCR divisor must be valid positive numbers.");
}
const INPUT_PATH = path.join(
  BASE_DIR,
  "trajectory_logs",
  ALGORITHM,
  `seed_${SEED}`,
  "production1_daily_trajectory.csv"
);
const OUTPUT_DIR = path.join(
  BASE_DIR,
  "analysis_plots",
  "post_training_dscr",
  `${ALGORITHM}_seed_${SEED}_recent_100_survival_gt_90`
);
const REQUIRED_EPISODES = 100;
const MIN_SURVIVAL_DAYS_EXCLUSIVE = 90;

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filePath, rows) {
  if (rows.length === 0) {
    throw new Error(`Cannot write an empty CSV: ${filePath}`);
  }
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.map(csvEscape).join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ];
  await fsp.writeFile(filePath, `${lines.join("\n")}\n`, "utf8");
}

function quantile(sorted, probability) {
  const position = (sorted.length - 1) * probability;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) return sorted[lowerIndex];
  return (
    sorted[lowerIndex] +
    (sorted[upperIndex] - sorted[lowerIndex]) * (position - lowerIndex)
  );
}

function calculateBoxplot(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) throw new Error("No finite DSCR observations found.");
  const q1 = quantile(sorted, 0.25);
  const median = quantile(sorted, 0.5);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;
  const retained = sorted.filter(
    (value) => value >= lowerFence && value <= upperFence
  );
  return {
    statistics: {
      count_all: sorted.length,
      count_retained: retained.length,
      count_outliers: sorted.length - retained.length,
      outlier_share_percent:
        (100 * (sorted.length - retained.length)) / sorted.length,
      minimum_all: sorted[0],
      maximum_all: sorted[sorted.length - 1],
      q1,
      median,
      q3,
      iqr,
      lower_fence: lowerFence,
      upper_fence: upperFence,
      lower_whisker: retained[0],
      upper_whisker: retained[retained.length - 1],
      mean_all: sorted.reduce((total, value) => total + value, 0) / sorted.length,
      mean_retained:
        retained.reduce((total, value) => total + value, 0) / retained.length,
      median_retained: quantile(retained, 0.5),
    },
    retained,
  };
}

function createEpisode(episode) {
  return {
    episode,
    survivalDays: 0,
    previousDscrCount: 0,
    observations: [],
    validBeforeMinimumFilter: 0,
    removedByMinimumFilter: 0,
  };
}

function finishEpisode(current, selected) {
  if (current === null) return;
  if (current.survivalDays <= MIN_SURVIVAL_DAYS_EXCLUSIVE) return;
  selected.push(current);
  if (selected.length > REQUIRED_EPISODES) selected.shift();
}

async function selectRecentEpisodes() {
  const input = fs.createReadStream(INPUT_PATH, { encoding: "utf8" });
  const lines = readline.createInterface({ input, crlfDelay: Infinity });
  let columnIndex = null;
  let current = null;
  const selected = [];

  for await (const line of lines) {
    if (!line) continue;
    if (columnIndex === null) {
      const columns = line.replace(/^\uFEFF/, "").split(",");
      columnIndex = {
        episode: columns.indexOf("episode"),
        day: columns.indexOf("day"),
        dscr: columns.indexOf("raw_dscr"),
        dscrCount: columns.indexOf("raw_dscr_count"),
      };
      for (const [name, index] of Object.entries(columnIndex)) {
        if (index < 0) throw new Error(`Missing CSV column: ${name}`);
      }
      continue;
    }

    const fields = line.split(",");
    const episode = Number.parseInt(fields[columnIndex.episode], 10);
    const day = Number.parseInt(fields[columnIndex.day], 10);
    const dscr = Number.parseFloat(fields[columnIndex.dscr]);
    const dscrCount = Number.parseInt(fields[columnIndex.dscrCount], 10);
    if (!Number.isFinite(episode) || !Number.isFinite(day)) continue;

    if (current === null || episode !== current.episode) {
      finishEpisode(current, selected);
      current = createEpisode(episode);
    }
    current.survivalDays = Math.max(current.survivalDays, day);

    if (
      day >= 5 &&
      Number.isFinite(dscrCount) &&
      dscrCount > current.previousDscrCount &&
      Number.isFinite(dscr)
    ) {
      const adjustedDscr = dscr / DSCR_DIVISOR;
      current.validBeforeMinimumFilter += 1;
      if (adjustedDscr < MIN_DSCR) {
        current.removedByMinimumFilter += 1;
      } else {
        current.observations.push({
          algorithm: ALGORITHM,
          seed: SEED,
          episode,
          day,
          dscr: adjustedDscr,
          dscr_below_1: adjustedDscr < 1 ? 1 : 0,
        });
      }
    }
    if (Number.isFinite(dscrCount)) current.previousDscrCount = dscrCount;
  }
  finishEpisode(current, selected);
  return selected;
}

async function main() {
  if (!fs.existsSync(INPUT_PATH)) {
    throw new Error(`Input trajectory does not exist: ${INPUT_PATH}`);
  }
  await fsp.mkdir(OUTPUT_DIR, { recursive: true });

  const selected = await selectRecentEpisodes();
  if (selected.length < REQUIRED_EPISODES) {
    throw new Error(
      `Only ${selected.length} episodes have survival days > ${MIN_SURVIVAL_DAYS_EXCLUSIVE}; ${REQUIRED_EPISODES} are required.`
    );
  }

  const allRows = selected.flatMap((episode) => episode.observations);
  const allValues = allRows.map((row) => row.dscr);
  const { statistics, retained } = calculateBoxplot(allValues);

  const retainedCounts = new Map();
  for (const value of retained) {
    retainedCounts.set(value, (retainedCounts.get(value) ?? 0) + 1);
  }
  const retainedRows = [];
  for (const row of allRows) {
    const count = retainedCounts.get(row.dscr) ?? 0;
    if (count > 0) {
      retainedRows.push({ ...row, iqr_retained: 1 });
      retainedCounts.set(row.dscr, count - 1);
    }
  }

  const selectedEpisodeRows = selected.map((episode) => ({
    algorithm: ALGORITHM,
    seed: SEED,
    episode: episode.episode,
    survival_days: episode.survivalDays,
    valid_dscr_days_before_minimum_filter: episode.validBeforeMinimumFilter,
    removed_by_minimum_filter: episode.removedByMinimumFilter,
    valid_dscr_days: episode.observations.length,
    mean_dscr:
      episode.observations.reduce((total, row) => total + row.dscr, 0) /
      episode.observations.length,
    median_dscr: quantile(
      episode.observations.map((row) => row.dscr).sort((a, b) => a - b),
      0.5
    ),
    dscr_below_1_count: episode.observations.filter((row) => row.dscr < 1).length,
    dscr_below_1_share_percent:
      (100 * episode.observations.filter((row) => row.dscr < 1).length) /
      episode.observations.length,
  }));

  const belowOneAll = allRows.filter((row) => row.dscr < 1).length;
  const belowOneRetained = retainedRows.filter((row) => row.dscr < 1).length;
  const validBeforeMinimumFilter = selected.reduce(
    (total, episode) => total + episode.validBeforeMinimumFilter,
    0
  );
  const removedByMinimumFilter = selected.reduce(
    (total, episode) => total + episode.removedByMinimumFilter,
    0
  );
  const riskRow = {
    algorithm: ALGORITHM,
    seed: SEED,
    selected_episode_count: selected.length,
    selection_rule: "latest episodes with survival_days > 90",
    first_selected_episode: selected[0].episode,
    last_selected_episode: selected[selected.length - 1].episode,
    minimum_dscr_filter: Number.isFinite(MIN_DSCR) ? MIN_DSCR : "",
    valid_firm_day_observations_before_minimum_filter: validBeforeMinimumFilter,
    removed_by_minimum_filter: removedByMinimumFilter,
    valid_firm_day_observations_all: allRows.length,
    dscr_below_1_count_all: belowOneAll,
    dscr_below_1_share_percent_all: (100 * belowOneAll) / allRows.length,
    valid_firm_day_observations_iqr_retained: retainedRows.length,
    dscr_below_1_count_iqr_retained: belowOneRetained,
    dscr_below_1_share_percent_iqr_retained:
      (100 * belowOneRetained) / retainedRows.length,
  };

  await writeCsv(
    path.join(OUTPUT_DIR, "selected_episodes.csv"),
    selectedEpisodeRows
  );
  await writeCsv(
    path.join(OUTPUT_DIR, "production_dscr_all_firm_days.csv"),
    allRows
  );
  await writeCsv(
    path.join(OUTPUT_DIR, "production_dscr_iqr_retained_firm_days.csv"),
    retainedRows
  );
  await writeCsv(
    path.join(OUTPUT_DIR, "production_dscr_boxplot_statistics.csv"),
    [{ algorithm: ALGORITHM, seed: SEED, ...statistics }]
  );
  await writeCsv(
    path.join(OUTPUT_DIR, "production_dscr_below_1_share.csv"),
    [riskRow]
  );

  const metadata = {
    source: path.relative(BASE_DIR, INPUT_PATH),
    algorithm: ALGORITHM,
    seed: SEED,
    selection:
      "The most recent 100 episodes whose maximum recorded day is strictly greater than 90.",
    dscr_formula:
      DSCR_DIVISOR === 1
        ? "DSCR = money / (should_payback + iDebt), using stored raw_dscr."
        : `DSCR = money / (${DSCR_DIVISOR} * (should_payback + iDebt)), reconstructed as raw_dscr / ${DSCR_DIVISOR}.`,
    valid_dscr_observation:
      "day >= 5 and raw_dscr_count increases within the episode.",
    boxplot:
      "Q1 and Q3 use linear interpolation. Fences are Q1-1.5*IQR and Q3+1.5*IQR. Whiskers are observed extrema inside the fences.",
    risk_share_primary:
      "DSCR < 1 divided by all valid firm-day observations from the selected episodes.",
    minimum_dscr_filter: Number.isFinite(MIN_DSCR)
      ? `Observations with DSCR < ${MIN_DSCR} are removed before all statistics.`
      : "No minimum DSCR filter is applied.",
  };
  await fsp.writeFile(
    path.join(OUTPUT_DIR, "metadata.json"),
    `${JSON.stringify(metadata, null, 2)}\n`,
    "utf8"
  );

  process.stdout.write(
    `${JSON.stringify(
      {
        output_dir: OUTPUT_DIR,
        selected_episode_count: selected.length,
        first_selected_episode: selected[0].episode,
        last_selected_episode: selected[selected.length - 1].episode,
        survival_days_min: Math.min(...selected.map((item) => item.survivalDays)),
        survival_days_max: Math.max(...selected.map((item) => item.survivalDays)),
        boxplot: statistics,
        risk: riskRow,
      },
      null,
      2
    )}\n`
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
