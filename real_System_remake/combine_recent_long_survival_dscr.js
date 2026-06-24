const fs = require("fs/promises");
const path = require("path");

const BASE_DIR = __dirname;
const ROOT = path.join(BASE_DIR, "analysis_plots", "post_training_dscr");
const OUTPUT_DIR = path.join(ROOT, "three_algorithm_recent_100_survival_gt_90");
const SOURCES = [
  "GAIL+TD3_seed_191_recent_100_survival_gt_90",
  "TD3_seed_739_recent_100_survival_gt_90",
  "Transformer+GAIL+TD3_seed_192_recent_100_survival_gt_90",
];

function parseSingleRowCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  const values = lines[1].split(",");
  return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
}

async function combine(fileName, outputName) {
  const rows = [];
  for (const source of SOURCES) {
    const text = await fs.readFile(path.join(ROOT, source, fileName), "utf8");
    rows.push(parseSingleRowCsv(text.replace(/^\uFEFF/, "")));
  }
  const headers = Object.keys(rows[0]);
  const output = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => row[header]).join(",")),
  ].join("\n");
  await fs.writeFile(path.join(OUTPUT_DIR, outputName), `${output}\n`, "utf8");
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await combine(
    "production_dscr_boxplot_statistics.csv",
    "three_algorithm_boxplot_statistics.csv"
  );
  await combine(
    "production_dscr_below_1_share.csv",
    "three_algorithm_dscr_below_1_share.csv"
  );
  process.stdout.write(`${OUTPUT_DIR}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
