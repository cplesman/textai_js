#!/usr/bin/env node
const { TextCategoryAI, MIN_LABEL, MAX_LABEL } = require("./classifier");

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        result[key] = next;
        i += 1;
      } else {
        result[key] = true;
      }
    }
  }
  return result;
}

function printUsage() {
  console.log("TextCategoryAI CLI");
  console.log("");
  console.log("Commands:");
  console.log("  train --data <dataset.json> --model <model.json>");
  console.log("  predict --model <model.json> --text \"some paragraph\"");
  console.log("");
  console.log(`Label range: ${MIN_LABEL}-${MAX_LABEL}`);
}

async function runTrain(args) {
  if (!args.data || !args.model) {
    throw new Error("train requires --data and --model");
  }

  const samples = TextCategoryAI.loadSamplesFromFile(args.data);
  const ai = new TextCategoryAI();
  ai.train(samples);
  const savedPath = await ai.saveModel(args.model);

  console.log(`Model trained on ${samples.length} samples and saved to: ${savedPath}`);
}

async function runPredict(args) {
  if (!args.model || !args.text) {
    throw new Error("predict requires --model and --text");
  }

  const ai = await TextCategoryAI.loadModel(args.model);
  const prediction = ai.predict(args.text);

  console.log(prediction);
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);

  if (!command || command === "help" || command === "--help") {
    printUsage();
    return;
  }

  const args = parseArgs(rest);

  if (command === "train") {
    await runTrain(args);
    return;
  }

  if (command === "predict") {
    await runPredict(args);
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
