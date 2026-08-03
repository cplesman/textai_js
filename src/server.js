const express = require("express");
const { TextCategoryAI, assertValidLabel } = require("./classifier");

const app = express();
const PORT = 3200;

app.use(express.json({ limit: "2mb" }));

let aiCache = {};
const gModelPath = 'models/';

function validateModelPath(modelPath) {
  if (!modelPath || typeof modelPath !== "string" || modelPath.trim().length === 0) {
    throw new Error("modelPath must be a non-empty string.");
  }
  if(!modelPath.endsWith(".json")) {
    throw new Error("modelPath must end with .json");
  }
  if(modelPath.includes("..") || modelPath.includes("/") || modelPath.includes("\\")) {
    throw new Error("modelPath must not contain directory traversal or path separators.");
  }
}
async function validateModel(model, reset) {
  validateModelPath(model);
  let ai = aiCache[model];
  if(reset) {
    ai = aiCache[model] = new TextCategoryAI();
  }
  else if(!ai) {
    ai = await TextCategoryAI.loadModel(gModelPath + model); //load or create new model if not exists
    aiCache[model] = ai;
  }
  return ai;
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function validateSample(sample, indexLabel) {
  if (!sample || typeof sample !== "object") {
    throw new Error(`${indexLabel} must be an object with text and label.`);
  }

  if (!isNonEmptyString(sample.text)) {
    throw new Error(`${indexLabel}.text must be a non-empty string.`);
  }

  assertValidLabel(sample.label);
}

function validateSamples(samples, fieldName) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error(`${fieldName} must be a non-empty array of { text, label }.`);
  }

  for (let i = 0; i < samples.length; i += 1) {
    validateSample(samples[i], `${fieldName}[${i}]`);
  }
}

function evaluateSamples(model, samples) {
  validateSamples(samples, "samples");

  const ai = aiCache[model];
  if (!ai || !ai.trained) {
    throw new Error("No trained model is loaded. Call /train or /load first.");
  }

  let correct = 0;
  const results = samples.map((sample, index) => {
    const expected = Number(sample.label);
    const predicted = ai.predict(sample.text);
    const matches = expected === predicted;

    if (matches) {
      correct += 1;
    }

    return {
      index,
      expected,
      predicted,
      matches,
    };
  });

  const total = samples.length;
  const accuracy = total > 0 ? correct / total : 0;

  return {
    total,
    correct,
    incorrect: total - correct,
    accuracy,
    results,
  };
}

function asyncHandler(handler) {
  return async (req, res) => {
    try {
      await handler(req, res);
    } catch (err) {
      res.status(400).json({
        error: err.message || "Request failed.",
      });
    }
  };
}

app.get("/health", async (_req, res) => {
  let { model, all } = _req.query;
  if (!model || all) {
    res.json({
      ok: true,
      cachedModels: Object.keys(aiCache),
    });
    return;
  }
  const ai = await validateModel(model);
  res.json({
    ok: true,
    port: PORT,
    labelDocCounts: Object.fromEntries(ai.labelDocCounts),
    labelTokenTotals: Object.fromEntries(ai.labelTokenTotals),
    labelWordCounts: Object.fromEntries(
      [...ai.labelWordCounts.entries()].map(([label, counts]) => [label, Object.fromEntries(counts)]),
    ),
    modelTrained: ai.trained,
  });
});

app.post("/clearcache", asyncHandler(async (req, res) => {
    //this will save and clear the cache for the model or all models if all is true
  const { model,all } = req.body || {};
  validateModelPath(model);
  const modelPath = '../models/';
  if (all) {
    let aiKeys = Object.keys(aiCache);
    for(let i=0;i<aiKeys.length;i++) {
      const key = aiKeys[i];
      const path = modelPath + key;
      await aiCache[key].saveModel(path);
    }
    aiCache = {};
  } else {
    const ai = aiCache[model];
    await ai.saveModel(modelPath + model);
    delete aiCache[model];
  }
  res.json({
    message: all ? "Cache cleared for all models." : "Cache cleared for model.",
    model,
  });
}));

app.post("/train", asyncHandler(async (req, res) => {
  const { model, samples, reset = false } = req.body || {};
  validateSamples(samples, "samples");
  const ai = await validateModel(model,reset);

  ai.train(samples);

  res.json({
    message: "Training complete.",
    trainedOn: samples.length,
    reset,
    modelTrained: ai.trained,
  });
}));

app.post("/batch-train", asyncHandler(async (req, res) => {
  const { model, batches, reset = false } = req.body || {};

  if (!Array.isArray(batches) || batches.length === 0) {
    throw new Error("batches must be a non-empty array of sample arrays.");
  }
  
  const ai = await validateModel(model, reset);

  let totalSamples = 0;
  const batchStats = [];

  for (let i = 0; i < batches.length; i += 1) {
    const samples = batches[i];
    validateSamples(samples, `batches[${i}]`);
    ai.train(samples);
    totalSamples += samples.length;
    batchStats.push({ batchIndex: i, trainedOn: samples.length });
  }

  res.json({
    message: "Batch training complete.",
    batchCount: batches.length,
    totalSamples,
    reset,
    batches: batchStats,
    modelTrained: ai.trained,
  });
}));

app.post("/evaluate", asyncHandler(async (req, res) => {
  const { model, samples } = req.body || {};
  const evaluation = evaluateSamples(model, samples);

  res.json({
    message: "Evaluation complete.",
    ...evaluation,
  });
}));

app.post("/batch-evaluate", asyncHandler(async (req, res) => {
  const { model, batches } = req.body || {};

  if (!Array.isArray(batches) || batches.length === 0) {
    throw new Error("batches must be a non-empty array.");
  }

  const evaluations = batches.map((batch, index) => {
    const name = isNonEmptyString(batch?.name) ? batch.name : `batch-${index}`;
    const samples = batch?.samples;
    const summary = evaluateSamples(model, samples);

    return {
      name,
      batchIndex: index,
      ...summary,
    };
  });

  res.json({
    message: "Batch evaluation complete.",
    batchCount: evaluations.length,
    evaluations,
  });
}));

app.post("/save", asyncHandler(async (req, res) => {
  const { model } = req.body || {};

  const ai = await validateModel(model, false);
  const savedPath = await ai.saveModel(gModelPath + model);

  res.json({
    message: "Model saved.",
    modelPath: savedPath,
  });
}));

app.get("/models", asyncHandler(async (_req, res) => {
  const modelDir = 'models/';
  const fs = require("fs");
  const path = require("path");
  const models = fs.existsSync(path.resolve(modelDir))
    ? fs.readdirSync(path.resolve(modelDir)).filter((file) => file.endsWith(".json"))
    : [];

  res.json({
    models,
  });
}));

app.listen(PORT, () => {
  console.log(`TextCategoryAI API server listening on port ${PORT}`);
});
