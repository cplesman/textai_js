const fs = require("fs");
const path = require("path");

const MIN_LABEL = 0;
const MAX_LABEL = 63;

function assertValidLabel(label) {
  const n = Number(label);

  if (!Number.isInteger(n) || n < MIN_LABEL || n > MAX_LABEL) {
    throw new Error(`Label must be an integer between ${MIN_LABEL} and ${MAX_LABEL}. Received: ${label}`);
  }

  return n;
}

class TextCategoryAI {
  constructor(state = null) {
    this.labelDocCounts = new Map();
    this.labelWordCounts = new Map();
    this.labelTokenTotals = new Map();
    this.vocabulary = new Set();
    this.totalDocuments = 0;

    if (state) {
      this._restoreState(state);
    }

    this.trained = false;
  }

  static tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  _ensureLabelMaps(label) {
    if (!this.labelWordCounts.has(label)) {
      this.labelWordCounts.set(label, new Map());
      this.labelDocCounts.set(label, 0);
      this.labelTokenTotals.set(label, 0);
    }
  }

  train(samples) {
    if (!Array.isArray(samples) || samples.length === 0) {
      throw new Error("samples must be a non-empty array of { text, label } objects.");
    }

    for (const sample of samples) {
      if (!sample || typeof sample.text !== "string" || sample.text.trim().length === 0) {
        throw new Error("Each sample must include a non-empty text string.");
      }

      const label = assertValidLabel(sample.label);
      this._ensureLabelMaps(label);
      this.labelDocCounts.set(label, this.labelDocCounts.get(label) + 1);
      this.totalDocuments += 1;

      const tokens = TextCategoryAI.tokenize(sample.text);
      const wordCounts = this.labelWordCounts.get(label);

      for (const token of tokens) {
        this.vocabulary.add(token);
        wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
        this.labelTokenTotals.set(label, this.labelTokenTotals.get(label) + 1);
      }
    }

    if (this.labelDocCounts.size === 0) {
      throw new Error("No valid labeled samples were provided.");
    }

    this.trained = true;
  }

  predict(text) {
    if (!this.trained) {
      throw new Error("Model is not trained or loaded. Train first or load a saved model.");
    }

    if (typeof text !== "string" || text.trim().length === 0) {
      throw new Error("text must be a non-empty string.");
    }

    const tokens = TextCategoryAI.tokenize(text);
    const labels = [...this.labelDocCounts.keys()];

    if (labels.length === 0) {
      throw new Error("Model has no learned labels. Train with data first.");
    }

    const vocabSize = Math.max(this.vocabulary.size, 1);

    let bestLabel = labels[0];
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const label of labels) {
      const docCount = this.labelDocCounts.get(label);
      const prior = Math.log(docCount / this.totalDocuments);
      const tokenTotal = this.labelTokenTotals.get(label) || 0;
      const wordCounts = this.labelWordCounts.get(label);
      let score = prior;

      for (const token of tokens) {
        const count = wordCounts.get(token) || 0;
        const likelihood = (count + 1) / (tokenTotal + vocabSize);
        score += Math.log(likelihood);
      }

      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }

    return assertValidLabel(bestLabel);
  }

  _serializeState() {
    return {
      labelDocCounts: Object.fromEntries(this.labelDocCounts),
      labelTokenTotals: Object.fromEntries(this.labelTokenTotals),
      labelWordCounts: Object.fromEntries(
        [...this.labelWordCounts.entries()].map(([label, counts]) => [
          String(label),
          Object.fromEntries(counts),
        ])
      ),
      vocabulary: [...this.vocabulary],
      totalDocuments: this.totalDocuments,
    };
  }

  _restoreState(state) {
    this.labelDocCounts = new Map(
      Object.entries(state.labelDocCounts || {}).map(([label, count]) => [Number(label), Number(count)])
    );
    this.labelTokenTotals = new Map(
      Object.entries(state.labelTokenTotals || {}).map(([label, count]) => [Number(label), Number(count)])
    );
    this.labelWordCounts = new Map(
      Object.entries(state.labelWordCounts || {}).map(([label, counts]) => [
        Number(label),
        new Map(Object.entries(counts || {})),
      ])
    );
    this.vocabulary = new Set(state.vocabulary || []);
    this.totalDocuments = Number(state.totalDocuments || 0);
  }

  async saveModel(modelPath) {
    if (!this.trained) {
      throw new Error("Model is not trained or loaded. Nothing to save.");
    }

    const fullPath = path.resolve(modelPath);
    const dir = path.dirname(fullPath);

    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const state = this._serializeState();
    fs.writeFileSync(fullPath, JSON.stringify(state, null, 2), "utf8");
    return fullPath;
  }

  static async loadModel(modelPath) {
    const fullPath = path.resolve(modelPath);
    try {
      const raw = fs.readFileSync(fullPath, "utf8");
      const state = JSON.parse(raw);
      //if state is not valid that means the model is empty so create a new instance of TextCategoryAI
      if (!state || !state.labelDocCounts || Object.keys(state.labelDocCounts).length === 0) {
        return new TextCategoryAI();
      }
      const ai = new TextCategoryAI(state);
      ai.trained = true;
      return ai;
    } catch (err) {
      // If any error occurs (e.g., file not found or invalid JSON), return a new instance
      return new TextCategoryAI();
    }
  }

  static loadSamplesFromFile(filePath) {
    const fullPath = path.resolve(filePath);
    const raw = fs.readFileSync(fullPath, "utf8");
    const parsed = JSON.parse(raw);

    if (!Array.isArray(parsed)) {
      throw new Error("Dataset file must contain a JSON array of { text, label } objects.");
    }

    return parsed;
  }
}

module.exports = {
  TextCategoryAI,
  MIN_LABEL,
  MAX_LABEL,
  assertValidLabel,
};
