const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const { TextCategoryAI } = require("../src/classifier");

test("train, save, load, and predict", async () => {
  const samples = [
    { text: "apple banana fruit", label: 1 },
    { text: "orange pear fruit", label: 1 },
    { text: "car truck vehicle", label: 2 },
    { text: "bus train transport", label: 2 },
  ];

  const ai = new TextCategoryAI();
  ai.train(samples);

  const modelPath = path.join(__dirname, "tmp-model.json");
  await ai.saveModel(modelPath);

  const loaded = await TextCategoryAI.loadModel(modelPath);
  const prediction = loaded.predict("banana and orange smoothie");

  assert.equal(typeof prediction, "number");
  assert.ok(prediction >= 0 && prediction <= 63);

  if (fs.existsSync(modelPath)) {
    fs.unlinkSync(modelPath);
  }
});

test("throws for invalid labels", () => {
  const ai = new TextCategoryAI();

  assert.throws(() => {
    ai.train([{ text: "some text", label: 99 }]);
  }, /Label must be an integer between 0 and 63/);
});
