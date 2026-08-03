const path = require("path");
const { TextCategoryAI } = require("./classifier");

async function main() {
  const samples = [
    { text: "sports football team won the championship", label: 3 },
    { text: "goalkeeper saved a penalty in the final match", label: 3 },
    { text: "stock market investors expect lower inflation", label: 11 },
    { text: "quarterly earnings and balance sheet improved", label: 11 },
    { text: "new javascript framework improves frontend build speed", label: 42 },
    { text: "runtime performance benchmarks for node applications", label: 42 },
    { text: "hospital research discovered a new treatment", label: 55 },
    { text: "clinical trial reports better patient outcomes", label: 55 },
  ];

  const ai = new TextCategoryAI();
  ai.train(samples);

  const modelPath = path.join(__dirname, "..", "models", "demo-model.json");
  await ai.saveModel(modelPath);

  const loaded = await TextCategoryAI.loadModel(modelPath);
  const paragraph = "The company released strong earnings and investors reacted positively.";
  const label = loaded.predict(paragraph);

  console.log(`Input: ${paragraph}`);
  console.log(`Predicted label: ${label}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
