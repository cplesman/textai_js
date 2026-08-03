# textai_js

A Node.js text classifier that predicts an integer label in the range 0 to 63 (inclusive) from a paragraph.

Features:
- Trains on paragraph and label pairs
- Enforces labels between 0 and 63
- Saves model to disk
- Loads saved model and predicts
- Hosts an Express API for train/save/load/evaluate and batch operations
- Includes CLI, demo, and tests

## Install

```bash
npm install
```

## Dataset Format

Use a JSON array of objects:

```json
[
  { "text": "some paragraph", "label": 0 },
  { "text": "another paragraph", "label": 63 }
]
```

Each label must be an integer from 0 to 63.

## Train and Save a Model

```bash
npm run train -- --data data/sample-dataset.json --model models/my-model.json
```

## Load and Predict

```bash
npm run predict -- --model models/my-model.json --text "This paragraph is about market earnings and investors."
```

The output is a single integer label.

## Demo

```bash
npm run demo
```

The demo reads December 2025 transactions from the `account` collection in Firestore using `gridtow-firebase-adminsdk-tmexq-59ab2e1150.json`, trains an account model, saves it to `models/account-model.json`, and reloads it.

## Tests

```bash
npm test
```

## API Server (Port 3200)

Start server:

```bash
npm run server
```

Base URL:

```text
http://localhost:3200
```

### Endpoints

- `GET /health`
- `POST /train`
- `POST /batch-train`
- `POST /evaluate`
- `POST /batch-evaluate`
- `POST /save`
- `POST /load`

### Example Request Bodies

`POST /train`

```json
{
  "reset": true,
  "samples": [
    { "text": "sports football team won", "label": 3 },
    { "text": "stock market gained", "label": 11 }
  ]
}
```

`POST /batch-train`

```json
{
  "reset": true,
  "batches": [
    [
      { "text": "sports football team won", "label": 3 }
    ],
    [
      { "text": "stock market gained", "label": 11 }
    ]
  ]
}
```

`POST /evaluate`

```json
{
  "samples": [
    { "text": "team won the final", "label": 3 },
    { "text": "investors liked earnings", "label": 11 }
  ]
}
```

`POST /batch-evaluate`

```json
{
  "batches": [
    {
      "name": "set-a",
      "samples": [
        { "text": "team won the final", "label": 3 }
      ]
    },
    {
      "name": "set-b",
      "samples": [
        { "text": "investors liked earnings", "label": 11 }
      ]
    }
  ]
}
```

`POST /save`

```json
{
  "modelPath": "models/api-model.json"
}
```

`POST /load`

```json
{
  "modelPath": "models/api-model.json"
}
```

## Programmatic Usage

```js
const { TextCategoryAI } = require("./src/classifier");

async function run() {
  const ai = new TextCategoryAI();

  ai.train([
    { text: "sports news and match summary", label: 3 },
    { text: "quarterly revenue and market outlook", label: 11 }
  ]);

  await ai.saveModel("models/model.json");

  const loaded = await TextCategoryAI.loadModel("models/model.json");
  const label = loaded.predict("The team won the game in overtime.");
  console.log(label); // integer in [0, 63]
}

run();
```
