const fs = require("fs");
const path = require("path");
const admin = require("firebase-admin");
const {FieldPath} = require("firebase-admin/firestore");
const { TextCategoryAI } = require("./classifier");

const TRANSACTION_LABEL_BY_TYPE = {
  misc: 0,
  fuel: 1,
  "bank fee": 2,
  maintenance: 3,
  wages: 4,
  insurance: 5,
  registration: 6,
  "equipment lease": 7,
  "home office": 8,
  utility: 9,
  "license or permit": 10,
  shareholder: 11,
  sales: 12,
  interest: 13,
  taxes: 14,
  travel: 15,
  "professional fees": 16,
};
const TRANSACTION_TYPE_BY_LABEL = Object.fromEntries(
  Object.entries(TRANSACTION_LABEL_BY_TYPE).map(([type, label]) => [label, type])
);

const ACCOUNT_MODEL_PATH = path.join(__dirname, "..", "models", "account-model.json");
const SERVICE_ACCOUNT_PATH = path.join(__dirname, "..", "gridtow-firebase-adminsdk-tmexq-59ab2e1150.json");

function ensureFirebaseApp() {
  if (admin.apps.length > 0) {
    return admin.app();
  }

  if (!fs.existsSync(SERVICE_ACCOUNT_PATH)) {
    throw new Error(`Missing Firebase service account file: ${SERVICE_ACCOUNT_PATH}`);
  }

  const serviceAccount = require(SERVICE_ACCOUNT_PATH);

  return admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
}

function toDate(value) {
  if (!value) {
    return null;
  }

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  if (typeof value === "number") {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }

    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      const milliseconds = numeric < 1e12 ? numeric * 1000 : numeric;
      const date = new Date(milliseconds);
      return Number.isNaN(date.getTime()) ? null : date;
    }

    const date = new Date(trimmed);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  if (typeof value.toDate === "function") {
    const date = value.toDate();
    return Number.isNaN(date.getTime()) ? null : date;
  }

  if (typeof value.seconds === "number") {
    const milliseconds = value.seconds * 1000 + Math.floor((value.nanoseconds || 0) / 1e6);
    const date = new Date(milliseconds);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  return null;
}

function getTransactionDate(doc) {
  const data = doc.data() || {};
  return (
    toDate(doc.id) ||
    toDate(data.timestamp) ||
    toDate(data.date) ||
    toDate(data.createdAt) ||
    toDate(data.time)
  );
}

function getLabelFromType(type) {
  if (typeof type !== "string") {
    return null;
  }

  const normalizedType = type.trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(TRANSACTION_LABEL_BY_TYPE, normalizedType)
    ? TRANSACTION_LABEL_BY_TYPE[normalizedType]
    : null;
}

function buildTrainingText(transaction) {
  const pieces = [transaction.desc, `amount ${transaction.amount}`].filter((value) => value !== undefined && value !== null && String(value).trim().length > 0);
  return pieces.join(" ");
}

function documentIdFromDate(date) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    throw new Error("Invalid date provided to documentIdFromDate.");
  }
  return date.getTime().toString();
}
async function loadMonth(month, year) {
  ensureFirebaseApp();
  const db = admin.firestore();
  let accountRef = db.collection("account").orderBy(FieldPath.documentId());
  const start = new Date(Date.UTC(year, month - 1, 1, 0, 0, 0, 0)); //convert month to 0-based index for Date.UTC
  const end = new Date(Date.UTC(year, month, 1, 0, 0, 0, 0)); //convert month to 0-based index for Date.UTC
  const snapshot = await accountRef.startAt(documentIdFromDate(start)).endBefore(documentIdFromDate(end)).get();

  const samples = [];
  let skipped = 0;

  snapshot.forEach((doc) => {
    const data = doc.data() || {};
    const transactionDate = getTransactionDate(doc);
    const label = getLabelFromType(data.type);

    if (!transactionDate || transactionDate < start || transactionDate >= end || label === null) {
      skipped += 1;
      return;
    }

    const text = buildTrainingText(data);
    if (!text) {
      skipped += 1;
      return;
    }

    samples.push({
      text,
      label,
    });
  });

  return { samples, skipped };
}

async function main() {
  const { samples, skipped } = await loadMonth(12, 2025);

  if (samples.length === 0) {
    throw new Error("No December 2025 transactions were found in the 'account' collection.");
  }

  const ai = new TextCategoryAI();
  ai.train(samples);

  await ai.saveModel(ACCOUNT_MODEL_PATH);

  const loaded = await TextCategoryAI.loadModel(ACCOUNT_MODEL_PATH);
  const exampleParagraph = "etransfer sent";
  const predictedLabel = loaded.predict(exampleParagraph);

  console.log(`Loaded ${samples.length} December 2025 transactions from Firestore.`);
  console.log(`Skipped ${skipped} transactions that were outside the month or had an unmapped type.`);
  console.log(`Saved account model to: ${ACCOUNT_MODEL_PATH}`);
  console.log(`Example input: ${exampleParagraph}`);
  console.log(`Predicted label: ${predictedLabel} (${TRANSACTION_TYPE_BY_LABEL[predictedLabel] || "unknown"})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
