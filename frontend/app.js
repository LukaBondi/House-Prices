const API = window.API_BASE || (
  // When served from docker-compose (port 3000), nginx proxies /api -> backend.
  window.location.port === "3000" ? `${window.location.origin}/api` : "http://localhost:8000"
);

const form = document.getElementById("predict-form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const salePriceEl = document.getElementById("sale-price");
const loadSampleBtn = document.getElementById("load-sample");
const clearBtn = document.getElementById("clear-form");

const SAMPLE = {
  OverallQual: 7,
  OverallCond: 5,
  GrLivArea: 1710,
  LotArea: 8450,
  YearBuilt: 2003,
  TotalBsmtSF: 856,
  GarageCars: 2,
  FullBath: 2,
  Neighborhood: "CollgCr",
  MSZoning: "RL",
  KitchenQual: "Gd",
  ExterQual: "Gd",
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#9b2c2c" : "";
}

function collectFeatures() {
  const data = new FormData(form);
  const features = {};
  const numericFields = new Set([
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "LotArea",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
  ]);
  for (const [key, value] of data.entries()) {
    if (value === "") continue;
    features[key] = numericFields.has(key) ? Number(value) : value;
  }
  return features;
}

function fillForm(values) {
  Object.entries(values).forEach(([key, value]) => {
    const el = form.elements.namedItem(key);
    if (el) el.value = value;
  });
}

loadSampleBtn.addEventListener("click", async () => {
  setStatus("Loading sample from API schema…");
  try {
    const res = await fetch(`${API}/schema`);
    if (!res.ok) throw new Error(`Schema request failed (${res.status})`);
    const schema = await res.json();
    const sample = schema.sample || SAMPLE;
    const subset = {};
    Object.keys(SAMPLE).forEach((key) => {
      subset[key] =
        sample[key] !== undefined && sample[key] !== null ? sample[key] : SAMPLE[key];
    });
    fillForm(subset);
    setStatus("Sample house loaded.");
  } catch (err) {
    fillForm(SAMPLE);
    setStatus(`API schema unavailable; used local sample. (${err.message})`, true);
  }
});

clearBtn.addEventListener("click", () => {
  form.reset();
  resultEl.classList.add("hidden");
  setStatus("Form cleared.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultEl.classList.add("hidden");
  setStatus("Requesting prediction…");

  const features = collectFeatures();
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      throw new Error(detail || `Predict failed (${res.status})`);
    }
    salePriceEl.textContent = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(body.sale_price);
    resultEl.classList.remove("hidden");
    setStatus("Prediction received.");
  } catch (err) {
    setStatus(err.message || "Prediction failed.", true);
  }
});
