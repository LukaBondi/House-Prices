"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "..", "frontend", "dashboard.js"), "utf8");

class Element {
  constructor(id = "") {
    this.id = id;
    this.textContent = "";
    this.hidden = false;
    this.disabled = false;
    this.value = "15";
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.listeners = {};
    this.dateTime = "";
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  replaceChildren(...children) {
    this.children = children;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }
}

function makeSummary(overrides = {}) {
  const summary = {
    as_of: 1200,
    window: { minutes: 15, start: 300, end: 1200 },
    requests: {
      total: 0,
      completed: 0,
      pending: 0,
      aborted: 0,
      logging_failed: 0,
      rate_per_minute: null,
      distinct_inputs: 0,
    },
    latency: { handler_p50_ms: null, handler_p95_ms: null },
    errors: { "4xx": { count: 0, rate: null }, "5xx": { count: 0, rate: null } },
    cache: { hits: 0, misses: 0, coalesced: 0 },
    logging: { enabled: true, healthy: true, failures: 0 },
    model: { version: "bootstrap-test" },
    labels: { count: 0, distinct_observations: 0 },
    retraining: { last_run: null },
    drift: {
      status: "insufficient_data",
      reference_id: "reference-test",
      sample_count: 0,
      distinct_inputs: 0,
      minimum_samples: 100,
      threshold: 0.1,
      features: {},
    },
  };
  return Object.assign(summary, overrides);
}

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

function createDashboard(fetchImpl, { fastTimeout = false } = {}) {
  const elements = new Map();
  for (const id of [
    "window-minutes", "refresh-dashboard", "dashboard-error", "refresh-status",
    "monitoring-panels", "api-health", "request-total", "request-breakdown",
    "traffic-empty", "request-rate", "latency-p50", "latency-p95", "error-4xx",
    "error-5xx", "cache-hits", "cache-breakdown", "logging-health", "logging-detail",
    "model-version", "label-count", "label-detail", "retraining-status", "retraining-detail",
    "drift-rows", "drift-status", "drift-samples", "drift-distinct", "drift-reference",
    "drift-threshold", "drift-empty", "summary-time", "summary-window",
    "summary-window-start", "summary-window-end",
  ]) elements.set(id, new Element(id));
  elements.get("request-total").textContent = "Unavailable";
  elements.get("retraining-status").textContent = "Unavailable";
  elements.get("retraining-detail").textContent = "Waiting for monitoring data.";
  const document = {
    getElementById: (id) => elements.get(id),
    createElement: (tag) => new Element(tag),
  };
  const window = {
    API_BASE: "http://api.test",
    setTimeout: (callback, delay) => globalThis.setTimeout(callback, fastTimeout && delay === 3000 ? 10 : delay),
    clearTimeout: (timer) => globalThis.clearTimeout(timer),
    setInterval: () => 1,
  };
  vm.runInNewContext(script, { window, document, fetch: fetchImpl, AbortController, Intl, Date, Number, Math, Object, Set, Error });
  return { elements, window };
}

async function until(predicate, message) {
  const deadline = Date.now() + 1000;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  assert.fail(message);
}

test("renders explicit null run, logging failures, nested gate metrics, reason, and UTC bounds", async () => {
  const summary = makeSummary({
    requests: {
      total: 1, completed: 1, pending: 0, aborted: 0, logging_failed: 1,
      rate_per_minute: 0.07, distinct_inputs: 1,
    },
    errors: { "4xx": { count: 0, rate: 0 }, "5xx": { count: 1, rate: 1 } },
    retraining: {
      last_run: {
        run_id: "run-1",
        parent_version: "bootstrap-test",
        status: "rejected",
        completed_at: 1190,
        details: { candidate_version: "candidate-1", reason: "rmsle_gate" },
        metrics: { champion: { rmsle: 0.2 }, challenger: { rmsle: 0.21 } },
      },
    },
  });
  const dashboard = createDashboard(async (url) => url.endsWith("/health")
    ? response({ ready: true, model_loaded: true })
    : response(summary));
  const elements = dashboard.elements;
  await until(() => elements.get("refresh-status").textContent.startsWith("Updated"), "dashboard did not complete its first refresh");

  assert.match(elements.get("request-breakdown").textContent, /Logging failed 1 \(included above\)/);
  assert.match(elements.get("retraining-detail").textContent, /Candidate candidate-1/);
  assert.match(elements.get("retraining-detail").textContent, /RMSLE 0\.2000 → 0\.2100/);
  assert.match(elements.get("retraining-detail").textContent, /Reason rmsle_gate/);
  assert.equal(elements.get("summary-window-start").textContent, "1970-01-01 00:05:00 UTC");
  assert.equal(elements.get("summary-window-end").textContent, "1970-01-01 00:20:00 UTC");

  const noRunDashboard = createDashboard(async (url) => url.endsWith("/health")
    ? response({ ready: true })
    : response(makeSummary()));
  await until(() => noRunDashboard.elements.get("refresh-status").textContent.startsWith("Updated"), "explicit null run was not rendered");
  assert.equal(noRunDashboard.elements.get("retraining-status").textContent, "No run recorded");
  assert.equal(noRunDashboard.elements.get("latency-p50").textContent, "Unavailable p50");
});

test("rejects empty monitoring envelopes and leaves refresh in an error state", async () => {
  const dashboard = createDashboard(async (url) => url.endsWith("/health")
    ? response({ ready: true })
    : response({}));
  await until(() => !dashboard.elements.get("dashboard-error").hidden, "invalid empty envelope did not raise a visible error");
  assert.match(dashboard.elements.get("dashboard-error").textContent, /Monitoring API returned an invalid/);
  assert.equal(dashboard.elements.get("refresh-status").textContent, "Monitoring data unavailable");
  assert.equal(dashboard.elements.get("request-total").textContent, "Unavailable");

  const missingRun = createDashboard(async (url) => url.endsWith("/health")
    ? response({ ready: true })
    : response(makeSummary({ retraining: {} })));
  await until(() => !missingRun.elements.get("dashboard-error").hidden, "missing retraining result was treated as a successful refresh");
  assert.equal(missingRun.elements.get("retraining-status").textContent, "Unavailable");
  assert.notEqual(missingRun.elements.get("retraining-status").textContent, "No run recorded");
});

test("queues a fresh generation after a window flip away and back while an old poll is pending", async () => {
  let releaseFirstSummary;
  const firstSummary = new Promise((resolve) => { releaseFirstSummary = resolve; });
  const summaryUrls = [];
  let summaryCalls = 0;
  const dashboard = createDashboard(async (url) => {
    if (url.endsWith("/health")) return response({ ready: true, model_loaded: true });
    summaryUrls.push(url);
    summaryCalls += 1;
    if (summaryCalls === 1) return firstSummary;
    return response(makeSummary({
      requests: {
        total: 9, completed: 9, pending: 0, aborted: 0, logging_failed: 0,
        rate_per_minute: 0.6, distinct_inputs: 9,
      },
    }));
  });
  await until(() => summaryCalls === 1, "initial summary request did not start");
  const select = dashboard.elements.get("window-minutes");
  select.value = "60";
  select.listeners.change();
  select.value = "15";
  select.listeners.change();
  releaseFirstSummary(response(makeSummary({
    requests: {
      total: 1, completed: 1, pending: 0, aborted: 0, logging_failed: 0,
      rate_per_minute: 0.07, distinct_inputs: 1,
    },
  })));

  await until(() => summaryCalls === 2 && dashboard.elements.get("request-total").textContent === "9", "latest window generation was not fetched and rendered");
  assert.deepEqual(summaryUrls, [
    "http://api.test/monitoring/summary?window_minutes=15",
    "http://api.test/monitoring/summary?window_minutes=15",
  ]);
  assert.equal(dashboard.elements.get("dashboard-error").hidden, true);
});

test("suppresses an old-window failure and refreshes the latest selected window", async () => {
  let failFirstSummary;
  const firstSummary = new Promise((_resolve, reject) => { failFirstSummary = reject; });
  let summaryCalls = 0;
  const dashboard = createDashboard(async (url) => {
    if (url.endsWith("/health")) return response({ ready: true });
    summaryCalls += 1;
    if (summaryCalls === 1) return firstSummary;
    return response(makeSummary());
  });
  await until(() => summaryCalls === 1, "initial summary request did not start");
  const select = dashboard.elements.get("window-minutes");
  select.value = "60";
  select.listeners.change();
  select.value = "15";
  select.listeners.change();
  failFirstSummary(new Error("stale window request failed"));

  await until(() => summaryCalls === 2 && dashboard.elements.get("refresh-status").textContent.startsWith("Updated"), "queued latest-window refresh did not complete");
  assert.equal(dashboard.elements.get("dashboard-error").hidden, true);
  assert.equal(dashboard.elements.get("request-total").textContent, "0");
});

test("bounds a stalled monitoring fetch and reports the timeout", async () => {
  let summaryStarted = false;
  const dashboard = createDashboard((url, options) => {
    if (url.endsWith("/health")) return Promise.resolve(response({ ready: true }));
    summaryStarted = true;
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
    });
  }, { fastTimeout: true });

  await until(() => summaryStarted && !dashboard.elements.get("dashboard-error").hidden, "stalled monitoring fetch was not bounded");
  assert.match(dashboard.elements.get("dashboard-error").textContent, /timed out/);
});
