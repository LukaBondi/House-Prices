"use strict";

(function () {
  const API = window.API_BASE || "http://localhost:8000";
  const REFRESH_MS = 5000;
  const REQUEST_TIMEOUT_MS = 3000;
  const allowedWindows = new Set([15, 60, 1440]);
  const windowSelect = document.getElementById("window-minutes");
  const refreshButton = document.getElementById("refresh-dashboard");
  const errorBox = document.getElementById("dashboard-error");
  const refreshStatus = document.getElementById("refresh-status");
  const panels = document.getElementById("monitoring-panels");
  let pollInProgress = false;
  let refreshQueued = false;
  let refreshGeneration = 0;
  let activeController = null;
  let lastSuccessfulRefresh = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function text(id, value) {
    byId(id).textContent = value;
  }

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(value, key);
  }

  function count(value) {
    return Number.isFinite(value) ? new Intl.NumberFormat("en-US").format(value) : "Unavailable";
  }

  function metric(value, suffix) {
    return Number.isFinite(value) ? value.toFixed(2) + suffix : "Unavailable";
  }

  function percent(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(2) + "%" : "Unavailable";
  }

  function timestamp(value) {
    if (!Number.isFinite(value)) return "Unavailable";
    const date = new Date(value * 1000);
    return Number.isNaN(date.getTime()) ? "Unavailable" : date.toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC");
  }

  function requireRecord(value, label) {
    if (!isRecord(value)) throw new Error("Monitoring API returned an invalid " + label + " object.");
    return value;
  }

  function requireNumber(value, label, nullable) {
    if (nullable && value === null) return;
    if (!Number.isFinite(value)) throw new Error("Monitoring API returned an invalid " + label + ".");
  }

  function requireCount(value, label) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error("Monitoring API returned an invalid " + label + " count.");
  }

  function validateSummary(summary, requestedWindow) {
    requireRecord(summary, "summary");
    requireNumber(summary.as_of, "as_of");
    const win = requireRecord(summary.window, "window");
    requireNumber(win.minutes, "window.minutes");
    requireNumber(win.start, "window.start");
    requireNumber(win.end, "window.end");
    if (!allowedWindows.has(win.minutes) || win.minutes !== requestedWindow || win.start > win.end || win.end > summary.as_of) {
      throw new Error("Monitoring API returned window bounds that do not match the selected window.");
    }

    const requests = requireRecord(summary.requests, "requests");
    ["total", "completed", "pending", "aborted", "logging_failed"].forEach(function (key) {
      requireCount(requests[key], "requests." + key);
    });
    requireNumber(requests.rate_per_minute, "requests.rate_per_minute", true);

    const latency = requireRecord(summary.latency, "latency");
    requireNumber(latency.handler_p50_ms, "latency.handler_p50_ms", true);
    requireNumber(latency.handler_p95_ms, "latency.handler_p95_ms", true);
    const errors = requireRecord(summary.errors, "errors");
    ["4xx", "5xx"].forEach(function (key) {
      const group = requireRecord(errors[key], "errors." + key);
      requireCount(group.count, "errors." + key);
      requireNumber(group.rate, "errors." + key + ".rate", true);
    });
    const cache = requireRecord(summary.cache, "cache");
    ["hits", "misses", "coalesced"].forEach(function (key) {
      requireCount(cache[key], "cache." + key);
    });
    const logging = requireRecord(summary.logging, "logging");
    if (typeof logging.enabled !== "boolean" || typeof logging.healthy !== "boolean") {
      throw new Error("Monitoring API returned invalid logging health fields.");
    }
    requireCount(logging.failures, "logging.failures");
    const model = requireRecord(summary.model, "model");
    if (!hasOwn(model, "version") || (model.version !== null && typeof model.version !== "string")) {
      throw new Error("Monitoring API returned an invalid model version.");
    }
    const labels = requireRecord(summary.labels, "labels");
    requireCount(labels.count, "labels.count");
    requireCount(labels.distinct_observations, "labels.distinct_observations");
    const retraining = requireRecord(summary.retraining, "retraining");
    if (!hasOwn(retraining, "last_run") || (retraining.last_run !== null && !isRecord(retraining.last_run))) {
      throw new Error("Monitoring API returned an invalid retraining result.");
    }
    const drift = requireRecord(summary.drift, "drift");
    if (typeof drift.status !== "string") throw new Error("Monitoring API returned an invalid drift status.");
    requireCount(drift.sample_count, "drift.sample_count");
    requireCount(drift.distinct_inputs, "drift.distinct_inputs");
    requireNumber(drift.threshold, "drift.threshold", true);
    requireRecord(drift.features, "drift.features");
    return summary;
  }

  function updateErrors(errors) {
    const client = errors["4xx"];
    const server = errors["5xx"];
    text("error-4xx", "4xx: " + count(client.count) + " (" + percent(client.rate) + ")");
    text("error-5xx", "5xx: " + count(server.count) + " (" + percent(server.rate) + ")");
  }

  function renderRetraining(retraining) {
    if (!isRecord(retraining) || !hasOwn(retraining, "last_run")) {
      text("retraining-status", "Unavailable");
      text("retraining-detail", "The monitoring response did not include retraining status.");
      return;
    }
    const run = retraining.last_run;
    if (run === null) {
      text("retraining-status", "No run recorded");
      text("retraining-detail", "No retraining result is available yet.");
      return;
    }
    if (!isRecord(run)) {
      text("retraining-status", "Unavailable");
      text("retraining-detail", "The latest retraining result has an unsupported shape.");
      return;
    }

    const detailsRecord = isRecord(run.details) ? run.details : {};
    const metricsRecord = isRecord(run.metrics) ? run.metrics : {};
    const status = typeof run.status === "string" ? run.status : "Outcome unavailable";
    const details = [];
    if (run.run_id) details.push("Run " + run.run_id);
    if (run.completed_at !== undefined && run.completed_at !== null) details.push("Finished " + timestamp(run.completed_at));
    if (detailsRecord.candidate_version) details.push("Candidate " + detailsRecord.candidate_version);
    if (run.parent_version) details.push("Parent " + run.parent_version);
    const champion = isRecord(metricsRecord.champion) ? metricsRecord.champion.rmsle : undefined;
    const challenger = isRecord(metricsRecord.challenger) ? metricsRecord.challenger.rmsle : undefined;
    if (Number.isFinite(champion) && Number.isFinite(challenger)) {
      details.push("RMSLE " + champion.toFixed(4) + " → " + challenger.toFixed(4));
    }
    const reason = detailsRecord.reason || detailsRecord.error || metricsRecord.rejection_reason;
    if (typeof reason === "string" && reason) details.push("Reason " + reason);
    text("retraining-status", status);
    text("retraining-detail", details.length ? details.join(" · ") : "Latest run details are unavailable.");
  }

  function featureCount(feature, countKey, rateKey) {
    if (Number.isFinite(feature[countKey])) return count(feature[countKey]);
    if (Number.isFinite(feature[rateKey])) return percent(feature[rateKey]);
    return "Unavailable";
  }

  function renderDriftFeatures(features) {
    const body = byId("drift-rows");
    body.replaceChildren();
    const entries = Object.entries(features);
    entries.sort(function (left, right) {
      const a = left[1] && Number.isFinite(left[1].distance) ? left[1].distance : -1;
      const b = right[1] && Number.isFinite(right[1].distance) ? right[1].distance : -1;
      return b - a || left[0].localeCompare(right[0]);
    });
    const top = entries.slice(0, 10);
    if (!top.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.className = "table-placeholder";
      cell.textContent = "Per-feature drift is unavailable.";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    top.forEach(function (entry) {
      const name = entry[0];
      const feature = isRecord(entry[1]) ? entry[1] : {};
      const row = document.createElement("tr");
      const cells = [
        name,
        metric(feature.distance, ""),
        featureCount(feature, "observed_count", "observed_rate"),
        featureCount(feature, "missing_count", "missing_rate"),
        featureCount(feature, "unknown_count", "unknown_rate"),
      ];
      cells.forEach(function (value, index) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function render(summary) {
    const requests = summary.requests;
    const latency = summary.latency;
    const cache = summary.cache;
    const logging = summary.logging;
    const model = summary.model;
    const labels = summary.labels;
    const drift = summary.drift;

    text("request-total", count(requests.total));
    text("request-breakdown", "Completed " + count(requests.completed) + " · Logging failed " + count(requests.logging_failed) + " (included above) · Pending " + count(requests.pending) + " · Aborted " + count(requests.aborted));
    const trafficEmpty = byId("traffic-empty");
    trafficEmpty.hidden = requests.total !== 0;
    trafficEmpty.textContent = requests.total === 0 ? "No prediction requests in this window." : "";
    text("request-rate", Number.isFinite(requests.rate_per_minute) ? requests.rate_per_minute.toFixed(2) + " requests/minute" : "Request rate unavailable for this window");
    text("latency-p50", metric(latency.handler_p50_ms, " ms") + " p50");
    text("latency-p95", metric(latency.handler_p95_ms, " ms") + " p95");
    updateErrors(summary.errors);
    text("cache-hits", count(cache.hits) + " hits");
    text("cache-breakdown", count(cache.misses) + " misses · " + count(cache.coalesced) + " coalesced");
    text("logging-health", logging.healthy ? "Healthy" : "Degraded");
    text("logging-detail", (logging.enabled ? "Enabled" : "Disabled") + " · " + count(logging.failures) + " write failures");
    text("model-version", typeof model.version === "string" && model.version ? model.version : "Unavailable");
    text("label-count", count(labels.count));
    text("label-detail", count(labels.distinct_observations) + " distinct eligible observations");
    renderRetraining(summary.retraining);

    const state = drift.status;
    const sampleCount = count(drift.sample_count);
    const distinctInputs = count(drift.distinct_inputs);
    const minimum = Number.isFinite(drift.minimum_samples) ? drift.minimum_samples : undefined;
    const reference = typeof drift.reference_id === "string" && drift.reference_id ? drift.reference_id : "Unavailable";
    text("drift-status", state.replace(/_/g, " "));
    text("drift-samples", "Valid observations: " + sampleCount + (minimum === undefined ? "" : " / " + count(minimum) + " minimum"));
    text("drift-distinct", "Distinct inputs: " + distinctInputs);
    text("drift-reference", "Reference: " + reference);
    text("drift-threshold", "Advisory threshold: " + metric(drift.threshold, ""));
    const empty = byId("drift-empty");
    if (state === "insufficient_data") {
      const remaining = minimum !== undefined && Number.isFinite(drift.sample_count) ? Math.max(0, minimum - drift.sample_count) : undefined;
      empty.textContent = remaining === undefined
        ? "Insufficient valid prediction observations to display an advisory drift result."
        : "Insufficient data: " + count(remaining) + " more valid prediction observations are needed before an advisory drift result is shown.";
      empty.hidden = false;
    } else if (drift.sample_count === 0) {
      empty.textContent = "No valid prediction observations in this window.";
      empty.hidden = false;
    } else {
      empty.hidden = true;
      empty.textContent = "";
    }
    renderDriftFeatures(drift.features);

    const asOf = timestamp(summary.as_of);
    byId("summary-time").textContent = asOf;
    byId("summary-time").dateTime = new Date(summary.as_of * 1000).toISOString();
    const minutes = summary.window.minutes;
    text("summary-window", minutes === 1440 ? "24 hours" : minutes + " minutes");
    text("summary-window-start", timestamp(summary.window.start));
    text("summary-window-end", timestamp(summary.window.end));
    panels.setAttribute("aria-busy", "false");
    lastSuccessfulRefresh = new Date();
  }

  async function fetchWithTimeout(url, options, parentSignal) {
    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = window.setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
    const abortForParent = function () { controller.abort(); };
    if (parentSignal) {
      if (parentSignal.aborted) abortForParent();
      else parentSignal.addEventListener("abort", abortForParent, { once: true });
    }
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
      if (timedOut) throw new Error("Monitoring request timed out.");
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
      if (parentSignal) parentSignal.removeEventListener("abort", abortForParent);
    }
  }

  async function refreshHealth(signal, generation) {
    const indicator = byId("api-health");
    try {
      const response = await fetchWithTimeout(API + "/health", {
        headers: { Accept: "application/json" },
        cache: "no-store",
      }, signal);
      let body = {};
      try {
        body = await response.json();
      } catch (_error) {
        body = {};
      }
      if (generation !== refreshGeneration) return;
      const ready = response.ok && body.ready !== false && body.status !== "degraded";
      const modelLoaded = body.model_loaded === true;
      indicator.dataset.state = ready ? "ready" : "error";
      indicator.textContent = ready
        ? "API ready · " + (modelLoaded ? "model loaded" : "model status unavailable")
        : "API not ready (HTTP " + response.status + "). " + (body.status || "Readiness check failed.");
    } catch (error) {
      if (generation !== refreshGeneration) return;
      indicator.dataset.state = "error";
      indicator.textContent = "API health check failed: " + (error && error.message ? error.message : "service unavailable");
    }
  }

  async function drainRefreshQueue() {
    if (pollInProgress) return;
    pollInProgress = true;
    while (refreshQueued) {
      refreshQueued = false;
      const generation = refreshGeneration;
      const requestedWindow = Number(allowedWindows.has(Number(windowSelect.value)) ? windowSelect.value : "15");
      const controller = new AbortController();
      activeController = controller;
      refreshButton.disabled = true;
      panels.setAttribute("aria-busy", "true");
      errorBox.hidden = true;
      errorBox.textContent = "";
      await refreshHealth(controller.signal, generation);
      try {
        if (generation !== refreshGeneration) continue;
        const response = await fetchWithTimeout(API + "/monitoring/summary?window_minutes=" + requestedWindow, {
          headers: { Accept: "application/json" },
          cache: "no-store",
        }, controller.signal);
        if (!response.ok) throw new Error("Monitoring request failed (HTTP " + response.status + ").");
        const summary = validateSummary(await response.json(), requestedWindow);
        if (generation !== refreshGeneration) continue;
        if (requestedWindow !== Number(windowSelect.value)) {
          refreshGeneration += 1;
          refreshQueued = true;
          continue;
        }
        render(summary);
        const clockText = new Intl.DateTimeFormat(undefined, { timeStyle: "medium" }).format(lastSuccessfulRefresh);
        refreshStatus.textContent = "Updated " + clockText + " · next refresh in five seconds";
      } catch (error) {
        if (generation !== refreshGeneration) continue;
        const reason = error && error.message ? error.message : "Monitoring API is unavailable.";
        errorBox.textContent = reason + (lastSuccessfulRefresh ? " Last successful update: " + lastSuccessfulRefresh.toLocaleTimeString() + "." : "");
        errorBox.hidden = false;
        text("retraining-status", "Unavailable");
        text("retraining-detail", "The monitoring refresh did not provide a verified retraining result.");
        refreshStatus.textContent = "Monitoring data unavailable";
        panels.setAttribute("aria-busy", "false");
      } finally {
        if (activeController === controller) activeController = null;
      }
    }
    pollInProgress = false;
    refreshButton.disabled = false;
  }

  function requestRefresh(forceCurrent = true) {
    refreshQueued = true;
    if (forceCurrent) {
      refreshGeneration += 1;
      if (activeController) activeController.abort();
    } else if (pollInProgress) {
      return;
    } else {
      refreshGeneration += 1;
    }
    void drainRefreshQueue();
  }

  windowSelect.addEventListener("change", function () { requestRefresh(true); });
  refreshButton.addEventListener("click", function () { requestRefresh(true); });
  requestRefresh(true);
  window.setInterval(function () { requestRefresh(false); }, REFRESH_MS);
}());
