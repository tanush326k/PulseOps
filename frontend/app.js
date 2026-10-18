/**
 * PulseOps frontend — app.js
 *
 * Handles: incident selection, running the pipeline, rendering results.
 * Uses fetch() to the PulseOps backend API (same origin when served via FastAPI).
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? ''   // same origin
  : '';  // still same origin (FastAPI serves both)

// ─── DOM refs ───────────────────────────────────────────────────────────────

const $grid       = document.getElementById('incident-grid');
const $runBtn     = document.getElementById('run-btn');
const $results    = document.getElementById('results');
const $steps      = document.getElementById('progress-steps');
const $errBanner  = document.getElementById('error-banner');
const $statusDot  = document.getElementById('status-dot');
const $statusText = document.getElementById('status-text');

let selectedId = null;

// ─── Utility ─────────────────────────────────────────────────────────────────

function showError(msg) {
  $errBanner.textContent = msg;
  $errBanner.classList.add('visible');
}
function clearError() {
  $errBanner.classList.remove('visible');
}

function setStep(id, state) { // state: 'pending'|'active'|'done'
  const el = document.getElementById(id);
  el.className = 'step ' + state;
}

// ─── Health check ────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch(API_BASE + '/api/health');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.decider && data.decider.reachable) {
      $statusDot.className  = 'status-dot ok';
      $statusText.innerHTML = `Strands Decider 2B <strong>online</strong> · checkpoint: <code>${data.decider.checkpoint || 'loaded'}</code>`;
    } else {
      $statusDot.className  = 'status-dot warn';
      $statusText.innerHTML = '<strong>Decider server not reachable</strong> — run: <code>strands-decider serve &lt;checkpoint&gt; --port 8099</code>';
    }
  } catch (e) {
    $statusDot.className  = 'status-dot error';
    $statusText.innerHTML = '<strong>Backend not reachable</strong> — is the FastAPI server running?';
  }
}

// ─── Incidents ───────────────────────────────────────────────────────────────

async function loadIncidents() {
  try {
    const res = await fetch(API_BASE + '/api/incidents');
    const data = await res.json();
    renderIncidentGrid(data.incidents);
  } catch (e) {
    showError('Could not load incident list from backend: ' + e.message);
  }
}

function renderIncidentGrid(incidents) {
  $grid.innerHTML = '';
  incidents.forEach(inc => {
    const card = document.createElement('div');
    card.className = 'incident-card';
    card.dataset.id = inc.id;
    card.innerHTML = `
      <span class="sev-badge sev-${inc.severity}">${inc.severity}</span>
      <div class="card-info">
        <div class="card-title">${esc(inc.title)}</div>
        <div class="card-meta">${esc(inc.id)} · ${esc(inc.service)}</div>
      </div>`;
    card.addEventListener('click', () => selectIncident(inc.id, card));
    $grid.appendChild(card);
  });
}

function selectIncident(id, cardEl) {
  document.querySelectorAll('.incident-card').forEach(c => c.classList.remove('selected'));
  cardEl.classList.add('selected');
  selectedId = id;
  $runBtn.disabled = false;
  $results.classList.remove('visible');
  clearError();
}

// ─── Run pipeline ────────────────────────────────────────────────────────────

$runBtn.addEventListener('click', runIncident);

async function runIncident() {
  if (!selectedId) return;
  clearError();
  $results.classList.remove('visible');
  $steps.classList.add('visible');
  $runBtn.classList.add('loading');
  $runBtn.disabled = true;

  setStep('step-decider', 'active');
  setStep('step-tool',    'pending');
  setStep('step-bedrock', 'pending');

  try {
    const res = await fetch(API_BASE + '/api/run-incident', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ incident_id: selectedId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    setStep('step-decider', 'done');
    setStep('step-tool', 'active');

    const data = await res.json();

    setStep('step-tool', 'done');
    setStep('step-bedrock', data.assessment.available ? 'done' : 'done');

    renderResults(data);
    $results.classList.add('visible');

  } catch (e) {
    showError('Pipeline error: ' + e.message);
    setStep('step-decider', 'pending');
    setStep('step-tool',    'pending');
    setStep('step-bedrock', 'pending');
  } finally {
    $runBtn.classList.remove('loading');
    $runBtn.disabled = false;
    $steps.classList.remove('visible');
  }
}

// ─── Render results ───────────────────────────────────────────────────────────

function renderResults(data) {
  renderDecider(data.decider);
  renderIncidentSummary(data.incident);
  renderEvidence(data.tool_result);
  renderAssessment(data.assessment);
}

function renderDecider(d) {
  document.getElementById('action-name').textContent = d.action;
  document.getElementById('decision-conf').innerHTML =
    `Confidence: <strong>${(d.confidence * 100).toFixed(1)}%</strong> &nbsp;·&nbsp; ` +
    (d.latency_ms ? `Decider latency: <strong>${d.latency_ms.toFixed(0)} ms</strong>` : '');

  const container = document.getElementById('prob-bars');
  container.innerHTML = '';

  // sort by probability descending
  const entries = Object.entries(d.probabilities).sort((a,b) => b[1] - a[1]);

  entries.forEach(([name, prob]) => {
    const isWinner = name === d.action;
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'prob-row' + (isWinner ? ' winner' : '');
    row.innerHTML = `
      <span class="prob-label" title="${esc(name)}">${esc(name)}</span>
      <div class="prob-track"><div class="prob-fill${isWinner ? ' winner' : ''}" style="width:0%" data-pct="${pct}"></div></div>
      <span class="prob-val">${pct}%</span>`;
    container.appendChild(row);
  });

  // animate bars after a small delay
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.prob-fill').forEach(el => {
        el.style.width = el.dataset.pct + '%';
      });
    }, 80);
  });

  if (d.latency_ms) {
    document.getElementById('latency-badge').textContent = `⏱ ${d.latency_ms.toFixed(0)} ms`;
  }
}

function renderIncidentSummary(inc) {
  const sev = inc.severity;
  let html = `
    <div style="margin-bottom:14px">
      <span class="sev-badge sev-${sev}" style="display:inline-block;margin-bottom:8px">${sev}</span>
      <div style="font-weight:700;font-size:1rem;margin-bottom:6px">${esc(inc.title)}</div>
      <div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:10px">${esc(inc.id)} · ${esc(inc.service)}</div>
      <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6">${esc(inc.description)}</div>
    </div>`;
  if (inc.signals && inc.signals.length) {
    html += '<div style="margin-top:14px"><div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-bottom:8px">Observed signals</div>';
    html += inc.signals.map(s =>
      `<div style="font-family:'JetBrains Mono',monospace;font-size:0.77rem;color:var(--text-secondary);padding:3px 0;border-bottom:1px solid var(--border)">→ ${esc(s)}</div>`
    ).join('');
    html += '</div>';
  }
  document.getElementById('incident-summary').innerHTML = html;
}

function renderEvidence(tool) {
  document.getElementById('evidence-text').textContent = tool.evidence;
  document.getElementById('key-finding').innerHTML =
    `<strong>Key finding:</strong> ${esc(tool.key_finding)}`;
}

function renderAssessment(assessment) {
  const badge = document.getElementById('model-badge');
  const text  = document.getElementById('assessment-text');
  if (assessment.available) {
    badge.className = 'model-badge';
    badge.textContent = '🤖 ' + (assessment.model || 'Bedrock');
  } else {
    badge.className = 'model-badge unavailable';
    badge.textContent = 'Bedrock unavailable';
  }
  text.textContent = assessment.assessment;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Init ─────────────────────────────────────────────────────────────────────

(async function init() {
  await checkHealth();
  await loadIncidents();
})();
