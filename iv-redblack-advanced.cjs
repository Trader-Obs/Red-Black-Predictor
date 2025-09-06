// iv-redblack-advanced.cjs
// Enhanced Red/Black/Green predictor for SportyBet (CommonJS, Node.js).
// - Data collection (JSON), statistical tests, softmax logistic regression,
//   Markov model with recency decay, ensemble predictions, walk-forward evaluation.
// - Predictor-only: reads outcomes, logs, predicts next. DOES NOT place bets.

const puppeteer = require('puppeteer');
const readlineSync = require('readline-sync');
const fs = require('fs');
const path = require('path');

// Try to use simple-statistics for convenience (chi-square p-values etc.)
// but we include fallbacks if it's not installed.
let ss = null;
try { ss = require('simple-statistics'); } catch (e) { /* optional */ }

// ---------- CONFIG ----------
const DATA_FILE = path.join(__dirname, 'rb_history.json'); // stored scraped outcomes
const MODEL_FILE = path.join(__dirname, 'rb_model.json');  // optional persisted model
const LOGFILE = path.join(__dirname, 'rb_predictions_log.csv');

const TIMEOUT = 60000;
const SCRAPE_SELECTOR = '#app > div > div > div.game-container-pad > div.align-items-center.d-flex.justify-content-center.mt-1.win-lose';
const NEXT_HAND_SELECTOR = '#app > div > div > div.game-container-pad > div.align-items-center.d-flex.justify-content-center.mt-1.win-lose > div:nth-child(2)';

const OUTCOMES = ['RED', 'BLACK', 'GREEN'];

// Feature window sizes and modeling hyperparams
const N_WINDOW = 10;         // how many past outcomes to encode/count
const EPOCHS = 120;          // epochs for logistic training per call (can be tuned)
const LR = 0.05;             // learning rate for softmax SGD
const L2 = 0.001;            // L2 regularization weight
const DECAY = 0.985;         // recency decay for counts / Markov (same idea as earlier)
const MIX_MARKOV = 0.35;     // ensemble weight for Markov model
const MIX_LOGREG = 0.55;     // ensemble weight for logistic model
const MIX_MARG = 0.10;       // marginal fallback weight
const STREAK_WINDOW = 3;
const STREAK_BONUS = 0.06;   // small bias away from long streak outcome

// Accuracy tracking
let lastPick = null;         // prediction for the upcoming round
let totalPred = 0;
let correctPred = 0;
const RECENT_WINDOW = 20;
let recentPredictions = [];  // booleans; true if correct

// ---------- Utilities & Data I/O ----------
function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

function loadHistory() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const raw = fs.readFileSync(DATA_FILE, 'utf8');
      return JSON.parse(raw);
    }
  } catch (e) { /* ignore */ }
  return []; // entries: { ts, outcome (RED/BLACK/GREEN) }
}
function saveHistory(history) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(history, null, 2));
}
function appendLogCsv(line) {
  try {
    if (!fs.existsSync(LOGFILE)) fs.writeFileSync(LOGFILE, 'ts,observed,prevPred,prob_red,prob_black,prob_green,correct\n');
    fs.appendFileSync(LOGFILE, line + '\n');
  } catch (e) { /* ignore logging errors */ }
}

// Normalize messy UI text into RED|BLACK|GREEN using whole-word match
// **Improved**: remove non-breaking spaces, punctuation, and collapse text neighbors
function normalizeOutcomeText(raw) {
  if (!raw) return null;
  // convert to string, normalize spaces including NBSP
  let up = String(raw).replace(/\u00A0/g, ' ').replace(/[\u200B-\u200D]/g, '').trim().toUpperCase();
  // remove punctuation that may be stuck to words (commas, currency symbols, etc)
  up = up.replace(/[^\w\s]/g, ' ');
  // collapse multiple spaces
  up = up.replace(/\s+/g, ' ').trim();
  // whole-word search
  const m = up.match(/\b(GREEN|RED|BLACK)\b/);
  if (m && m[1]) return m[1];
  return null;
}

// ---------- Statistical Tests (unchanged) ----------

function chiSquareTest(history) {
  const counts = Object.fromEntries(OUTCOMES.map(o => [o, 0]));
  for (const r of history) { if (counts.hasOwnProperty(r.outcome)) counts[r.outcome]++; }
  const total = history.length || 1;
  const expected = total / OUTCOMES.length;
  let chi2 = 0;
  for (const o of OUTCOMES) {
    const diff = counts[o] - expected;
    chi2 += diff * diff / (expected || 1);
  }
  let p = null;
  if (ss && typeof ss.chiSquaredProbability === 'function') {
    try { p = 1 - ss.chiSquaredDistribution(chi2, OUTCOMES.length - 1); } catch (e) { p = null; }
  }
  return { chi2, counts, total, p };
}

function runsTest(history) {
  const seq = history.map(h => h.outcome).filter(o => o !== 'GREEN');
  if (seq.length < 2) return null;
  const mapped = seq.map(o => (o === 'RED' ? 1 : 0));
  const n1 = mapped.reduce((a, b) => a + b, 0);
  const n2 = mapped.length - n1;
  if (n1 === 0 || n2 === 0) return null;
  let runs = 1;
  for (let i = 1; i < mapped.length; i++) if (mapped[i] !== mapped[i - 1]) runs++;
  const expected = (2 * n1 * n2) / (n1 + n2) + 1;
  const varRuns = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1) || 1);
  const z = (runs - expected) / Math.sqrt(varRuns || 1);
  const p = 2 * (1 - normalCdf(Math.abs(z)));
  return { runs, expected, variance: varRuns, z, p, n1, n2 };
}

function normalCdf(x) {
  const t = 1 / (1 + 0.2316419 * x);
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return 1 - prob;
}

function lag1Autocorr(history) {
  const seq = history.map(h => h.outcome).filter(o => o !== 'GREEN');
  if (seq.length < 2) return null;
  const nums = seq.map(o => (o === 'RED' ? 1 : 0));
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  let num = 0, denom = 0;
  for (let i = 0; i < nums.length - 1; i++) {
    num += (nums[i] - mean) * (nums[i + 1] - mean);
  }
  for (let i = 0; i < nums.length; i++) denom += (nums[i] - mean) ** 2;
  return denom ? num / denom : null;
}

// ---------- Feature Engineering (unchanged) ----------
function buildFeatureFromHistory(history, i) {
  const feat = [];
  const start = Math.max(0, i - N_WINDOW);
  const window = history.slice(start, i);
  const counts = { RED: 0, BLACK: 0, GREEN: 0 };
  for (const w of window) counts[w.outcome] = (counts[w.outcome] || 0) + 1;
  const windowLen = Math.max(1, window.length);
  feat.push(counts.RED / windowLen, counts.BLACK / windowLen, counts.GREEN / windowLen);

  const last = (window.length ? window[window.length - 1].outcome : null);
  feat.push(last === 'RED' ? 1 : 0, last === 'BLACK' ? 1 : 0, last === 'GREEN' ? 1 : 0);

  let streak = 0;
  if (window.length) {
    const val = window[window.length - 1].outcome;
    for (let j = window.length - 1; j >= 0; j--) {
      if (window[j].outcome === val) streak++;
      else break;
    }
  }
  feat.push(Math.min(streak, 9) / 9);

  const pos = (history.length > 0 ? (history.length % 5) : 0);
  for (let p = 0; p < 5; p++) feat.push(pos === p ? 1 : 0);

  const lastTs = (window.length ? window[window.length - 1].ts : Date.now());
  const dt = new Date(lastTs);
  const hour = dt.getHours() + dt.getMinutes() / 60;
  feat.push(Math.sin((2 * Math.PI * hour) / 24), Math.cos((2 * Math.PI * hour) / 24));

  const longStart = Math.max(0, i - 200);
  const longWindow = history.slice(longStart, i);
  const longCounts = { RED: 0, BLACK: 0, GREEN: 0 };
  for (const w of longWindow) longCounts[w.outcome]++;
  const longLen = Math.max(1, longWindow.length);
  feat.push(longCounts.RED / longLen, longCounts.BLACK / longLen, longCounts.GREEN / longLen);

  return feat;
}

// ---------- Softmax logistic (unchanged) ----------
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map(e => e / sum);
}
function zeros(rows, cols) {
  const M = new Array(rows);
  for (let i = 0; i < rows; i++) M[i] = new Array(cols).fill(0);
  return M;
}
function dotRowVec(row, vec) {
  let s = 0;
  for (let i = 0; i < row.length; i++) s += row[i] * vec[i];
  return s;
}
function trainSoftmax(X, Y, opts = {}) {
  const N = X.length;
  if (N === 0) return null;
  const D = X[0].length;
  const K = OUTCOMES.length;
  const lr = opts.lr || LR;
  const epochs = opts.epochs || EPOCHS;
  const l2 = opts.l2 || L2;
  const batchSize = Math.min(64, Math.max(8, Math.floor(N / 8)));
  const W = zeros(K, D);
  const b = new Array(K).fill(0);
  for (let k = 0; k < K; k++) for (let d = 0; d < D; d++) W[k][d] = (Math.random() - 0.5) * 0.01;
  for (let ep = 0; ep < epochs; ep++) {
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let bstart = 0; bstart < N; bstart += batchSize) {
      const bend = Math.min(N, bstart + batchSize);
      const gradW = zeros(K, D);
      const gradb = new Array(K).fill(0);
      for (let ii = bstart; ii < bend; ii++) {
        const i = idx[ii];
        const x = X[i];
        const logits = new Array(K);
        for (let k = 0; k < K; k++) logits[k] = dotRowVec(W[k], x) + b[k];
        const probs = softmax(logits);
        const y = Y[i];
        for (let k = 0; k < K; k++) {
          const err = probs[k] - (k === y ? 1 : 0);
          gradb[k] += err;
          for (let d = 0; d < D; d++) gradW[k][d] += err * x[d];
        }
      }
      const bs = (bend - bstart) || 1;
      for (let k = 0; k < K; k++) {
        gradb[k] /= bs;
        b[k] -= lr * gradb[k];
        for (let d = 0; d < D; d++) {
          gradW[k][d] = gradW[k][d] / bs + l2 * W[k][d];
          W[k][d] -= lr * gradW[k][d];
        }
      }
    }
  }
  return { W, b };
}
function predictSoftmaxModel(model, x) {
  if (!model) return null;
  const K = OUTCOMES.length;
  const logits = new Array(K);
  for (let k = 0; k < K; k++) logits[k] = dotRowVec(model.W[k], x) + model.b[k];
  const probs = softmax(logits);
  const map = {};
  for (let k = 0; k < K; k++) map[OUTCOMES[k]] = probs[k];
  return map;
}

// ---------- Markov & marginals (unchanged) ----------
function buildMarkov(history) {
  const counts = {};
  for (const a of OUTCOMES) counts[a] = Object.fromEntries(OUTCOMES.map(o => [o, 0.0]));
  const n = history.length;
  for (let i = 1; i < n; i++) {
    const prev = history[i - 1].outcome;
    const cur = history[i].outcome;
    if (!OUTCOMES.includes(prev) || !OUTCOMES.includes(cur)) continue;
    const age = n - 1 - i;
    const w = Math.pow(DECAY, age);
    counts[prev][cur] += w;
  }
  const probs = {};
  for (const a of OUTCOMES) {
    const row = counts[a];
    let s = 0;
    for (const b of OUTCOMES) s += row[b];
    if (s === 0) {
      probs[a] = Object.fromEntries(OUTCOMES.map(o => [o, 1 / OUTCOMES.length]));
    } else {
      probs[a] = {};
      for (const b of OUTCOMES) probs[a][b] = row[b] / s;
    }
  }
  return probs;
}
function buildMarginal(history) {
  const counts = Object.fromEntries(OUTCOMES.map(o => [o, 0.0]));
  const n = history.length;
  for (let i = 0; i < n; i++) {
    const o = history[i].outcome;
    const age = n - 1 - i;
    const w = Math.pow(DECAY, age);
    counts[o] += w;
  }
  const s = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const marg = {};
  for (const o of OUTCOMES) marg[o] = counts[o] / s;
  return marg;
}

// ---------- Helpers: argmax with random tie-break ----------
function argmaxWithRandomTie(dist) {
  // collect max prob(s)
  let bestP = -Infinity;
  for (const k of Object.keys(dist)) if (dist[k] > bestP) bestP = dist[k];
  // tolerance for floating point equality
  const tol = 1e-9;
  const candidates = Object.keys(dist).filter(k => Math.abs(dist[k] - bestP) <= tol);
  if (candidates.length === 1) return candidates[0];
  // If multiple maxima (or near-equal), pick randomly among them
  return candidates[Math.floor(Math.random() * candidates.length)];
}

// ---------- Ensemble ----------
function ensemblePredict(history, model) {
  const margl = buildMarginal(history);
  const markov = buildMarkov(history);
  const i = history.length;
  const feat = buildFeatureFromHistory(history, i);
  const logregProbs = model ? predictSoftmaxModel(model, feat) : Object.fromEntries(OUTCOMES.map(o => [o, 1 / OUTCOMES.length]));
  const last = (history.length ? history[history.length - 1].outcome : null);
  const markovProbs = (last && markov[last]) ? markov[last] : margl;
  const combined = {};
  for (const o of OUTCOMES) {
    combined[o] = MIX_LOGREG * (logregProbs[o] || 0) + MIX_MARKOV * (markovProbs[o] || 0) + MIX_MARG * (margl[o] || 0);
  }
  const cl = applyStreakBias(combined, history.map(h => h.outcome));
  const total = Object.values(cl).reduce((a, b) => a + b, 0) || 1;
  for (const k of Object.keys(cl)) cl[k] = cl[k] / total;

  // Debug: if model collapsed to single outcome, print a warning
  const probsArr = Object.values(cl);
  const maxP = Math.max(...probsArr);
  if (maxP > 0.92) {
    console.log(`⚠️ Model strongly biased to ${Object.keys(cl).find(k => cl[k]===maxP)} (${(maxP*100).toFixed(1)}%).`);
  }

  return cl;
}

function applyStreakBias(probs, historyList) {
  if (!historyList || historyList.length < STREAK_WINDOW) return probs;
  const tail = historyList.slice(-STREAK_WINDOW);
  if (tail.every(v => v === tail[0]) && OUTCOMES.includes(tail[0])) {
    const repeated = tail[0];
    const adj = Object.assign({}, probs);
    const shift = Math.min(STREAK_BONUS, adj[repeated] * 0.5);
    adj[repeated] = Math.max(0, adj[repeated] - shift);
    const others = OUTCOMES.filter(o => o !== repeated);
    const addEach = shift / Math.max(1, others.length);
    others.forEach(o => adj[o] = (adj[o] || 0) + addEach);
    return adj;
  }
  return probs;
}

// ---------- Walk-forward evaluation (unchanged) ----------
function walkForwardEvaluate(history, options = {}) {
  const minTrain = options.minTrain || Math.max(20, N_WINDOW + 5);
  if (history.length <= minTrain) return { tested: 0, accuracy: 0, perOutcome: {} };
  let correct = 0, tested = 0;
  const perOutcome = Object.fromEntries(OUTCOMES.map(o => [o, { total: 0, correct: 0 }]));
  for (let i = minTrain; i < history.length; i++) {
    const trainHist = history.slice(0, i);
    const testSample = history[i];
    const X = [], Y = [];
    for (let j = N_WINDOW; j < trainHist.length; j++) {
      X.push(buildFeatureFromHistory(trainHist, j));
      const label = OUTCOMES.indexOf(trainHist[j].outcome);
      if (label >= 0) Y.push(label);
    }
    if (X.length < 10) continue;
    const model = trainSoftmax(X, Y, { lr: 0.08, epochs: 60, l2: 0.002 });
    const dist = ensemblePredict(trainHist, model);
    const pick = argmaxWithRandomTie(dist);
    perOutcome[testSample.outcome].total++;
    if (pick === testSample.outcome) {
      perOutcome[testSample.outcome].correct++;
      correct++;
    }
    tested++;
  }
  const accuracy = tested ? (correct / tested) : 0;
  return { tested, accuracy, perOutcome };
}

// ---------- Scraper (Puppeteer) ----------
async function scrapeOneOutcome(page) {
  try {
    const handle = await page.waitForFunction(
      (sel) => {
        const el = document.querySelector(sel);
        if (!el) return null;
        const txt = (el.textContent || '').toUpperCase();
        if (/\b(GREEN|RED|BLACK)\b/.test(txt)) return txt;
        return null;
      },
      { timeout: TIMEOUT },
      SCRAPE_SELECTOR
    );
    const raw = await handle.jsonValue();
    const normalized = normalizeOutcomeText(raw);
    return normalized;
  } catch (err) {
    return null;
  }
}

// Append scraped outcome to DATA_FILE
function recordOutcome(outcome) {
  if (!outcome) return;
  const history = loadHistory();
  const lastEntry = history.length ? history[history.length - 1] : null;
  const last = lastEntry ? lastEntry.outcome : null;

  // improved duplicate handling:
  // - if same outcome and last timestamp is within 2 seconds -> assume UI still showing same round, skip
  // - otherwise allow append (covers real repeated identical outcomes when they truly occur)
  if (last === outcome && lastEntry && (Date.now() - lastEntry.ts) < 2000) {
    // skip ultra-fast duplicate
    return;
  }

  history.push({ ts: Date.now(), outcome });
  saveHistory(history);
}

// ---------- Main scrape & predict loop ----------
async function runScrapeAndPredictLoop() {
  const phone = readlineSync.question('📱 Enter your SportyBet phone number: ');
  const pass = readlineSync.question('🔐 Enter your SportyBet phone password: ', { hideEchoBack: true });

  const browser = await puppeteer.launch({ headless: false, slowMo: 50, defaultViewport: null, args: ['--start-maximized'] });
  const page = await browser.newPage();

  // LOGIN
  try {
    await page.goto('https://www.sportybet.com/ng/m/', { waitUntil: 'domcontentloaded', timeout: TIMEOUT });
    const phoneSel = '#loginStep > div.login-container > form > div.verifyInputs.m-input-wap-wrapper.m-input-wap-group.m-input-wap-group--prepend input';
    const passSel = '#loginStep > div.login-container > form > div:nth-child(3) input';
    const loginBut = '#loginStep > div.login-container > form > button';
    await page.waitForSelector(phoneSel, { timeout: TIMEOUT });
    await page.type(phoneSel, phone);
    await page.waitForSelector(passSel, { timeout: TIMEOUT });
    await page.type(passSel, pass);
    await page.waitForSelector(loginBut, { timeout: TIMEOUT });
    await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle0', timeout: TIMEOUT }), page.click(loginBut)]);
    console.log('✅ Login successful!');
  } catch (e) {
    console.error('❌ Login failed:', e.message || e);
    await browser.close();
    return;
  }

  try {
    await page.goto('https://www.sportybet.com/ng/sportygames/red-black', { waitUntil: 'domcontentloaded', timeout: TIMEOUT });
  } catch (e) {
    console.warn('⚠ Could not navigate directly to red-black page:', e.message || e);
  }

  console.log('🔎 Starting scrape -> update -> predict loop. CTRL+C to stop.');

  let model = null;
  const histInit = loadHistory();
  if (histInit.length > Math.max(30, N_WINDOW)) {
    const X = [], Y = [];
    for (let i = N_WINDOW; i < histInit.length; i++) {
      X.push(buildFeatureFromHistory(histInit, i));
      Y.push(OUTCOMES.indexOf(histInit[i].outcome));
    }
    if (X.length > 10) {
      console.log('⏳ Training initial logistic model on existing history...');
      model = trainSoftmax(X, Y, { lr: 0.06, epochs: 120, l2: 0.0015 });
      console.log('✅ Initial model trained.');
    }
  }

  while (true) {
    try {
      const observed = await scrapeOneOutcome(page);
      if (!observed) {
        console.log('⚠️ Could not read outcome from page. Retrying after delay...');
        await sleep(3000);
        continue;
      }

      console.log(`🎲 Observed outcome: ${observed}`);

      // Evaluate last prediction (if any)
      if (lastPick !== null) {
        totalPred++;
        const ok = lastPick === observed;
        if (ok) correctPred++;
        recentPredictions.push(ok);
        if (recentPredictions.length > RECENT_WINDOW) recentPredictions.shift();
        const recentAcc = (recentPredictions.filter(Boolean).length / Math.max(1, recentPredictions.length) * 100).toFixed(1);
        const overallAcc = (correctPred / Math.max(1, totalPred) * 100).toFixed(1);
        console.log(`📈 Prediction eval -> lastPick=${lastPick} | correct=${ok ? 'YES' : 'NO'} | recent(${recentPredictions.length})=${recentAcc}% overall=${overallAcc}%`);
      }

      // Record outcome (skip very-fast duplicates)
      recordOutcome(observed);
      const hist = loadHistory();

      // Retrain model occasionally (every 5 new samples)
      if (!model || hist.length % 5 === 0) {
        const X = [], Y = [];
        for (let i = N_WINDOW; i < hist.length; i++) {
          X.push(buildFeatureFromHistory(hist, i));
          Y.push(OUTCOMES.indexOf(hist[i].outcome));
        }
        if (X.length > 10) {
          console.log('⏳ Retraining logistic model (incremental retrain)...');
          model = trainSoftmax(X, Y, { lr: 0.06, epochs: 80, l2: 0.0015 });
        }
      }

      // Ensemble predict for next
      const dist = ensemblePredict(hist, model);
      const pick = argmaxWithRandomTie(dist);

      // Log results (prevPred = lastPick)
      const ts = new Date().toISOString();
      const prevPred = lastPick || '';
      const correctFlag = (prevPred && prevPred === observed) ? 1 : 0;
      appendLogCsv(`${ts},${observed},${prevPred},${(dist.RED || 0).toFixed(4)},${(dist.BLACK || 0).toFixed(4)},${(dist.GREEN || 0).toFixed(4)},${correctFlag}`);

      console.log('🔮 Next probabilities ->', OUTCOMES.map(o => `${o}:${(dist[o] * 100).toFixed(1)}%`).join('  '));
      console.log('👉 Predicted next:', pick);

      // store the pick so when the next observed outcome arrives we can evaluate it
      lastPick = pick;

      // Move to next round (Play Next Hand)
      try {
        await page.waitForSelector(NEXT_HAND_SELECTOR, { timeout: 20000 });
        await page.click(NEXT_HAND_SELECTOR);
      } catch (e) {
        console.log('⚠️ Could not click Play Next Hand (maybe not available):', e.message || e);
      }

      await sleep(2000);
    } catch (err) {
      console.log('⚠️ Loop error:', err.message || err);
      try { await page.reload({ waitUntil: 'domcontentloaded', timeout: TIMEOUT }); } catch (e) { /* ignore */ }
      await sleep(2000);
    }
  }
}

// ---------- Offline CLI (data/analysis/model) ----------
function printSummaryStats() {
  const hist = loadHistory();
  console.log(`Stored outcomes: ${hist.length}`);
  if (hist.length === 0) return;
  const chi = chiSquareTest(hist);
  console.log('Counts:', chi.counts, 'Chi2:', chi.chi2.toFixed(3), 'p:', chi.p || 'N/A');
  const runs = runsTest(hist);
  if (runs) console.log('Runs test:', runs);
  else console.log('Runs test: insufficient binary data (greens removed or too small).');
  const ac = lag1Autocorr(hist);
  console.log('Lag-1 autocorr (RED vs BLACK):', ac);
}

function buildDatasetFromHistory() {
  const hist = loadHistory();
  const X = [], Y = [];
  for (let i = N_WINDOW; i < hist.length; i++) {
    X.push(buildFeatureFromHistory(hist, i));
    Y.push(OUTCOMES.indexOf(hist[i].outcome));
  }
  return { X, Y, hist };
}

function cmdTrainAndEvaluate() {
  const { X, Y, hist } = buildDatasetFromHistory();
  if (X.length < 20) { console.log('Not enough data to train (need at least ~20 samples).'); return; }
  console.log('Training softmax logistic regression...');
  const model = trainSoftmax(X, Y, { lr: 0.06, epochs: 200, l2: 0.001 });
  console.log('Model trained. Running walk-forward evaluation (this may take a little while)...');
  const wfe = walkForwardEvaluate(hist);
  console.log('Walk-forward eval:', wfe);
  try {
    fs.writeFileSync(MODEL_FILE, JSON.stringify(model, null, 2));
    console.log('Saved model to', MODEL_FILE);
  } catch (e) { console.log('Could not save model:', e.message); }
}

async function mainCLI() {
  console.log('iv-redblack-advanced CLI');
  console.log('Commands: scrape (live), stats, train, predict, backtest, exit');
  const rl = require('readline').createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
  rl.prompt();
  rl.on('line', async (line) => {
    const cmd = line.trim().split(/\s+/)[0];
    try {
      if (cmd === 'scrape') {
        await runScrapeAndPredictLoop();
      } else if (cmd === 'stats') {
        printSummaryStats();
      } else if (cmd === 'train') {
        cmdTrainAndEvaluate();
      } else if (cmd === 'predict') {
        let model = null;
        if (fs.existsSync(MODEL_FILE)) model = JSON.parse(fs.readFileSync(MODEL_FILE, 'utf8'));
        const hist = loadHistory();
        const dist = ensemblePredict(hist, model);
        console.log('Prediction distribution:', dist);
        console.log('Pick:', argmaxWithRandomTie(dist));
      } else if (cmd === 'backtest') {
        const hist = loadHistory();
        const res = walkForwardEvaluate(hist);
        console.log('Backtest / walk-forward result:', res);
      } else if (cmd === 'exit' || cmd === 'quit') {
        rl.close(); process.exit(0);
      } else {
        console.log('Unknown command — available: scrape, stats, train, predict, backtest, exit');
      }
    } catch (e) {
      console.log('Error:', e.message || e);
    }
    rl.prompt();
  });
}

if (require.main === module) {
  mainCLI();
}
