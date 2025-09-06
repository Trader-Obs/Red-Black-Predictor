// sportypredictor.cjs
// Enhanced Red/Black/Green predictor: ensemble of slot-Bayes + Markov + pattern + streak
// Usage: node sportypredictor.cjs

const fs = require('fs');
const path = require('path');
const readline = require('readline');

////////////////////
// CONFIG
////////////////////
const MODE = 'redblackgreen';
const OUTCOMES = ['R','B','G']; // R=Red, B=Black, G=Green (neither)
const GAMES_PER_ROUND = 5;

const ALPHA = 0.5;    // Dirichlet prior
const DECAY = 0.97;   // recency decay
const MIX_GLOBAL = 0.25; // blending with global in slot posterior (kept for stability)

// Ensemble weights (can be tuned with 'tune' command)
let W_SLOT   = 0.5;
let W_MARKOV = 0.35;
let W_PATTERN= 0.12; // Slightly increased for patterns with 3 outcomes
let W_STREAK = 0.03; // Slightly adjusted

// Streak tweak parameters
const STREAK_WINDOW = 3; // lookback rounds for streak detection
const STREAK_BONUS = 0.06; // Reduced slightly for 3 outcomes; probability mass to shift towards other outcomes

const DATA_FILE = path.join(__dirname, 'sporty_history.json');

////////////////////
// IO
////////////////////
function loadHistory() {
  if (fs.existsSync(DATA_FILE)) {
    try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); }
    catch (e) { return []; }
  }
  return [];
}
function saveHistory(history) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(history, null, 2));
}

////////////////////
// UTIL
////////////////////
function parseRoundInput(raw) {
  const tokens = raw.replace(/,/g,' ').trim().split(/\s+/).map(t => t.toUpperCase());
  if (tokens.length !== GAMES_PER_ROUND) throw new Error(`Need exactly ${GAMES_PER_ROUND} outcomes (${OUTCOMES.join('/')}).`);
  if (!tokens.every(t => OUTCOMES.includes(t))) throw new Error(`Outcomes must be one of ${OUTCOMES.join(',')}`);
  return tokens;
}

function recencyWeights(n) {
  const w = new Array(n);
  for (let i=0;i<n;i++){
    const age = (n-1)-i;
    w[i] = Math.pow(DECAY, age);
  }
  return w;
}

function normalize(obj) {
  const sum = Object.values(obj).reduce((a,b)=>a+b,0) || 1;
  const out = {};
  for (const k of Object.keys(obj)) out[k] = obj[k]/sum;
  return out;
}

////////////////////
// MODELS: compute counts & posteriors
////////////////////
function buildModels(history) {
  const n = history.length;
  // slotCounts[slot][outcome]
  const slotCounts = Array.from({length:GAMES_PER_ROUND}, () => Object.fromEntries(OUTCOMES.map(o=>[o, ALPHA])));
  const globalCounts = Object.fromEntries(OUTCOMES.map(o=>[o, ALPHA]));

  // markovCounts[slot][prevOutcome][nextOutcome]
  const markovCounts = Array.from({length:GAMES_PER_ROUND}, () => {
    return Object.fromEntries(OUTCOMES.map(prev => [prev, Object.fromEntries(OUTCOMES.map(o=>[o, ALPHA]))]));
  });

  // patternCounts[key][slot][outcome]  (key = previousRound.join(','))
  const patternCounts = {}; // { key: [ {R:count,B:count,G:count}, ... slots ] }

  if (n>0) {
    const w = recencyWeights(n);
    for (let i=0;i<n;i++){
      const round = history[i].games;
      const wi = w[i];
      // slot/global
      for (let s=0;s<GAMES_PER_ROUND;s++){
        const o = round[s];
        if (OUTCOMES.includes(o)) {
          slotCounts[s][o] += wi;
          globalCounts[o] += wi;
        }
      }
      // markov: need previous round to form transitions
      if (i>0) {
        const prev = history[i-1].games;
        for (let s=0;s<GAMES_PER_ROUND;s++){
          const p = prev[s], cur = round[s];
          if (OUTCOMES.includes(p) && OUTCOMES.includes(cur)) {
            markovCounts[s][p][cur] += wi;
          }
        }
      }
      // pattern mapping: if i>0, map prev round -> current round's slot outcomes
      if (i>0) {
        const key = history[i-1].games.join(',');
        if (!patternCounts[key]) patternCounts[key] = Array.from({length:GAMES_PER_ROUND}, () => Object.fromEntries(OUTCOMES.map(o=>[o, ALPHA])));
        for (let s=0;s<GAMES_PER_ROUND;s++){
          const cur = round[s];
          if (OUTCOMES.includes(cur)) patternCounts[key][s][cur] += wi;
        }
      }
    }
  }

  // Convert to posteriors (normalize)
  const slotPosteriors = slotCounts.map(c => normalize(c));
  const globalPosterior = normalize(globalCounts);

  // Markov probabilities normalized per prevOutcome
  const markovProbs = markovCounts.map(slotObj => {
    const byPrev = {};
    for (const prev of OUTCOMES) byPrev[prev] = normalize(slotObj[prev]);
    return byPrev;
  });

  // Pattern probs normalized
  const patternProbs = {};
  for (const key of Object.keys(patternCounts)) {
    patternProbs[key] = patternCounts[key].map(c => normalize(c));
  }

  return { slotPosteriors, globalPosterior, markovProbs, patternProbs };
}

////////////////////
// ENSEMBLE PREDICTION
////////////////////
function applyStreakAdjustment(probs, history, slotIndex) {
  // if last STREAK_WINDOW rounds for this slot are identical, distribute STREAK_BONUS equally to other outcomes
  const n = history.length;
  if (n < STREAK_WINDOW) return probs; // nothing to do
  const tail = history.slice(-STREAK_WINDOW);
  const vals = tail.map(r => r.games[slotIndex]);
  if (vals.every(v=>v === vals[0]) && OUTCOMES.includes(vals[0])) {
    const streakOutcome = vals[0];
    const otherOutcomes = OUTCOMES.filter(o => o !== streakOutcome);
    if (otherOutcomes.length === 0) return probs;
    const adj = Object.assign({}, probs);
    const transferPerOther = STREAK_BONUS / otherOutcomes.length;
    otherOutcomes.forEach(opp => {
      adj[opp] = Math.min(1, adj[opp] + transferPerOther);
    });
    // renormalize
    const sum = Object.values(adj).reduce((a,b)=>a+b,0);
    for (const k of Object.keys(adj)) adj[k] = adj[k]/sum;
    return adj;
  }
  return probs;
}

function predictEnsemble(history, models, weights = null) {
  // weights: optional override {slot,markov,pattern,streak}
  const w_slot = weights?.slot ?? W_SLOT;
  const w_markov = weights?.markov ?? W_MARKOV;
  const w_pattern = weights?.pattern ?? W_PATTERN;
  const w_streak = weights?.streak ?? W_STREAK;
  // ensure sum -> 1 (normalize)
  const sumW = w_slot + w_markov + w_pattern + w_streak || 1;
  const ns = { slot: w_slot/sumW, markov: w_markov/sumW, pattern: w_pattern/sumW, streak: w_streak/sumW };

  const { slotPosteriors, globalPosterior, markovProbs, patternProbs } = models;
  const blendedSlots = []; // final per-slot prob distributions

  const lastRoundKey = history.length ? history[history.length-1].games.join(',') : null;
  for (let s=0;s<GAMES_PER_ROUND;s++){
    const slotP = slotPosteriors[s];
    // ensure blend with global for stability (same as before)
    const slotWithGlobal = {};
    for (const o of OUTCOMES) slotWithGlobal[o] = (1 - MIX_GLOBAL) * slotP[o] + MIX_GLOBAL * globalPosterior[o];

    // markov based on previous round same slot outcome
    let markovP = Object.fromEntries(OUTCOMES.map(o=>[o, 1/OUTCOMES.length]));
    if (history.length > 0) {
      const prev = history[history.length-1].games[s];
      if (OUTCOMES.includes(prev) && markovProbs[s] && markovProbs[s][prev]) markovP = markovProbs[s][prev];
    }

    // pattern-based if exact previous-round key exists
    let patternP = null;
    if (lastRoundKey && patternProbs[lastRoundKey]) {
      patternP = patternProbs[lastRoundKey][s];
    }

    // assemble combined probability
    const combined = {};
    for (const o of OUTCOMES) {
      const p_slot = slotWithGlobal[o] || 0;
      const p_markov = markovP[o] || 0;
      const p_pattern = (patternP ? (patternP[o] || 0) : p_slot); // fallback to slotP if pattern absent
      // base combination
      combined[o] = ns.slot * p_slot + ns.markov * p_markov + ns.pattern * p_pattern;
    }

    // apply streak adjustment if applicable (streak handled as a small bias)
    const withStreak = applyStreakAdjustment(combined, history, s);

    // final normalize
    blendedSlots.push(normalize(withStreak));
  }

  // choose argmax picks
  const picks = blendedSlots.map(dist => {
    let best = OUTCOMES[0], bestP = dist[best];
    for (const o of OUTCOMES.slice(1)) if (dist[o] > bestP) { best=o; bestP=dist[o]; }
    return { pick: best, probs: dist };
  });

  return { picks, blendedSlots, models };
}

////////////////////
// EVALUATION & TUNING
////////////////////
function walkForwardEvaluate(history, weightsOverride=null) {
  // simulate predictions from history[0..i-1] and check vs history[i]
  if (history.length < 2) return { rounds:0, correctRounds:0, slotAccuracy:0, totalSlots:0 };

  let totalRounds = 0, correctRounds = 0, correctSlots = 0, totalSlots = 0;
  const prefix = [];
  for (let i=0;i<history.length-1;i++){
    prefix.push(history[i]);
    const models = buildModels(prefix);
    const { picks } = predictEnsemble(prefix, models, weightsOverride);
    const actual = history[i+1].games;
    totalRounds++;
    const roundCorrect = picks.every((p, idx)=>p.pick === actual[idx]);
    if (roundCorrect) correctRounds++;
    picks.forEach((p, idx) => { if (p.pick === actual[idx]) correctSlots++; totalSlots++; });
  }
  return {
    rounds: totalRounds,
    correctRounds,
    roundAccuracy: totalRounds ? correctRounds/totalRounds : 0,
    slotAccuracy: totalSlots ? correctSlots/totalSlots : 0
  };
}

function coarseTune(history) {
  // coarse grid over w_slot and w_markov in 0.1 steps, w_pattern = 1 - slot - markov - small stash for streak
  if (history.length < 3) return null;
  let best = { score: -1, weights: null, res: null };
  const step = 0.1;
  for (let s=0; s<=1; s+=step) {
    for (let m=0; m<=1-s; m+=step) {
      const p = 1 - s - m;
      const weights = { slot: s, markov: m, pattern: p*0.9, streak: p*0.1 }; // split p into pattern+streak
      const res = walkForwardEvaluate(history, weights);
      const score = res.slotAccuracy; // optimize slot accuracy
      if (score > best.score) best = { score, weights, res };
    }
  }
  return best;
}

////////////////////
// CLI
////////////////////
function printHelp() {
  console.log(`
Commands:
  add <5 outcomes>     Add a round (R/B/G). Example: add R,B,R,B,G
  predict              Predict next round (uses ensemble)
  show                 Show last 10 rounds
  undo                 Remove last round
  evaluate             Evaluate walk-forward accuracy on stored history
  tune                 Coarse tune ensemble weights on history
  config               Show current weights and hyperparams
  saveweights          Save current weights to disk (weights.json)
  loadweights          Load weights from disk (if present)
  reset                Clear all history
  help                 Show commands
  exit                 Quit
`);
}

function printPredictions(result) {
  if (!result || !result.picks || !Array.isArray(result.picks)) {
    console.log('Error: Invalid prediction result. Please try again.');
    return;
  }
  console.log('\n🔮 Predictions for next round:');
  result.picks.forEach((p,i) => {
    const probs = OUTCOMES.map(o => `${o}:${(p.probs[o]*100).toFixed(1)}%`).join('  ');
    console.log(`  Game ${i+1}: ${p.pick}  |  ${probs}`);
  });
  console.log('');
}

function showConfig() {
  console.log('Config:');
  console.log('  MODE:', MODE);
  console.log('  ALPHA:', ALPHA, 'DECAY:', DECAY, 'MIX_GLOBAL:', MIX_GLOBAL);
  console.log('  Ensemble weights: slot,markov,pattern,streak =',
    W_SLOT.toFixed(2), W_MARKOV.toFixed(2), W_PATTERN.toFixed(2), W_STREAK.toFixed(2));
  console.log('  STREAK_WINDOW:', STREAK_WINDOW, 'STREAK_BONUS:', STREAK_BONUS);
  console.log('');
}

async function runCLI() {
  console.log('Red, Black & Green ensemble predictor — rounds of', GAMES_PER_ROUND);
  let history = loadHistory();

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
  printHelp(); rl.prompt();

  rl.on('line', async (line) => {
    const trimmed = line.trim();
    if (!trimmed) return rl.prompt();
    const [cmd, ...rest] = trimmed.split(/\s+/);
    try {
      if (cmd === 'add') {
        const raw = rest.join(' ');
        const games = parseRoundInput(raw);
        history.push({ ts: Date.now(), games });
        saveHistory(history);
        console.log('✅ Round added:', games.join(','));
        // show immediate prediction
        const models = buildModels(history);
        printPredictions(predictEnsemble(history, models));
      } else if (cmd === 'predict') {
        const models = buildModels(history);
        printPredictions(predictEnsemble(history, models));
      } else if (cmd === 'show') {
        const last = history.slice(-10);
        if (!last.length) console.log('No history yet.');
        else last.forEach((r,i)=>console.log(`${history.length-last.length+i+1}: ${r.games.join(',')}`));
      } else if (cmd === 'undo') {
        if (!history.length) console.log('Nothing to remove.');
        else { history.pop(); saveHistory(history); console.log('↩️ Removed last round.'); }
      } else if (cmd === 'reset') {
        history = []; saveHistory(history); console.log('Cleared history.');
      } else if (cmd === 'evaluate') {
        const res = walkForwardEvaluate(history);
        console.log('Evaluation (walk-forward):');
        console.log('  Rounds tested:', res.rounds);
        console.log('  Round accuracy:', (res.roundAccuracy*100).toFixed(2)+'%');
        console.log('  Slot accuracy:', (res.slotAccuracy*100).toFixed(2)+'%');
      } else if (cmd === 'tune') {
        console.log('Running coarse tuning (this may take a moment)...');
        const best = coarseTune(history);
        if (!best) console.log('Not enough history (need at least 3 rounds).');
        else {
          console.log('Best score:', (best.score*100).toFixed(2)+'%');
          console.log('Best weights:', best.weights);
          // apply weights
          W_SLOT = best.weights.slot; W_MARKOV = best.weights.markov;
          W_PATTERN = best.weights.pattern; W_STREAK = best.weights.streak;
          console.log('Applied best weights.');
        }
      } else if (cmd === 'config') {
        showConfig();
      } else if (cmd === 'saveweights') {
        fs.writeFileSync(path.join(__dirname,'weights.json'), JSON.stringify({W_SLOT,W_MARKOV,W_PATTERN,W_STREAK},null,2));
        console.log('Saved weights to weights.json');
      } else if (cmd === 'loadweights') {
        try {
          const w = JSON.parse(fs.readFileSync(path.join(__dirname,'weights.json'),'utf8'));
          W_SLOT=w.W_SLOT; W_MARKOV=w.W_MARKOV; W_PATTERN=w.W_PATTERN; W_STREAK=w.W_STREAK;
          console.log('Loaded weights.');
        } catch(e) { console.log('No weights file found.'); }
      } else if (cmd === 'help') {
        printHelp();
      } else if (cmd === 'exit' || cmd === 'quit') {
        rl.close(); return;
      } else {
        console.log('Unknown command. Type help.');
      }
    } catch (e) {
      console.log('Error:', e.message);
    }
    rl.prompt();
  });

  rl.on('close', () => {
    console.log('Bye — history saved to sporty_history.json');
    process.exit(0);
  });
}

runCLI();