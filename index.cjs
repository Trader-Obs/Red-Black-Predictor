// redblack_predictor_fixed.cjs
// Improved extraction + change-detection to avoid spamming same outcome.
// Predictor-only: reads outcomes, logs accuracy, predicts next (does NOT place bets)

const puppeteer = require('puppeteer');
const readlineSync = require('readline-sync');
const fs = require('fs');

const TIMEOUT = 60000;
const SLEEP_SHORT = 3000;
const DECAY = 0.985;
const ALPHA = 0.5;
const MIX_GLOBAL = 0.20;
const STREAK_WINDOW = 3;
const STREAK_BONUS = 0.08;
const RECENT_WINDOW = 20;

const OUTCOMES = ['RED','BLACK','GREEN'];
const OUTCOME_FILE = 'rb_outcomes.json';
const LOGFILE = 'rb_predictions_with_green.log';

// model state
let lastOutcome = null;
let prevPrediction = null;
const trans = { RED:{RED:ALPHA,BLACK:ALPHA,GREEN:ALPHA}, BLACK:{RED:ALPHA,BLACK:ALPHA,GREEN:ALPHA}, GREEN:{RED:ALPHA,BLACK:ALPHA,GREEN:ALPHA} };
const marg  = { RED:ALPHA, BLACK:ALPHA, GREEN:ALPHA };

let totalPred = 0, correctPred = 0;
let recent = []; // booleans
let outcomesList = []; // in-memory sequence of observed outcomes (most recent last)

// helpers
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function normalizeOutcomeText(txt){
  if(!txt) return null;
  const up = txt.toUpperCase();
  if (/\bGREEN\b/.test(up)) return 'GREEN';
  if (/\bRED\b/.test(up)) return 'RED';
  if (/\bBLACK\b/.test(up)) return 'BLACK';
  return null;
}
function decayCounts(){
  for(const a of OUTCOMES){
    marg[a]*=DECAY;
    for(const b of OUTCOMES) trans[a][b]*=DECAY;
  }
}
function updateModel(observed){
  decayCounts();
  marg[observed] += 1;
  if (lastOutcome) trans[lastOutcome][observed] += 1;
  lastOutcome = observed;
}
function normalizeDict(obj){
  const sum = Object.values(obj).reduce((a,b)=>a+b,0) || 1;
  const out = {};
  for(const k of Object.keys(obj)) out[k] = obj[k]/sum;
  return out;
}
function applyStreakBias(probs, history){
  if(!history || history.length < STREAK_WINDOW) return probs;
  const tail = history.slice(-STREAK_WINDOW);
  if(tail.every(v=>v===tail[0]) && OUTCOMES.includes(tail[0])){
    const repeated = tail[0];
    const adj = Object.assign({}, probs);
    const shift = Math.min(STREAK_BONUS, adj[repeated]*0.5);
    adj[repeated] = Math.max(0, adj[repeated]-shift);
    const others = OUTCOMES.filter(o=>o!==repeated);
    const addEach = shift / others.length;
    others.forEach(o=> adj[o] = (adj[o]||0) + addEach);
    return normalizeDict(adj);
  }
  return probs;
}
function predictDistribution(historySnapshot){
  let baseCond;
  if (lastOutcome && trans[lastOutcome]) baseCond = normalizeDict(trans[lastOutcome]);
  else baseCond = normalizeDict(marg);
  const global = normalizeDict(marg);
  const blended = {};
  for(const o of OUTCOMES) blended[o] = (1 - MIX_GLOBAL) * (baseCond[o]||0) + MIX_GLOBAL * (global[o]||0);
  const biased = applyStreakBias(blended, historySnapshot);
  return normalizeDict(biased);
}
function argmax(dict){
  let best=null, bestP=-Infinity;
  for(const k of Object.keys(dict)) if(dict[k] > bestP){ best=k; bestP=dict[k]; }
  return best;
}
function logAccuracy(){
  const overall = totalPred ? (100 * correctPred / totalPred).toFixed(1) : '—';
  const windowAcc = recent.length ? (100 * recent.filter(Boolean).length / recent.length).toFixed(1) : '—';
  console.log(`📈 Accuracy — last ${recent.length || 0}: ${windowAcc}% | overall: ${overall}%`);
}
function appendLine(file, line){
  try{ fs.appendFileSync(file, line); } catch(e){ /* ignore */ }
}

// load persisted outcomes (if any)
function loadPersistedOutcomes(){
  try{
    if(fs.existsSync(OUTCOME_FILE)){
      const arr = JSON.parse(fs.readFileSync(OUTCOME_FILE,'utf8'));
      if(Array.isArray(arr)){
        outcomesList = arr.slice(-500); // cap
        // replay to seed model
        lastOutcome = null;
        for(const o of outcomesList) if(OUTCOMES.includes(o)) updateModel(o);
        console.log(`🗂️ Loaded ${outcomesList.length} persisted outcomes.`);
      }
    }
  }catch(e){ console.warn('⚠️ Could not load persisted outcomes:', e.message || e); }
}

// persist observed outcome
function persistOutcome(o){
  try{
    outcomesList.push(o);
    if(outcomesList.length > 1000) outcomesList.shift();
    fs.writeFileSync(OUTCOME_FILE, JSON.stringify(outcomesList, null, 2));
  }catch(e){ /* ignore */ }
}

// main
async function start(){
  loadPersistedOutcomes();

  const phone = readlineSync.question('📱 Enter your SportyBet phone number: ');
  const pass  = readlineSync.question('🔐 Enter your password: ', { hideEchoBack: true });

  const browser = await puppeteer.launch({ headless: false, slowMo: 50, defaultViewport: null, args: ['--start-maximized'] });
  const page = await browser.newPage();

  // login
  try{
    await page.goto('https://www.sportybet.com/ng/m/', { waitUntil:'domcontentloaded', timeout: TIMEOUT });
    const phoneSelector = '#loginStep > div.login-container > form > div.verifyInputs.m-input-wap-wrapper.m-input-wap-group.m-input-wap-group--prepend input';
    const passSelector  = '#loginStep > div.login-container > form > div:nth-child(3) input';
    const loginButton   = '#loginStep > div.login-container > form > button';

    await page.waitForSelector(phoneSelector, { timeout: TIMEOUT });
    await page.type(phoneSelector, phone);
    await page.waitForSelector(passSelector, { timeout: TIMEOUT });
    await page.type(passSelector, pass);
    await page.waitForSelector(loginButton, { timeout: TIMEOUT });

    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle0', timeout: TIMEOUT }),
      page.click(loginButton)
    ]);

    console.log('✅ Login successful!');
  }catch(e){
    console.error('❌ Login failed:', e.message || e);
    return;
  }

  // navigate to red-black
  try{
    await page.goto('https://www.sportybet.com/ng/sportygames/red-black', { waitUntil:'domcontentloaded', timeout: TIMEOUT });
  }catch(e){ console.warn('⚠️ Nav to red-black failed:', e.message || e); }

  await monitorLoop(page);
}

// Improved monitor loop: uses change-detection + strict token match
async function monitorLoop(page){
  const outcomeSelector = '#app > div > div > div.game-container-pad > div.align-items-center.d-flex.justify-content-center.mt-1.win-lose';
  const nextHandSelector = outcomeSelector + ' > div:nth-child(2)';
  let lastSeenRawText = ''; // track the last raw text we processed

  while(true){
    try{
      console.log('⏳ Waiting for a new round result (detects change + token)...');

      // wait for the container text to contain the token and differ from last seen raw text
      const handle = await page.waitForFunction(
        (sel, lastSeen) => {
          const el = document.querySelector(sel);
          if(!el) return null;
          const raw = (el.textContent || '').trim();
          const up = raw.toUpperCase();
          const match = up.match(/\b(RED|BLACK|GREEN)\b/);
          if(!match) return null;
          if(raw === lastSeen) return null; // skip if identical to last processed
          // return the raw text (we'll extract token in Node)
          return raw;
        },
        { timeout: TIMEOUT },
        outcomeSelector,
        lastSeenRawText
      );

      const rawText = await handle.jsonValue();
      // debug: show rawText (can be long)
      // console.log('DEBUG rawText:', JSON.stringify(rawText).slice(0,200));

      // extract first matching token
      const up = (rawText || '').toUpperCase();
      const m = up.match(/\b(RED|BLACK|GREEN)\b/);
      const observed = m ? m[1] : null;

      if(!observed){
        console.log('⚠️ Could not normalize observed token from rawText. Retrying...');
        lastSeenRawText = rawText;
        await sleep(1500);
        continue;
      }

      // mark lastSeenRawText so we don't re-handle same display
      lastSeenRawText = rawText;

      console.log(`🎲 Observed (raw): ${rawText}`);
      console.log(`🎯 Normalized outcome: ${observed}`);

      // compare with previous prediction (if we had one)
      if(prevPrediction){
        const ok = prevPrediction === observed;
        totalPred += 1;
        if(ok) correctPred += 1;
        recent.push(ok);
        if(recent.length > RECENT_WINDOW) recent.shift();

        const logLine = `${new Date().toISOString()} | observed=${observed} | prevPred=${prevPrediction} | ${ok ? 'OK' : 'WRONG'}\n`;
        appendLine(LOGFILE, logLine);
        logAccuracy();
      }

      // update model and persist
      updateModel(observed);
      persistOutcome(observed);

      // maintain small outcomesList already persisted
      // (updateModel already updated lastOutcome)
      // Predict distribution using the in-memory outcomesList
      const dist = predictDistribution(outcomesList);
      const nextPred = argmax(dist);
      prevPrediction = nextPred;

      console.log('🔮 Next probabilities -> ' + OUTCOMES.map(o => `${o}:${(dist[o]*100).toFixed(1)}%`).join('  '));
      console.log(`👉 Predicted next: ${nextPred}`);

      // try to click Play Next Hand to progress
      try{
        await page.waitForSelector(nextHandSelector, { timeout: 20000 });
        await page.click(nextHandSelector);
        await sleep(SLEEP_SHORT);
      }catch(e){
        console.log('⚠️ Could not click Play Next Hand:', e.message || e);
      }

      // small delay before next detection cycle
      await sleep(1200);

    }catch(err){
      console.log('⚠️ Loop error:', err.message || err);
      try{ await page.reload({ waitUntil:'domcontentloaded', timeout: TIMEOUT }); }catch(e){}
      await sleep(2000);
    }
  }
}

// run
start();
