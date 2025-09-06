const puppeteer = require('puppeteer');
const readlineSync = require('readline-sync');
const fs = require('fs');
const path = require('path');
const natural = require('natural');

const DATA_FILE = path.join(__dirname, 'virtual_match_data.json');

// Load match data
function loadMatchData() {
  if (fs.existsSync(DATA_FILE)) {
    const raw = fs.readFileSync(DATA_FILE);
    return JSON.parse(raw);
  }
  return [];
}

// Save match data
function saveMatchData(data) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
}

// Train a Naive Bayes model
function trainModel(data) {
  const classifier = new natural.BayesClassifier();
  data.forEach(match => {
    const features = `${match.home}-${match.away}`;
    classifier.addDocument(features, match.result);
  });
  classifier.train();
  return classifier;
}

// Predict match outcomes
function predictMatches(classifier, matches) {
  return matches.map(({ home, away }) => {
    const features = `${home}-${away}`;
    const prediction = classifier.classify(features);
    return { home, away, prediction };
  });
}

async function login(page, phoneNumber, password) {
  const phoneSelector = '#loginStep input[type="tel"]';
  const passSelector = '#loginStep input[type="password"]';
  const loginButtonSelector = '#loginStep button';

  await page.waitForSelector(phoneSelector);
  await page.type(phoneSelector, phoneNumber);
  await page.waitForSelector(passSelector);
  await page.type(passSelector, password);
  await page.waitForSelector(loginButtonSelector);
  await Promise.all([
    page.waitForNavigation({ waitUntil: 'domcontentloaded', timeout: 60000 }),
    page.click(loginButtonSelector)
  ]);
  console.log('✅ Login successful!');
}

// Scrape past match history
async function scrapeMatchHistory(page) {
  await page.goto('https://www.sportybet.com/ng/instant-virtuals/', { timeout: 120000 });
  await page.waitForSelector('#iv-live-score-result > div.result', { timeout: 120000 });

  const matchData = await page.$$eval('#iv-live-score-result > div.result', nodes => {
    return nodes.map(node => {
      const teamsText = node.innerText.match(/(.+?)\s+(\d+)\s+-\s+(\d+)\s+(.+)/);
      if (!teamsText) return null;
      const [, home, homeScore, awayScore, away] = teamsText;
      const hScore = parseInt(homeScore);
      const aScore = parseInt(awayScore);
      let result = 'draw';
      if (hScore > aScore) result = 'win';
      else if (aScore > hScore) result = 'lose';
      return {
        home: home.trim(),
        away: away.trim(),
        result
      };
    }).filter(Boolean);
  });

  return matchData;
}

// Scrape your selected upcoming matches
async function scrapeUpcomingMatches(page) {
  await page.waitForSelector('#quick-game-match-container > div.event-lists.scroll-level > div.m-table', { timeout: 120000 });

  const picks = await page.$$eval('#quick-game-match-container > div.event-lists.scroll-level > div.m-table > div > div.m-table-cell.table-team-column', nodes => {
    return nodes.map(node => {
      const teamsText = node.innerText.trim().split('vs');
      return {
        home: teamsText[0].trim(),
        away: teamsText[1].trim()
      };
    }).filter(m => m.home && m.away);
  });

  return picks;
}

async function start() {
  const phoneNumber = readlineSync.question('📱 Enter your SportyBet phone number: ');
  const password = readlineSync.question('🔐 Enter your password: ', { hideEchoBack: true });

  const browser = await puppeteer.launch({ headless: false, slowMo: 50, defaultViewport: null, args: ['--start-maximized'] });
  const page = await browser.newPage();
  await page.goto('https://www.sportybet.com/ng/m/', { waitUntil: 'domcontentloaded', timeout: 60000 });

  try {
    await login(page, phoneNumber, password);
  } catch (err) {
    console.error('❌ Login failed:', err);
    await browser.close();
    return;
  }

  try {
    console.log('📊 Scraping past match results...');
    const allMatchData = loadMatchData();
    const newMatchData = await scrapeMatchHistory(page);
    const updatedData = [...allMatchData, ...newMatchData];
    saveMatchData(updatedData);

    const model = trainModel(updatedData);

    console.log('🔍 Reading your selected matches...');
    const upcomingMatches = await scrapeUpcomingMatches(page);
    const predictions = predictMatches(model, upcomingMatches.slice(0, 10));

    predictions.forEach((p, i) => {
      console.log(`${i + 1}. ${p.home} vs ${p.away} => Predicted: ${p.prediction.toUpperCase()}`);
    });
  } catch (err) {
    console.error('❌ Failed during scraping or prediction:', err);
  }

  await browser.close();
}

start();
