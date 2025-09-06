# IV Red–Black Predictor 🎲

An experimental predictor for SportyBet’s **Red–Black (with Green)** game.  
This project uses **Puppeteer** to scrape live game outcomes, applies **Markov chains, streak bias, and probability weighting**, and attempts to predict the next outcome.  

⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a guaranteed way to win bets. Use responsibly.  

---

## ✨ Features
- ✅ Puppeteer-based scraping of live Red–Black–Green outcomes  
- 🔮 Prediction model using Markov transitions, global marginals, recency weighting, and streak adjustments  
- 📊 Accuracy tracking (overall and recent window)  
- 📝 Logs all outcomes and predictions for later backtesting  
- ⚡ Configurable hyperparameters for tuning model behavior  
- 🧪 Backtesting mode with saved logs or simulated sequences  

---

## 🛠 Tech Stack
- [Node.js](https://nodejs.org/)  
- [Puppeteer](https://pptr.dev/)  
- [simple-statistics](https://simplestatistics.org/) (optional, randomness/bias testing)  
- [TensorFlow.js](https://www.tensorflow.org/js) or [ml.js](https://github.com/mljs) (optional future ML upgrades)  

---

## 🚀 Installation
```bash
# clone repo
git clone https://github.com/yourusername/iv-redblack-predictor.git
cd iv-redblack-predictor

# install dependencies
npm install puppeteer readline-sync simple-statistics
