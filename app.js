/**
 * Main Application Logic & UI Controller for Fake News Shield
 * Manages views, runs NLP analysis, executes model training sandbox, and drives the quiz game.
 */

document.addEventListener('DOMContentLoaded', () => {
  
  // --- STATE SYSTEM ---
  let activeView = 'analyzer';
  let activeDataset = [...window.ml.BASELINE_DATASET];
  let preprocessedDataset = [];
  
  // Model Instances
  const vectorizer = new window.ml.TfIdfVectorizer();
  const nbModel = new window.ml.NaiveBayesClassifier();
  const lrModel = new window.ml.LogisticRegressionClassifier();
  const pacModel = new window.ml.PassiveAggressiveClassifier();
  
  // Sandbox state
  let sandboxHistory = []; // { epoch, loss, accuracy }
  let activeSandboxModel = 'logistic-regression';
  
  // Quiz state
  let quizIndex = 0;
  let quizScore = 0;
  const quizQuestions = window.quizData.questions;

  // --- HTML ELEMENTS ---
  
  // Navigation Tabs
  const navItems = {
    analyzer: document.getElementById('nav-item-analyzer'),
    sandbox: document.getElementById('nav-item-sandbox'),
    quiz: document.getElementById('nav-item-quiz'),
    learn: document.getElementById('nav-item-learn')
  };
  const navButtons = {
    analyzer: document.getElementById('btn-nav-analyzer'),
    sandbox: document.getElementById('btn-nav-sandbox'),
    quiz: document.getElementById('btn-nav-quiz'),
    learn: document.getElementById('btn-nav-learn')
  };
  const panels = {
    analyzer: document.getElementById('panel-analyzer'),
    sandbox: document.getElementById('panel-sandbox'),
    quiz: document.getElementById('panel-quiz'),
    learn: document.getElementById('panel-learn')
  };

  // Analyzer View Elements
  const selectPreset = document.getElementById('select-preset');
  const textareaInput = document.getElementById('textarea-input');
  const btnRunAnalysis = document.getElementById('btn-run-analysis');
  const lblCharCount = document.getElementById('lbl-char-count');
  
  const lblCredibilityScore = document.getElementById('lbl-credibility-score');
  const gaugeCredibilityFill = document.getElementById('gauge-credibility-fill');
  const badgeVerdict = document.getElementById('badge-verdict');
  
  const lblFeatClickbait = document.getElementById('lbl-feat-clickbait');
  const barFeatClickbait = document.getElementById('bar-feat-clickbait');
  const lblFeatSubjectivity = document.getElementById('lbl-feat-subjectivity');
  const barFeatSubjectivity = document.getElementById('bar-feat-subjectivity');
  const lblFeatCaps = document.getElementById('lbl-feat-caps');
  const barFeatCaps = document.getElementById('bar-feat-caps');
  const lblFeatReadability = document.getElementById('lbl-feat-readability');
  const barFeatReadability = document.getElementById('bar-feat-readability');
  
  const predNb = document.getElementById('pred-nb');
  const confNb = document.getElementById('conf-nb');
  const predLr = document.getElementById('pred-lr');
  const confLr = document.getElementById('conf-lr');
  const predPac = document.getElementById('pred-pac');
  const confPac = document.getElementById('conf-pac');
  
  const containerHighlightedText = document.getElementById('container-highlighted-text');

  // Sandbox View Elements
  const selectSandboxAlgo = document.getElementById('select-sandbox-algo');
  const sliderEpochs = document.getElementById('slider-epochs');
  const lblValEpochs = document.getElementById('lbl-val-epochs');
  const sliderLr = document.getElementById('slider-lr');
  const lblValLr = document.getElementById('lbl-val-lr');
  const sliderReg = document.getElementById('slider-reg');
  const lblValReg = document.getElementById('lbl-val-reg');
  const lblRegTitle = document.getElementById('lbl-reg-title');
  const sliderSplit = document.getElementById('slider-split');
  const lblValSplit = document.getElementById('lbl-val-split');
  
  const btnTrainSandbox = document.getElementById('btn-train-sandbox');
  const canvasSandbox = document.getElementById('chart-sandbox');
  const consoleLogs = document.getElementById('console-logs');
  const tableDatasetBody = document.getElementById('table-dataset-body');
  const lblDatasetSize = document.getElementById('lbl-dataset-size');
  
  const inputNewSample = document.getElementById('input-new-sample');
  const selectNewLabel = document.getElementById('select-new-label');
  const btnAddSample = document.getElementById('btn-add-sample');
  
  const cellTp = document.getElementById('cell-tp');
  const cellTn = document.getElementById('cell-tn');
  const cellFp = document.getElementById('cell-fp');
  const cellFn = document.getElementById('cell-fn');
  
  const lblPrecision = document.getElementById('lbl-precision');
  const lblRecall = document.getElementById('lbl-recall');
  const lblF1 = document.getElementById('lbl-f1');
  const lblTestAccuracy = document.getElementById('lbl-test-accuracy');

  const containerHyperEpochs = document.getElementById('container-hyper-epochs');
  const containerHyperLr = document.getElementById('container-hyper-lr');
  const containerHyperReg = document.getElementById('container-hyper-reg');

  // Quiz View Elements
  const lblQuizProgress = document.getElementById('lbl-quiz-progress');
  const lblQuizScore = document.getElementById('lbl-quiz-score');
  const barQuizProgress = document.getElementById('bar-quiz-progress');
  const cardQuiz = document.getElementById('card-quiz');
  
  const quizSource = document.getElementById('quiz-source');
  const quizHeadlineText = document.getElementById('quiz-headline');
  const btnQuizReal = document.getElementById('btn-quiz-real');
  const btnQuizFake = document.getElementById('btn-quiz-fake');
  
  const quizActualVerdict = document.getElementById('quiz-actual-verdict');
  const quizVerdictTitle = document.getElementById('quiz-verdict-title');
  const quizExplanation = document.getElementById('quiz-explanation');
  const btnQuizNext = document.getElementById('btn-quiz-next');
  
  const containerQuizGame = document.getElementById('container-quiz-game');
  const containerQuizSummary = document.getElementById('container-quiz-summary');
  const lblSummaryScore = document.getElementById('lbl-summary-score');
  const lblSummaryFeedback = document.getElementById('lbl-summary-feedback');
  const btnQuizRestart = document.getElementById('btn-quiz-restart');

  // --- PRESET TEXT VALUES ---
  const PRESETS = {
    'preset-1': "SHOCKING: Doctors Can't Believe This One Miracle Fruit Cures Diabetes In 24 Hours! Pharma Companies Are Silencing This Secret!",
    'preset-2': "NASA's James Webb Space Telescope has captured a highly detailed image of the Ring Nebula, showing the intricate structure of the dying star.",
    'preset-3': "BREAKING: Global banks are shutting down all ATMs tomorrow at midnight. Get your cash out now! A massive financial coverup has begun!",
    'preset-4': "The European Central Bank lowered its key deposit rate by 25 basis points to 3.5% on Thursday, citing cooling inflation across the eurozone."
  };

  // --- INITIALIZATION ---
  function init() {
    // 1. Prepare routing
    setupRouter();
    
    // 2. Preprocess default dataset
    preprocessedDataset = window.ml.preprocessDataset(activeDataset);
    renderDatasetTable();
    
    // 3. Train models initially to prepare Analyzer
    vectorizer.fit(preprocessedDataset.map(item => item.stemmedTokens));
    nbModel.fit(preprocessedDataset, vectorizer);
    
    // Dummy training runs on background context to enable live predictions immediately
    lrModel.fit(preprocessedDataset, vectorizer, 0.1, 10, 0.01);
    pacModel.fit(preprocessedDataset, vectorizer, 1.0, 10);
    
    // 4. Initial event handlers
    setupEventListeners();
    setupLearnAccordions();
    updateCharCount();
    
    // 5. Build default canvas charts size
    resizeCanvas();
    drawInitialChart();
  }

  // --- ROUTER & VIEW MANAGEMENT ---
  function setupRouter() {
    Object.keys(navButtons).forEach(viewName => {
      navButtons[viewName].addEventListener('click', () => {
        switchView(viewName);
      });
    });
  }

  function switchView(viewName) {
    activeView = viewName;
    
    // Update active nav menu
    Object.keys(navItems).forEach(key => {
      if (key === viewName) {
        navItems[key].classList.add('active');
      } else {
        navItems[key].classList.remove('active');
      }
    });

    // Toggle panels
    Object.keys(panels).forEach(key => {
      if (key === viewName) {
        panels[key].classList.add('active');
      } else {
        panels[key].classList.remove('active');
      }
    });

    // View specific actions
    if (viewName === 'sandbox') {
      resizeCanvas();
      drawChart();
    } else if (viewName === 'quiz') {
      resetQuiz();
    }
  }

  // --- EVENT LISTENERS REGISTRY ---
  function setupEventListeners() {
    // Preset dropdown change
    selectPreset.addEventListener('change', (e) => {
      const selected = e.target.value;
      if (selected !== 'empty') {
        textareaInput.value = PRESETS[selected];
        updateCharCount();
        runAnalysis();
      }
    });

    // TextArea updates
    textareaInput.addEventListener('input', updateCharCount);

    // Analysis trigger
    btnRunAnalysis.addEventListener('click', runAnalysis);

    // Hyperparameter live updates
    sliderEpochs.addEventListener('input', (e) => {
      lblValEpochs.textContent = e.target.value;
    });
    
    sliderLr.addEventListener('input', (e) => {
      lblValLr.textContent = parseFloat(e.target.value).toFixed(2);
    });

    sliderReg.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      lblValReg.textContent = activeSandboxModel === 'passive-aggressive' ? val.toFixed(1) : val.toFixed(3);
    });

    sliderSplit.addEventListener('input', (e) => {
      const val = e.target.value;
      lblValSplit.textContent = `${val}% / ${100 - val}%`;
    });

    // Algo selector switches controls
    selectSandboxAlgo.addEventListener('change', (e) => {
      activeSandboxModel = e.target.value;
      updateSandboxControlsVisibility();
    });

    // Sandbox execute training
    btnTrainSandbox.addEventListener('click', executeSandboxTraining);

    // Add Sample
    btnAddSample.addEventListener('click', handleAddSample);

    // Quiz Buttons
    btnQuizReal.addEventListener('click', () => submitQuizAnswer('real'));
    btnQuizFake.addEventListener('click', () => submitQuizAnswer('fake'));
    btnQuizNext.addEventListener('click', advanceQuizQuestion);
    btnQuizRestart.addEventListener('click', resetQuiz);

    // Dynamic canvas resize support
    window.addEventListener('resize', () => {
      if (activeView === 'sandbox') {
        resizeCanvas();
        drawChart();
      }
    });
  }

  function setupLearnAccordions() {
    document.querySelectorAll('.accordion-trigger').forEach(trigger => {
      trigger.addEventListener('click', () => {
        const item = trigger.closest('.accordion-item');
        const isOpen = item.classList.contains('open');
        
        // Close all others
        document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('open'));
        
        if (!isOpen) {
          item.classList.add('open');
        }
      });
    });
  }

  function updateCharCount() {
    const text = textareaInput.value;
    const charCount = text.length;
    const wordCount = window.nlp.tokenize(text).length;
    lblCharCount.textContent = `${charCount} characters | ${wordCount} words`;
  }

  function updateSandboxControlsVisibility() {
    if (activeSandboxModel === 'naive-bayes') {
      containerHyperEpochs.style.display = 'none';
      containerHyperLr.style.display = 'none';
      containerHyperReg.style.display = 'none';
    } else if (activeSandboxModel === 'logistic-regression') {
      containerHyperEpochs.style.display = 'flex';
      containerHyperLr.style.display = 'flex';
      containerHyperReg.style.display = 'flex';
      lblRegTitle.textContent = "L2 Regularization Coefficient (lambda)";
      // Reset slider bounds
      sliderReg.min = "0.001";
      sliderReg.max = "0.5";
      sliderReg.step = "0.005";
      sliderReg.value = "0.01";
      lblValReg.textContent = "0.010";
    } else if (activeSandboxModel === 'passive-aggressive') {
      containerHyperEpochs.style.display = 'flex';
      containerHyperLr.style.display = 'none';
      containerHyperReg.style.display = 'flex';
      lblRegTitle.textContent = "PAC Aggressiveness Bound (C)";
      // Adapt slider bounds
      sliderReg.min = "0.1";
      sliderReg.max = "10.0";
      sliderReg.step = "0.1";
      sliderReg.value = "1.0";
      lblValReg.textContent = "1.0";
    }
  }

  // --- ANALYZER BUSINESS LOGIC ---
  function runAnalysis() {
    const text = textareaInput.value.trim();
    if (text.length < 15) {
      alert("Please enter a longer text snippet (at least 15 characters) to compile a valid NLP signature.");
      return;
    }

    // 1. Trigger NLP Extractor
    const nlpData = window.nlp.analyzeText(text);

    // 2. Compute Model predictions
    const tokens = nlpData.stemmedTokens;
    const nbPred = nbModel.predict(tokens);
    const lrPred = lrModel.predict(tokens, vectorizer);
    const pacPred = pacModel.predict(tokens, vectorizer);

    // Update individual classifier views
    updateClassifierUI(predNb, confNb, nbPred);
    updateClassifierUI(predLr, confLr, lrPred);
    updateClassifierUI(predPac, confPac, pacPred);

    // 3. Compute combined Credibility Score
    const mlCombinedRealProb = (nbPred.probabilities.real + lrPred.probabilities.real + pacPred.probabilities.real) / 3;
    
    // Incorporate linguistic penalty: high clickbait, subjectivity, caps ratio drops credibility
    const clickbaitRatio = nlpData.clickbaitScore / 100;
    const subjectivityRatio = nlpData.subjectivityScore / 100;
    const capsPen = Math.min(1.0, nlpData.stylometrics.capsRatio * 6); // 16%+ caps ratio is heavily penalized
    
    const stylePenalty = (clickbaitRatio * 0.45) + (subjectivityRatio * 0.25) + (capsPen * 0.3);
    
    // Base credibility calculation
    let credibility = Math.round(mlCombinedRealProb * 55 + (1 - stylePenalty) * 45);
    credibility = Math.max(2, Math.min(99, credibility)); // Bound score between 2% and 99%

    // 4. Update Main Gauge
    lblCredibilityScore.textContent = `${credibility}%`;
    
    // Circle path offset animation: 502 stroke dash array
    const offset = 502 - (502 * credibility / 100);
    gaugeCredibilityFill.style.strokeDashoffset = offset;

    // Apply color themes based on security range
    badgeVerdict.style.display = 'inline-block';
    badgeVerdict.className = 'verdict-tag ';
    
    if (credibility >= 70) {
      badgeVerdict.textContent = "Verified Factual";
      badgeVerdict.classList.add('verdict-real');
      gaugeCredibilityFill.style.stroke = 'var(--color-real)';
      gaugeCredibilityFill.style.filter = 'drop-shadow(0 0 6px var(--color-real-glow))';
    } else if (credibility >= 40) {
      badgeVerdict.textContent = "Suspicious Content";
      badgeVerdict.classList.add('verdict-suspicious');
      gaugeCredibilityFill.style.stroke = 'var(--color-warning)';
      gaugeCredibilityFill.style.filter = 'drop-shadow(0 0 6px rgba(245, 158, 11, 0.3))';
    } else {
      badgeVerdict.textContent = "Misleading / Fake";
      badgeVerdict.classList.add('verdict-fake');
      gaugeCredibilityFill.style.stroke = 'var(--color-fake)';
      gaugeCredibilityFill.style.filter = 'drop-shadow(0 0 6px var(--color-fake-glow))';
    }

    // 5. Update stylometric charts
    animateBar(lblFeatClickbait, barFeatClickbait, nlpData.clickbaitScore, '%');
    animateBar(lblFeatSubjectivity, barFeatSubjectivity, nlpData.subjectivityScore, '%');
    animateBar(lblFeatCaps, barFeatCaps, Math.round(nlpData.stylometrics.capsRatio * 100), '%');
    lblFeatReadability.textContent = `Grade ${nlpData.readabilityGrade}`;
    barFeatReadability.style.width = `${Math.min(100, (nlpData.readabilityGrade / 16) * 100)}%`;

    // 6. Sentence Highlights Map
    renderRedFlagsMap(text, nlpData.clickbaitTriggers);
  }

  function updateClassifierUI(predElement, confElement, predictionObj) {
    const isReal = predictionObj.label === 'real';
    predElement.textContent = isReal ? 'Real' : 'Fake';
    predElement.className = 'algo-prediction ' + (isReal ? 'algo-pred-real' : 'algo-pred-fake');
    confElement.textContent = `${(predictionObj.confidence * 100).toFixed(1)}%`;
  }

  function animateBar(labelEl, barEl, scoreValue, suffix = '') {
    labelEl.textContent = `${scoreValue}${suffix}`;
    barEl.style.width = `${scoreValue}%`;
  }

  function renderRedFlagsMap(originalText, triggers) {
    if (triggers.length === 0) {
      containerHighlightedText.innerHTML = `<span style="color: var(--color-real);">Excellent. No severe stylistic clickbait or sensationalism patterns detected in this text sample.</span>`;
      return;
    }

    let highlightedHtml = originalText;
    
    // Sort triggers by match length descending to avoid nested string replacements breaking indexes
    const sortedTriggers = [...triggers].sort((a, b) => b.matchedText.length - a.matchedText.length);
    
    const replacedPhrases = new Set();
    
    sortedTriggers.forEach(t => {
      const match = t.matchedText;
      if (replacedPhrases.has(match.toLowerCase())) return;
      replacedPhrases.add(match.toLowerCase());

      // Replace match globally using safe HTML replacement wrapper
      // We escape characters and place highlights
      const regex = new RegExp(`\\b(${escapeRegExp(match)})\\b`, 'gi');
      highlightedHtml = highlightedHtml.replace(regex, (m) => {
        return `<span class="highlight-flagged" data-tooltip="${t.patternName}: ${t.description}">${m}</span>`;
      });
    });

    containerHighlightedText.innerHTML = highlightedHtml;
  }

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // --- MODEL TRAINING SANDBOX ENGINE ---
  function printLog(text, type = 'normal') {
    const line = document.createElement('div');
    line.className = 'console-line';
    if (type === 'header') line.classList.add('console-line-header');
    if (type === 'success') line.classList.add('console-line-success');
    
    line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    consoleLogs.appendChild(line);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
  }

  async function executeSandboxTraining() {
    // 1. Disable button
    btnTrainSandbox.disabled = true;
    btnTrainSandbox.style.opacity = '0.6';
    printLog(`Initializing training pipeline for classifier: ${activeSandboxModel.toUpperCase()}`, 'header');

    // 2. Read parameters
    const splitRatio = parseInt(sliderSplit.value) / 100;
    const numEpochs = parseInt(sliderEpochs.value);
    const learningRate = parseFloat(sliderLr.value);
    const regParam = parseFloat(sliderReg.value);

    // Shuffle active dataset
    const datasetCopy = [...preprocessedDataset];
    datasetCopy.sort(() => Math.random() - 0.5);

    // Train/Test Split
    const trainSize = Math.floor(datasetCopy.length * splitRatio);
    const trainData = datasetCopy.slice(0, trainSize);
    const testData = datasetCopy.slice(trainSize);

    printLog(`Corpus split compiled: ${trainData.length} training records, ${testData.length} validation records.`);

    // Fit Vectorizer on Training set
    vectorizer.fit(trainData.map(d => d.stemmedTokens));
    printLog(`Vocab vocabulary compiled. Features dimensions: ${vectorizer.vocabList.length} unique tokens.`);

    // 3. Clear chart state
    sandboxHistory = [];
    drawChart();

    const epochCallback = async (epoch, loss, acc) => {
      sandboxHistory.push({ epoch, loss, accuracy: acc });
      drawChart();
      printLog(`Epoch ${epoch}/${numEpochs} completed - Combined Hinge Loss: ${loss.toFixed(4)} | Training Acc: ${(acc*100).toFixed(1)}%`);
      // Pause slightly for a beautiful, smooth transition visual effect in the web browser
      await new Promise(r => setTimeout(r, 60));
    };

    // 4. Model specific execution
    if (activeSandboxModel === 'naive-bayes') {
      nbModel.fit(trainData, vectorizer);
      printLog(`Bayes probabilities calculated. Class Priors: Real: ${(nbModel.classPriors.real*100).toFixed(1)}%, Fake: ${(nbModel.classPriors.fake*100).toFixed(1)}%`);
      // Naive bayes trains in one step, fill simulated history
      sandboxHistory = [{ epoch: 1, loss: 0, accuracy: 1.0 }];
      drawChart();
    } else if (activeSandboxModel === 'logistic-regression') {
      await lrModel.fit(trainData, vectorizer, learningRate, numEpochs, regParam, epochCallback);
    } else if (activeSandboxModel === 'passive-aggressive') {
      await pacModel.fit(trainData, vectorizer, regParam, numEpochs, epochCallback);
    }

    // 5. Evaluate on test set
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    testData.forEach(item => {
      let pred;
      if (activeSandboxModel === 'naive-bayes') {
        pred = nbModel.predict(item.stemmedTokens);
      } else if (activeSandboxModel === 'logistic-regression') {
        pred = lrModel.predict(item.stemmedTokens, vectorizer);
      } else if (activeSandboxModel === 'passive-aggressive') {
        pred = pacModel.predict(item.stemmedTokens, vectorizer);
      }

      const isActualReal = item.label === 'real';
      const isPredictedReal = pred.label === 'real';

      if (isActualReal && isPredictedReal) tp++;
      else if (!isActualReal && !isPredictedReal) tn++;
      else if (!isActualReal && isPredictedReal) fp++;
      else if (isActualReal && !isPredictedReal) fn++;
    });

    // 6. Metrics computations
    const total = testData.length;
    const testAccuracy = (tp + tn) / total;
    const precision = tp + fp > 0 ? (tp / (tp + fp)) : 0;
    const recall = tp + fn > 0 ? (tp / (tp + fn)) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    // Update matrix display
    cellTp.textContent = tp;
    cellTn.textContent = tn;
    cellFp.textContent = fp;
    cellFn.textContent = fn;

    // Update stats text
    lblPrecision.textContent = precision.toFixed(2);
    lblRecall.textContent = recall.toFixed(2);
    lblF1.textContent = f1.toFixed(2);
    lblTestAccuracy.textContent = `${(testAccuracy * 100).toFixed(1)}%`;

    printLog(`Training cycles completed successfully. Test Accuracy: ${(testAccuracy*100).toFixed(1)}% | Precision: ${precision.toFixed(2)} | F1: ${f1.toFixed(2)}`, 'success');

    // 7. Enable buttons
    btnTrainSandbox.disabled = false;
    btnTrainSandbox.style.opacity = '1';

    // Retrain main models on full dataset to update active analyzer model weights
    if (activeSandboxModel === 'naive-bayes') {
      nbModel.fit(preprocessedDataset, vectorizer);
    } else if (activeSandboxModel === 'logistic-regression') {
      lrModel.fit(preprocessedDataset, vectorizer, learningRate, numEpochs, regParam);
    } else if (activeSandboxModel === 'passive-aggressive') {
      pacModel.fit(preprocessedDataset, vectorizer, regParam, numEpochs);
    }
  }

  // --- CANVAS CHART RENDERER ---
  function resizeCanvas() {
    const parent = canvasSandbox.parentNode;
    canvasSandbox.width = parent.clientWidth;
    canvasSandbox.height = parent.clientHeight;
  }

  function drawInitialChart() {
    const ctx = canvasSandbox.getContext('2d');
    const width = canvasSandbox.width;
    const height = canvasSandbox.height;
    
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText("Model Sandbox Idle. Execute training cycle to compile statistics.", width / 2, height / 2);
  }

  function drawChart() {
    if (sandboxHistory.length === 0) {
      drawInitialChart();
      return;
    }

    const ctx = canvasSandbox.getContext('2d');
    const width = canvasSandbox.width;
    const height = canvasSandbox.height;
    ctx.clearRect(0, 0, width, height);

    const paddingLeft = 40;
    const paddingRight = 15;
    const paddingTop = 20;
    const paddingBottom = 30;

    const graphWidth = width - paddingLeft - paddingRight;
    const graphHeight = height - paddingTop - paddingBottom;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = paddingTop + (graphHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(width - paddingRight, y);
      ctx.stroke();

      // Labels on y-axis
      ctx.fillStyle = 'var(--text-muted)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText((1.0 - 0.25 * i).toFixed(2), paddingLeft - 8, y + 3);
    }

    // Plot data points
    const epochs = sandboxHistory.length;
    const getX = (index) => paddingLeft + (graphWidth / Math.max(1, epochs - 1)) * index;
    const getY = (val) => paddingTop + graphHeight * (1.0 - Math.max(0, Math.min(1.0, val)));

    // 1. Draw Accuracy Line (Teal)
    ctx.beginPath();
    ctx.strokeStyle = 'var(--color-teal)';
    ctx.lineWidth = 2.5;
    sandboxHistory.forEach((pt, idx) => {
      const x = getX(idx);
      const y = getY(pt.accuracy);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 2. Draw Loss Line (Rose Red)
    ctx.beginPath();
    ctx.strokeStyle = 'var(--color-fake)';
    ctx.lineWidth = 2;
    sandboxHistory.forEach((pt, idx) => {
      const x = getX(idx);
      // Map Loss between 0 and 2.0 roughly
      const normLoss = pt.loss / 2.0;
      const y = getY(normLoss);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels at Bottom
    ctx.fillStyle = 'var(--text-muted)';
    ctx.textAlign = 'center';
    
    // Label intervals
    const step = Math.ceil(epochs / 6);
    sandboxHistory.forEach((pt, idx) => {
      if (idx === 0 || idx === epochs - 1 || idx % step === 0) {
        ctx.fillText(`Ep ${pt.epoch}`, getX(idx), height - paddingBottom + 18);
      }
    });

    // Legend
    ctx.textAlign = 'left';
    ctx.fillStyle = 'var(--color-teal)';
    ctx.fillRect(paddingLeft + 10, paddingTop - 12, 12, 6);
    ctx.fillText("Training Accuracy", paddingLeft + 28, paddingTop - 7);

    ctx.fillStyle = 'var(--color-fake)';
    ctx.fillRect(paddingLeft + 150, paddingTop - 12, 12, 6);
    ctx.fillText("Hinge Loss (Scaled)", paddingLeft + 168, paddingTop - 7);
  }

  // --- LOCAL DATASET MANAGEMENT ---
  function renderDatasetTable() {
    tableDatasetBody.innerHTML = '';
    
    // We show only the last 15 elements to avoid making the page huge, or render all with scrolling
    activeDataset.forEach((item) => {
      const row = document.createElement('tr');
      const isReal = item.label === 'real';
      
      row.innerHTML = `
        <td style="width: 100px;">
          <span class="badge ${isReal ? 'badge-real' : 'badge-fake'}">${item.label}</span>
        </td>
        <td style="color: var(--text-secondary); max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${item.text}
        </td>
      `;
      tableDatasetBody.appendChild(row);
    });

    lblDatasetSize.textContent = `${activeDataset.length} Samples`;
  }

  function handleAddSample() {
    const text = inputNewSample.value.trim();
    const label = selectNewLabel.value;

    if (text.length < 15) {
      alert("Please enter a valid sentence snippet (at least 15 characters).");
      return;
    }

    // Add to datasets
    const newRecord = { text, label };
    activeDataset.push(newRecord);
    
    // Extract tokens
    const analysis = window.nlp.analyzeText(text);
    preprocessedDataset.push({
      text,
      label,
      cleanedTokens: analysis.cleanedTokens,
      stemmedTokens: analysis.stemmedTokens
    });

    inputNewSample.value = '';
    renderDatasetTable();
    printLog(`Added custom sample record of type '${label.toUpperCase()}' into the sandbox corpus.`);
  }

  // --- SPOT-THE-FAKE GAME ENGINE ---
  function resetQuiz() {
    quizIndex = 0;
    quizScore = 0;
    
    containerQuizSummary.style.display = 'none';
    containerQuizGame.style.display = 'block';
    cardQuiz.classList.remove('flipped');

    loadQuizQuestion();
  }

  function loadQuizQuestion() {
    const q = quizQuestions[quizIndex];
    
    lblQuizProgress.textContent = `Headline ${quizIndex + 1} of ${quizQuestions.length}`;
    lblQuizScore.textContent = `Score: ${quizScore} / ${quizIndex}`;
    
    // Update progress bar width
    const pct = (quizIndex / quizQuestions.length) * 100;
    barQuizProgress.style.width = `${pct}%`;
    
    quizHeadlineText.textContent = `"${q.headline}"`;
    quizSource.textContent = `Context: ${q.source}`;
  }

  function submitQuizAnswer(userGuess) {
    const q = quizQuestions[quizIndex];
    const isCorrect = userGuess === q.label;
    
    if (isCorrect) quizScore++;

    // Format back face
    quizVerdictTitle.textContent = isCorrect ? "Correct!" : "Incorrect!";
    quizVerdictTitle.className = 'quiz-back-title ' + (isCorrect ? 'correct' : 'incorrect');

    const realBadgeClass = q.label === 'real' ? 'badge-real' : 'badge-fake';
    quizActualVerdict.textContent = q.label === 'real' ? 'Real News' : 'Fake News';
    quizActualVerdict.className = `badge ${realBadgeClass}`;

    quizExplanation.textContent = q.explanation;

    // Trigger visual flip
    cardQuiz.classList.add('flipped');
  }

  function advanceQuizQuestion() {
    quizIndex++;
    
    if (quizIndex < quizQuestions.length) {
      cardQuiz.classList.remove('flipped');
      // Wait for card flip animation before updating content text
      setTimeout(loadQuizQuestion, 300);
    } else {
      // Completed, display final summary screen
      containerQuizGame.style.display = 'none';
      containerQuizSummary.style.display = 'flex';
      
      lblSummaryScore.textContent = `${quizScore} / ${quizQuestions.length}`;
      
      const pct = (quizScore / quizQuestions.length) * 100;
      
      // Update Conic Gradient score wheel
      const summaryWheel = document.querySelector('.summary-score-wheel');
      summaryWheel.style.background = `conic-gradient(var(--color-real) ${pct}%, rgba(255, 255, 255, 0.05) ${pct}%)`;

      // Set text responses
      if (pct >= 85) {
        lblSummaryFeedback.textContent = "Outstanding! You have master-level fact-checking instincts and instantly spot linguistic tells.";
      } else if (pct >= 60) {
        lblSummaryFeedback.textContent = "Great job! You detected most misinformation elements, but keep an eye out for hidden clickbait hooks.";
      } else {
        lblSummaryFeedback.textContent = "A good attempt. Keep practicing and reviewing the guidelines in the learning center to improve.";
      }

      // Fill progress bar fully
      barQuizProgress.style.width = `100%`;
    }
  }

  // --- LAUNCH APPLICATION ---
  init();
});
