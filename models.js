/**
 * Client-Side Machine Learning Classifiers for Fake News Detection
 * Contains TF-IDF Vectorizer, Naive Bayes, Logistic Regression, and Passive Aggressive Classifiers.
 */

class TfIdfVectorizer {
  constructor() {
    this.vocabulary = {}; // word -> index mapping
    this.idf = {};        // index -> idf value mapping
    this.vocabList = [];  // array of words aligned by index
  }

  /**
   * Builds vocabulary and calculates IDF from a corpus of documents
   * @param {Array<Array<string>>} docTokensList - Array of stemmed token lists
   */
  fit(docTokensList) {
    const docCount = docTokensList.length;
    const wordDocCounts = {}; // word -> number of docs containing word

    // Find all unique words and their document frequencies
    docTokensList.forEach(tokens => {
      const uniqueTokens = new Set(tokens);
      uniqueTokens.forEach(word => {
        wordDocCounts[word] = (wordDocCounts[word] || 0) + 1;
      });
    });

    // Build vocabulary index
    this.vocabulary = {};
    this.vocabList = [];
    let idx = 0;
    for (const word in wordDocCounts) {
      this.vocabulary[word] = idx;
      this.vocabList.push(word);
      
      // Calculate IDF with smoothing: log(1 + docCount / (1 + docCounts[word]))
      this.idf[idx] = Math.log(1 + (docCount / (1 + wordDocCounts[word])));
      idx++;
    }
  }

  /**
   * Transforms document tokens into a dense TF-IDF vector
   * @param {Array<string>} tokens - Array of stemmed tokens
   * @returns {Array<number>} Dense TF-IDF array aligned with vocabulary
   */
  transform(tokens) {
    const vector = new Array(this.vocabList.length).fill(0);
    if (tokens.length === 0) return vector;

    // Calculate term frequencies
    const tf = {};
    tokens.forEach(word => {
      if (word in this.vocabulary) {
        tf[word] = (tf[word] || 0) + 1;
      }
    });

    // Apply TF-IDF formula: tf * idf
    for (const word in tf) {
      const idx = this.vocabulary[word];
      // Term Frequency normalized by document length
      const termFreq = tf[word] / tokens.length;
      vector[idx] = termFreq * this.idf[idx];
    }

    return vector;
  }
}

class NaiveBayesClassifier {
  constructor() {
    this.vocabularySize = 0;
    this.classPriors = { real: 0.5, fake: 0.5 };
    this.wordLikelihoods = { real: {}, fake: {} };
    this.vocabWords = [];
  }

  /**
   * Train Naive Bayes model (using Laplace smoothing)
   */
  fit(trainData, vectorizer) {
    this.vocabWords = vectorizer.vocabList;
    this.vocabularySize = this.vocabWords.length;
    
    let realCount = 0;
    let fakeCount = 0;
    const realWordCounts = {};
    const fakeWordCounts = {};
    let realTotalWords = 0;
    let fakeTotalWords = 0;

    trainData.forEach(item => {
      const isReal = item.label === 'real';
      if (isReal) realCount++; else fakeCount++;

      item.stemmedTokens.forEach(word => {
        if (word in vectorizer.vocabulary) {
          if (isReal) {
            realWordCounts[word] = (realWordCounts[word] || 0) + 1;
            realTotalWords++;
          } else {
            fakeWordCounts[word] = (fakeWordCounts[word] || 0) + 1;
            fakeTotalWords++;
          }
        }
      });
    });

    // Calculate prior probabilities
    const totalDocs = trainData.length;
    this.classPriors.real = realCount / totalDocs;
    this.classPriors.fake = fakeCount / totalDocs;

    // Calculate likelihoods with Laplace smoothing
    this.wordLikelihoods = { real: {}, fake: {} };
    this.vocabWords.forEach(word => {
      const countReal = realWordCounts[word] || 0;
      const countFake = fakeWordCounts[word] || 0;

      // P(Word | Class) = (Count(Word in Class) + 1) / (Total Words in Class + Vocabulary Size)
      this.wordLikelihoods.real[word] = (countReal + 1) / (realTotalWords + this.vocabularySize);
      this.wordLikelihoods.fake[word] = (countFake + 1) / (fakeTotalWords + this.vocabularySize);
    });
  }

  /**
   * Predict probability and class label
   */
  predict(stemmedTokens) {
    // If empty tokens, return defaults
    if (stemmedTokens.length === 0) return { label: 'fake', confidence: 0.5, probabilities: { real: 0.5, fake: 0.5 } };

    // Calculate log-likelihoods to prevent underflow
    let logProbReal = Math.log(this.classPriors.real);
    let logProbFake = Math.log(this.classPriors.fake);

    stemmedTokens.forEach(word => {
      if (word in this.wordLikelihoods.real) {
        logProbReal += Math.log(this.wordLikelihoods.real[word]);
        logProbFake += Math.log(this.wordLikelihoods.fake[word]);
      }
    });

    // Convert back from log scale using soft probability mapping
    // To prevent numeric overflow, shift exponents
    const maxLog = Math.max(logProbReal, logProbFake);
    const expReal = Math.exp(logProbReal - maxLog);
    const expFake = Math.exp(logProbFake - maxLog);
    const sumExp = expReal + expFake;

    const probReal = expReal / sumExp;
    const probFake = expFake / sumExp;

    const label = probReal >= probFake ? 'real' : 'fake';
    const confidence = Math.max(probReal, probFake);

    return {
      label,
      confidence: parseFloat(confidence.toFixed(3)),
      probabilities: {
        real: parseFloat(probReal.toFixed(3)),
        fake: parseFloat(probFake.toFixed(3))
      }
    };
  }
}

class LogisticRegressionClassifier {
  constructor() {
    this.weights = [];
    this.bias = 0;
    this.featureCount = 0;
  }

  /**
   * Sigmoid activation function
   */
  sigmoid(z) {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z)))); // Bound z to prevent underflow/overflow
  }

  /**
   * Train Logistic Regression Classifier using Stochastic Gradient Descent (SGD)
   */
  async fit(trainData, vectorizer, learningRate = 0.1, epochs = 20, l2Reg = 0.01, onEpochEnd = null) {
    this.featureCount = vectorizer.vocabList.length;
    this.weights = new Array(this.featureCount).fill(0);
    this.bias = 0;

    // Transform documents to TF-IDF vectors
    const dataset = trainData.map(item => ({
      x: vectorizer.transform(item.stemmedTokens),
      y: item.label === 'real' ? 1 : 0
    }));

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let totalLoss = 0;

      // Shuffle training dataset
      dataset.sort(() => Math.random() - 0.5);

      dataset.forEach(item => {
        // Compute linear combination z = w.x + b
        let z = this.bias;
        for (let i = 0; i < this.featureCount; i++) {
          z += item.x[i] * this.weights[i];
        }

        // Apply Sigmoid to get probability
        const yPred = this.sigmoid(z);

        // Binary Cross-Entropy Loss
        const loss = -(item.y * Math.log(Math.max(1e-15, yPred)) + (1 - item.y) * Math.log(Math.max(1e-15, 1 - yPred)));
        totalLoss += loss;

        // Gradient Calculation
        const error = yPred - item.y;

        // Update weights and bias (incorporating L2 regularization)
        for (let i = 0; i < this.featureCount; i++) {
          const gradW = error * item.x[i] + l2Reg * this.weights[i];
          this.weights[i] -= learningRate * gradW;
        }
        this.bias -= learningRate * error;
      });

      const avgLoss = totalLoss / dataset.length;

      // Evaluate model accuracy on training data for live monitoring
      let correct = 0;
      dataset.forEach(item => {
        let z = this.bias;
        for (let i = 0; i < this.featureCount; i++) {
          z += item.x[i] * this.weights[i];
        }
        const yPred = this.sigmoid(z) >= 0.5 ? 1 : 0;
        if (yPred === item.y) correct++;
      });
      const trainAcc = correct / dataset.length;

      // Give visual updates back to UI
      if (onEpochEnd) {
        await onEpochEnd(epoch, parseFloat(avgLoss.toFixed(4)), parseFloat(trainAcc.toFixed(4)));
      }
    }
  }

  /**
   * Predict probability and class label
   */
  predict(stemmedTokens, vectorizer) {
    if (this.weights.length === 0) return { label: 'fake', confidence: 0.5, probabilities: { real: 0.5, fake: 0.5 } };
    
    const x = vectorizer.transform(stemmedTokens);
    let z = this.bias;
    for (let i = 0; i < this.featureCount; i++) {
      z += x[i] * this.weights[i];
    }

    const probReal = this.sigmoid(z);
    const probFake = 1 - probReal;

    const label = probReal >= 0.5 ? 'real' : 'fake';
    const confidence = label === 'real' ? probReal : probFake;

    return {
      label,
      confidence: parseFloat(confidence.toFixed(3)),
      probabilities: {
        real: parseFloat(probReal.toFixed(3)),
        fake: parseFloat(probFake.toFixed(3))
      }
    };
  }
}

class PassiveAggressiveClassifier {
  constructor() {
    this.weights = [];
    this.bias = 0;
    this.featureCount = 0;
  }

  /**
   * Train Passive Aggressive Classifier using online learning algorithm
   */
  async fit(trainData, vectorizer, aggressiveness = 1.0, epochs = 10, onEpochEnd = null) {
    this.featureCount = vectorizer.vocabList.length;
    this.weights = new Array(this.featureCount).fill(0);
    this.bias = 0;

    const dataset = trainData.map(item => ({
      x: vectorizer.transform(item.stemmedTokens),
      y: item.label === 'real' ? 1 : -1 // PAC uses -1 and +1 labels
    }));

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let totalLoss = 0;

      // Shuffle dataset
      dataset.sort(() => Math.random() - 0.5);

      dataset.forEach(item => {
        // Calculate prediction: y_hat = w.x + b
        let yHat = this.bias;
        for (let i = 0; i < this.featureCount; i++) {
          yHat += item.x[i] * this.weights[i];
        }

        // Hinge Loss: L = max(0, 1 - y * y_hat)
        const loss = Math.max(0, 1 - item.y * yHat);
        totalLoss += loss;

        if (loss > 0) {
          // Calculate step size tau: loss / (||x||^2 + 1 / (2 * C))
          let xNormSq = 0;
          for (let i = 0; i < this.featureCount; i++) {
            xNormSq += item.x[i] * item.x[i];
          }
          const tau = loss / (xNormSq + 1 / (2 * aggressiveness));

          // Update weights and bias
          for (let i = 0; i < this.featureCount; i++) {
            this.weights[i] += tau * item.y * item.x[i];
          }
          this.bias += tau * item.y;
        }
      });

      const avgLoss = totalLoss / dataset.length;

      // Evaluate accuracy
      let correct = 0;
      dataset.forEach(item => {
        let yHat = this.bias;
        for (let i = 0; i < this.featureCount; i++) {
          yHat += item.x[i] * this.weights[i];
        }
        const predY = yHat >= 0 ? 1 : -1;
        if (predY === item.y) correct++;
      });
      const trainAcc = correct / dataset.length;

      if (onEpochEnd) {
        await onEpochEnd(epoch, parseFloat(avgLoss.toFixed(4)), parseFloat(trainAcc.toFixed(4)));
      }
    }
  }

  /**
   * Predict probability and class label
   */
  predict(stemmedTokens, vectorizer) {
    if (this.weights.length === 0) return { label: 'fake', confidence: 0.5, probabilities: { real: 0.5, fake: 0.5 } };

    const x = vectorizer.transform(stemmedTokens);
    let yHat = this.bias;
    for (let i = 0; i < this.featureCount; i++) {
      yHat += x[i] * this.weights[i];
    }

    // Convert raw margin score to probability using standard platt scaling approximation
    const probReal = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, yHat * 2))));
    const probFake = 1 - probReal;

    const label = yHat >= 0 ? 'real' : 'fake';
    const confidence = label === 'real' ? probReal : probFake;

    return {
      label,
      confidence: parseFloat(confidence.toFixed(3)),
      probabilities: {
        real: parseFloat(probReal.toFixed(3)),
        fake: parseFloat(probFake.toFixed(3))
      }
    };
  }
}

// 45+ Diverse and engaging pre-baked real and fake news data
const BASELINE_DATASET = [
  // FAKE NEWS DATASET
  {
    label: 'fake',
    text: "SHOCKING: Doctors Can't Believe This One Miracle Fruit Cures Diabetes In 24 Hours! Pharma Companies Are Silencing This Secret!"
  },
  {
    label: 'fake',
    text: "NASA whistleblower exposes hidden alien city on the far side of the moon. Viral photos are deleted from public servers within hours!"
  },
  {
    label: 'fake',
    text: "Scientists confirm that drinking lemon juice and baking soda cure cancer completely. Big Pharma covers up the miracle recipe to protect profits!"
  },
  {
    label: 'fake',
    text: "URGENT WARNING: The government is deploying secret frequency waves to alter citizen mind patterns! Share this post before it's taken down!"
  },
  {
    label: 'fake',
    text: "Celebrity actor admits in leaked video that the Earth is actually flat and space exploration is a massive Hollywood CGI scam!"
  },
  {
    label: 'fake',
    text: "New study reveals that eating chocolate for every meal makes you lose 10 pounds a week. Nutritionists are furious over this simple trick!"
  },
  {
    label: 'fake',
    text: "Unbelievable! Leaked documents prove that a major technology giant is installing secret chips in drinking water to track everyone's movements."
  },
  {
    label: 'fake',
    text: "BREAKING: Global banks are shutting down all ATMs tomorrow at midnight. Get your cash out now! A massive financial coverup has begun!"
  },
  {
    label: 'fake',
    text: "This simple kitchen ingredient completely cures eyesight in three days! Doctors are shocked but they don't want you to know this secret."
  },
  {
    label: 'fake',
    text: "New energy device generates free power from thin air! The inventor went missing yesterday after oil executives threatened him."
  },
  {
    label: 'fake',
    text: "ALERT: Leaked footage reveals that historical landmarks are actually holographic projections built by secret societies to hide ancient ruins."
  },
  {
    label: 'fake',
    text: "A secret laboratory has successfully cloned prehistoric dinosaurs, and plans to release them in a private island theme park next month."
  },
  {
    label: 'fake',
    text: "Shocking investigation: Ancient pyramids in Egypt were actually giant wireless cell towers built by alien civilizations to power ships."
  },
  {
    label: 'fake',
    text: "Eat this one common weed from your backyard to instantly reverse aging! The cosmetic industry is trying to ban it from the market."
  },
  {
    label: 'fake',
    text: "Government agency admits they have the technology to control hurricanes and earthquakes, and are using them as secret weapons."
  },
  {
    label: 'fake',
    text: "This single cup of tea cleanses your lungs of all toxins in 10 minutes. Pharmaceutical companies are furious over this cheap home remedy."
  },
  {
    label: 'fake',
    text: "Billionaire tech founder arrested after secret server room containing human mind backups was uncovered in underground bunker."
  },
  {
    label: 'fake',
    text: "Ancient texts discovered in South America prove that humans lived alongside flying dragons just five hundred years ago. Read the truth!"
  },
  {
    label: 'fake',
    text: "Mysterious island suddenly appears in the Atlantic Ocean. Satellite images are being censored because of what is built on it!"
  },
  {
    label: 'fake',
    text: "Eating this raw vegetable prevents all viral infections. Health authorities are hiding the recipe to sell expensive vaccinations."
  },

  // REAL NEWS DATASET
  {
    label: 'real',
    text: "NASA's James Webb Space Telescope has captured a highly detailed image of the Ring Nebula, showing the intricate structure of the dying star."
  },
  {
    label: 'real',
    text: "Researchers from the University of Oxford have developed a new malaria vaccine that shows high efficacy in clinical trials with children."
  },
  {
    label: 'real',
    text: "The Federal Reserve announced an increase in interest rates by a quarter of a percentage point in its ongoing effort to stabilize inflation."
  },
  {
    label: 'real',
    text: "A team of marine biologists has discovered a new species of deep-sea jellyfish in the Mariana Trench, characterized by a glowing blue bioluminescence."
  },
  {
    label: 'real',
    text: "The World Health Organization reported a significant decline in global cases of tuberculosis due to improved screening and treatment programs."
  },
  {
    label: 'real',
    text: "SpaceX successfully launched its Falcon 9 rocket from Cape Canaveral, carrying 22 Starlink internet satellites into low Earth orbit."
  },
  {
    label: 'real',
    text: "A major archaeological study published in Nature confirms the discovery of a 4,000-year-old wooden structure in Zambia, altering historical timelines."
  },
  {
    label: 'real',
    text: "The European Union agreed on new legislation to mandate standard USB-C charging ports for all electronic devices by the end of next year."
  },
  {
    label: 'real',
    text: "Scientists at the Lawrence Livermore National Laboratory achieved net energy gain in a fusion reaction for the second time, improving yield."
  },
  {
    label: 'real',
    text: "Japan's lunar lander successfully transmitted data back to Earth after executing a precise landing near a crater on the moon's surface."
  },
  {
    label: 'real',
    text: "The international community signed a historic treaty to protect biodiversity in high seas, covering international waters outside country borders."
  },
  {
    label: 'real',
    text: "New clinical data suggests that a newly developed antibody treatment slows cognitive decline in patients with early-stage Alzheimer's disease."
  },
  {
    label: 'real',
    text: "An earthquake of magnitude 6.2 struck central Italy early Friday morning, causing minor damages to historical structures but no casualties."
  },
  {
    label: 'real',
    text: "The global automotive manufacturer announced plans to transition its entire European passenger vehicle lineup to fully electric models."
  },
  {
    label: 'real',
    text: "A peer-reviewed study in Science shows that reforestation efforts in sub-Saharan Africa have successfully restored millions of hectares of land."
  },
  {
    label: 'real',
    text: "The international space station crew completed a six-hour spacewalk to install upgraded solar arrays, improving power generation capability."
  },
  {
    label: 'real',
    text: "Central banks across Europe coordinated measures to address liquidity concerns, introducing short-term loans to stabilize regional lenders."
  },
  {
    label: 'real',
    text: "A newly discovered compound in broccoli shows potential in supporting cardiac health, according to a peer-reviewed research study."
  },
  {
    label: 'real',
    text: "Government statistics indicate that the unemployment rate fell to its lowest level in five years, driven by expansions in the tech and retail sectors."
  },
  {
    label: 'real',
    text: "The United Nations climate summit concluded with a joint declaration pledging to accelerate transition efforts toward renewable energy grids."
  }
];

// Helper to preprocess dataset
function preprocessDataset(dataset) {
  return dataset.map(item => {
    const analysis = window.nlp.analyzeText(item.text);
    return {
      text: item.text,
      label: item.label,
      cleanedTokens: analysis.cleanedTokens,
      stemmedTokens: analysis.stemmedTokens
    };
  });
}

// Export models and utilities
window.ml = {
  TfIdfVectorizer,
  NaiveBayesClassifier,
  LogisticRegressionClassifier,
  PassiveAggressiveClassifier,
  BASELINE_DATASET,
  preprocessDataset
};
