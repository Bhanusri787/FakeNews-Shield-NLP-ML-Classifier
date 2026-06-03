/**
 * NLP & Feature Extraction Module for Fake News Detection
 * Provides tokenization, stopword removal, stylometric analysis, and sentiment calculation.
 */

// Curated list of standard English stopwords
const STOPWORDS = new Set([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
  'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
  'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
  'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
  'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
  'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
  'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
  'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
  "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
  'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]);

// Clickbait patterns and corresponding reasons
const CLICKBAIT_PATTERNS = [
  { regex: /\b\d+\s+(things|ways|secrets|tricks|steps|reasons|facts)\b/i, label: 'Numbered Listicle Hook', desc: 'Lists are often used to oversimplify topics and drive clicks.' },
  { regex: /\b(you\s+won'?t\s+believe|what\s+happens\s+next|will\s+blow\s+your\s+mind)\b/i, label: 'Curiosity Gap Hook', desc: 'Intentionally withholding critical information to force a click.' },
  { regex: /\b(this\s+one\s+simple\s+trick|doctors?\s+hate|secret\s+the\s+industry)\b/i, label: 'Secret Formula Claim', desc: 'Claims an easy, secret solution to complex problems.' },
  { regex: /\b(shocking|unbelievable|miracle|magic|mind-blowing|jaw-dropping)\b/i, label: 'Extreme Sensationalism', desc: 'Uses emotionally charged adjectives to exaggerate significance.' },
  { regex: /\b(what\s+they\s+don'?t\s+want\s+you\s+to\s+know|hidden\s+truth|conspiracy|coverup|silenced)\b/i, label: 'Conspiratorial Tone', desc: 'Creates a false sense of forbidden knowledge or cover-ups.' },
  { regex: /\b(breaking\s+news|viral|must\s+watch|share\s+before\s+deleted)\b/i, label: 'Urgency Pressure', desc: 'Manufactures urgency to bypass critical thinking and force immediate sharing.' }
];

// Sentiment and Subjectivity Word Lists (Lexicon-based)
const BIAS_LEXICON = {
  subjective: new Set([
    'awesome', 'awful', 'terrible', 'horrible', 'fantastic', 'stupid', 'idiot', 'genius', 'fraud', 'scam',
    'outrageous', 'ridiculous', 'slams', 'blasts', 'demolishes', 'destroys', 'epic', 'historic', 'disaster',
    'catastrophe', 'treason', 'traitor', 'hero', 'savior', 'evil', 'corrupt', 'crooked', 'shameful', 'pathetic',
    'unprecedented', 'shocking', 'unbelievable', 'miracle', 'insane', 'nonsense', 'ludicrous', 'scandalous'
  ]),
  factual: new Set([
    'reported', 'announced', 'stated', 'according', 'study', 'research', 'published', 'officials', 'spokesperson',
    'indicated', 'confirmed', 'evidence', 'data', 'verified', 'documented', 'witnesses', 'records', 'observed'
  ])
};

/**
 * Perform basic suffix stripping to stem words (light Porter Stemmer approximation)
 */
function stemWord(word) {
  word = word.toLowerCase().trim();
  if (word.length <= 2) return word;
  
  // Plurals
  if (word.endsWith('sses')) word = word.slice(0, -2);
  else if (word.endsWith('ies')) word = word.slice(0, -3) + 'i';
  else if (word.endsWith('ss')) {}
  else if (word.endsWith('s') && !word.endsWith('us') && !word.endsWith('is') && !word.endsWith('as')) word = word.slice(0, -1);
  
  // Tenses & Derivations
  if (word.endsWith('eed')) {
    if (word.length > 4) word = word.slice(0, -1); // agreed -> agree, but feed -> feed
  } else if (word.endsWith('ing')) {
    word = word.slice(0, -3);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) word += 'e'; // creating -> create
  } else if (word.endsWith('ed')) {
    word = word.slice(0, -2);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) word += 'e';
  }
  
  return word;
}

/**
 * Counts syllables in a word for readability scoring
 */
function countSyllables(word) {
  word = word.toLowerCase().replace(/[^a-z]/g, '');
  if (word.length <= 2) return 1;
  
  // Count vowel sequences
  const matches = word.match(/[aeiouy]+/g);
  let count = matches ? matches.length : 0;
  
  // Adjust for common silent patterns
  if (word.endsWith('e') && !word.endsWith('le')) {
    count--; // silent e at end (e.g. rate, date)
  }
  
  return Math.max(1, count);
}

/**
 * Tokenizes text, removes punctuation, and outputs cleaned word arrays
 */
function tokenize(text) {
  if (!text) return [];
  // Match word boundaries with alphabetic characters plus apostrophes
  return text.toLowerCase()
    .replace(/[^\w\s']/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0 && /^[a-z']/.test(t));
}

/**
 * Full Text Analysis Pipeline
 */
function analyzeText(text) {
  if (!text || text.trim().length === 0) {
    return {
      wordCount: 0,
      sentenceCount: 0,
      charCount: 0,
      readabilityGrade: 0,
      stylometrics: {
        capsRatio: 0,
        exclamationCount: 0,
        questionCount: 0,
        avgWordLength: 0,
        allCapsWordsCount: 0
      },
      clickbaitScore: 0,
      clickbaitTriggers: [],
      subjectivityScore: 0,
      cleanedTokens: [],
      stemmedTokens: []
    };
  }

  const rawWords = tokenize(text);
  const wordCount = rawWords.length;
  const charCount = text.length;

  // Split into sentences (simple sentence boundaries: ., !, ?)
  const sentences = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 0);
  const sentenceCount = Math.max(1, sentences.length);

  // Readability analysis
  let totalSyllables = 0;
  rawWords.forEach(w => { totalSyllables += countSyllables(w); });
  // Flesch-Kincaid Grade Level formula
  let readabilityGrade = 0.39 * (wordCount / sentenceCount) + 11.8 * (totalSyllables / Math.max(1, wordCount)) - 15.59;
  readabilityGrade = Math.max(1, Math.min(20, parseFloat(readabilityGrade.toFixed(1))));

  // Stylometrics
  const letters = text.replace(/[^a-zA-Z]/g, '');
  const caps = text.replace(/[^A-Z]/g, '');
  const capsRatio = letters.length > 0 ? (caps.length / letters.length) : 0;

  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;

  let totalWordLen = 0;
  let allCapsWordsCount = 0;
  rawWords.forEach(w => {
    totalWordLen += w.length;
    if (w.length > 1 && w === w.toUpperCase() && !/^\d+$/.test(w)) {
      allCapsWordsCount++;
    }
  });
  const avgWordLength = wordCount > 0 ? parseFloat((totalWordLen / wordCount).toFixed(1)) : 0;

  // Clickbait Detection & Phrases Highlighting
  const clickbaitTriggers = [];
  CLICKBAIT_PATTERNS.forEach(pat => {
    let match;
    // Reset regex state if global
    pat.regex.lastIndex = 0;
    
    // We scan sentences for the pattern to pinpoint the issue
    sentences.forEach((sentence, sIdx) => {
      if (pat.regex.test(sentence)) {
        // Extract the exact matching text
        const matchText = sentence.match(pat.regex)[0];
        clickbaitTriggers.push({
          patternName: pat.label,
          description: pat.desc,
          matchedText: matchText,
          sentenceIndex: sIdx,
          fullSentence: sentence
        });
      }
    });
  });
  const clickbaitScore = Math.min(100, Math.round((clickbaitTriggers.length / Math.max(1, sentences.length)) * 100 + (exclamationCount > 1 ? 20 : 0)));

  // Subjectivity vs Objectivity Scoring
  let subjectiveWordCount = 0;
  let factualWordCount = 0;
  rawWords.forEach(w => {
    const stemmed = stemWord(w);
    if (BIAS_LEXICON.subjective.has(stemmed) || BIAS_LEXICON.subjective.has(w)) {
      subjectiveWordCount++;
    }
    if (BIAS_LEXICON.factual.has(stemmed) || BIAS_LEXICON.factual.has(w)) {
      factualWordCount++;
    }
  });

  // Calculate subjectivity ratio: higher subjective word concentration means higher subjectivity score (0-100)
  // Base baseline subjectivity on standard texts: normal writing has minor emotional adjectives.
  const biasFactor = (subjectiveWordCount) / Math.max(1, subjectiveWordCount + factualWordCount);
  const subjectivityScore = Math.min(100, Math.round(biasFactor * 100));

  // Process cleaned tokens for model consumption (remove stopwords)
  const cleanedTokens = rawWords.filter(w => !STOPWORDS.has(w));
  const stemmedTokens = cleanedTokens.map(w => stemWord(w));

  return {
    wordCount,
    sentenceCount,
    charCount,
    readabilityGrade,
    stylometrics: {
      capsRatio: parseFloat(capsRatio.toFixed(3)),
      exclamationCount,
      questionCount,
      avgWordLength,
      allCapsWordsCount
    },
    clickbaitScore,
    clickbaitTriggers,
    subjectivityScore,
    cleanedTokens,
    stemmedTokens
  };
}

// Export modules to window for browser use
window.nlp = {
  analyzeText,
  tokenize,
  stemWord,
  STOPWORDS
};
