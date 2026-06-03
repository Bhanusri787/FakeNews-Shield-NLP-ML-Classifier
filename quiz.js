/**
 * Interactive News Quiz Dataset and Explanations
 * Offers challenging, realistic news items for user evaluation.
 */

const QUIZ_QUESTIONS = [
  {
    id: 1,
    headline: "HEALTH ALERT: New Miracle Weed Discovered In Backyards Reverses Aging and Wrinkles Instantly. Cosmetics Companies Are Panicking!",
    label: "fake",
    source: "Unverified Health Blog",
    explanation: "This is a classic fake health claim. Linguistic red flags include: (1) Extreme sensationalism ('Miracle Weed', 'Reverses Aging Instantly'), (2) Conspiracy framing ('Cosmetics Companies Are Panicking!'), (3) Excessive capitalization ('HEALTH ALERT'), and (4) Zero scientific citations or names of researchers/universities."
  },
  {
    id: 2,
    headline: "NASA's Voyager 1 spacecraft has resumed sending engineering updates to Earth after engineers bypassed a faulty microchip from 1977.",
    label: "real",
    source: "Scientific American / NASA",
    explanation: "This is a real news event. Standard hallmarks of reliable reporting here include: (1) Specific details (Voyager 1, microchip from 1977), (2) Objective, unemotional tone, (3) Plausible and verifiable scientific context, and (4) Focus on resolving a technical engineering issue without sensationalist hype."
  },
  {
    id: 3,
    headline: "SHOCKING: Leaked CCTV footage from the central bank reveals top officials loading boxes of gold into private unmarked helicopters at 3 AM.",
    label: "fake",
    source: "Social Media Post",
    explanation: "This is fake news designed to stir financial panic. Red flags include: (1) The clickbait opener 'SHOCKING:', (2) Relies on vague 'leaked footage' with no specific source, date, or names, (3) Emotional trigger words ('private unmarked helicopters', 'secret gold transport'), and (4) Extremely dramatic framing meant to go viral."
  },
  {
    id: 4,
    headline: "The European Central Bank lowered its key deposit rate by 25 basis points to 3.5% on Thursday, citing cooling inflation across the eurozone.",
    label: "real",
    source: "Reuters / Financial Times",
    explanation: "This is a real financial report. Notice: (1) Complete lack of sensationalism or exclamation marks, (2) Precise numbers (25 basis points, 3.5%), (3) Specific dates and entities (Thursday, European Central Bank), and (4) Focus on objective economic data rather than conspiracy or alarmism."
  },
  {
    id: 5,
    headline: "MUST SHARE: Scientists find that eating raw onion slices before bed creates a protective shield in the stomach that neutralizes all viruses.",
    label: "fake",
    source: "Alternative Health Forum",
    explanation: "This is fake medical advice. Cues include: (1) The viral call-to-action 'MUST SHARE', (2) Pseudoscience jargon ('protective shield', 'neutralizes all viruses'), (3) Exaggerated claims for a common food item, and (4) Lack of peer-reviewed journal citations or institutional names."
  },
  {
    id: 6,
    headline: "A new study in the Journal of Climate indicates that deep ocean currents in the North Atlantic have slowed by 15% over the past two decades.",
    label: "real",
    source: "Nature Climate Change / Associated Press",
    explanation: "This is a factual scientific report. Key indicators of credibility: (1) Cites a specific scientific journal ('Journal of Climate'), (2) Focuses on measurable data (15% slow, past two decades), and (3) Uses cautious, academic phrasing ('indicates', 'slowed by') instead of alarmist or definitive absolute claims."
  },
  {
    id: 7,
    headline: "CONFIRMED: The secret world treaty signed in Switzerland last week mandates the gradual replacement of all paper currencies with microchip implants by 2028.",
    label: "fake",
    source: "Conspiracy Digest",
    explanation: "This is a classic disinformation headline. Cues include: (1) High-authority confirmation claim ('CONFIRMED:'), (2) Vague events with massive implications ('secret world treaty signed in Switzerland'), (3) Appeals to fear (forced microchip implants), and (4) Zero verifiable signatures, official press releases, or mainstream coverage."
  },
  {
    id: 8,
    headline: "Sweden has officially closed its last remaining coal-fired power plant, meeting its carbon reduction targets two years ahead of the original schedule.",
    label: "real",
    source: "BBC News / Sweden Government Portal",
    explanation: "This is a real environmental milestone. Standard factual news properties: (1) Clear, objective, and neutral vocabulary, (2) Verifiable facts (Sweden's power grid, closing of coal plant), and (3) Focuses on public policy and timetables without emotional appeal or urgency hooks."
  }
];

window.quizData = {
  questions: QUIZ_QUESTIONS
};
