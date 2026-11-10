# Click Appeal Score — Title Feature Lexicons Documentation

**Researcher-Defined Baseline Lexicons**  
*Disclaimer: These lexicons are researcher-defined operational baselines compiled specifically for YouTube video title analysis in online challenge, entertainment, and lifestyle content. They are not claimed to be scientifically validated psychological dictionaries or exhaustive universal standards.*

---

## 1. Overview and Operational Definitions

In YouTube title feature extraction, lexical indicators are quantified as proportions relative to title length:
$$\text{Ratio} = \frac{\text{Count of matched occurrences}}{L}$$
where $L = \max(1, \text{number of title whitespace tokens})$.

---

## 2. Trigger Word Lexicons ($f_{trigger}$)

Trigger words are sensationalist, curiosity-inducing, superlative, competitive, or high-urgency terms designed to elicit strong cognitive arousal and prompt clicks.

### A. English Trigger Lexicon
- **Superlatives & Scale:** `best`, `worst`, `biggest`, `largest`, `highest`, `hardest`, `toughest`, `fastest`, `hottest`, `insane`, `extreme`, `ultimate`, `epic`, `impossible`, `unbeatable`, `undefeated`, `unbeaten`, `brutal`, `massive`, `giant`, `heaviest`, `spiciest`, `cheapest`, `expensive`, `poorest`, `record`
- **Competition & Stakes:** `challenge`, `contest`, `competition`, `tournament`, `compete`, `vs`, `versus`, `won`, `win`, `winner`, `winning`, `lose`, `lost`, `prize`, `jackpot`, `cash`, `standing`, `last to`, `first to`, `survive`, `survived`
- **Urgency, Discovery & Hook Phrases:** `viral`, `secret`, `secretly`, `proof`, `revealed`, `exposed`, `truth`, `shocking`, `never`, `ever`, `finally`, `now`, `hours`, `days`, `minutes`, `free`, `worth it`, `must`, `stop`, `warning`

### B. Hindi / Hinglish Trigger Lexicon
- **Scale, Superlatives & Impact:** `sabse` (सबसे - most), `bada` / `badi` (बड़ा / बड़ी - huge), `maha` (महा - mega), `asli` (असली - real), `nakli` (नकली - fake), `khatarnak` (खतरनाक - dangerous), `zabardast` (ज़बरदस्त - tremendous), `gazab` (गज़ब - astonishing), `dhamaka` (धमाका - explosive), `bawal` (बवाल - sensational)
- **Monetary & Competitive Stakes:** `crore` / `cr` (करोड़), `lakh` (लाख), `inam` (इनाम - prize), `jeet` (जीत - win), `har` (हार - loss), `shart` (शर्त - challenge/bet), `muqabla` (मुक़ाबला - contest), `khatam` (खत्म - finished)

---

## 3. Sensitive Word Lexicons ($f_{sensitive}$)

Sensitive words are terms associated with physical peril, taboo, harm, criminal deception, controversy, illness, or visceral distress.

### A. English Sensitive Lexicon
- **Harm, Mortality & Violence:** `die`, `died`, `death`, `dead`, `kill`, `killed`, `murder`, `murdered`, `blood`, `bloody`, `weapon`, `weapons`, `fight`, `gun`, `shooting`, `bomb`, `destroy`, `destroyed`, `crash`, `crashed`, `attack`, `assault`, `heart attack`
- **Illegality, Deception & Controversy:** `illegal`, `crime`, `criminal`, `scam`, `scammer`, `scammers`, `scammed`, `cheat`, `cheater`, `cheating`, `jail`, `prison`, `police`, `arrest`, `arrested`, `fired`, `broke into`, `steal`, `stolen`, `pirates`, `cult`, `racist`, `racists`, `dictatorship`, `dictator`, `trapped`, `lost mail`, `roadkill`
- **Distress, Disease & Bodily Ailments:** `danger`, `dangerous`, `emergency`, `hospital`, `failed`, `failure`, `sick`, `vomit`, `puke`, `pee`, `poop`, `suffering`, `heartbreaking`, `rock bottom`, `cried`, `cry`, `pain`, `poison`, `toxic`

### B. Hindi / Hinglish Sensitive Lexicon
- **Harm & Peril:** `maut` (मौत - death), `khoon` (खून - blood), `mar` (मार - strike/kill), `zeher` (ज़हर - poison), `barbaad` (बर्बाद - ruined), `bura` (बुरा - evil/bad), `paap` (पाप - sin)
- **Legal & Deceptive Distress:** `police` (पुलिस), `chor` (चोर - thief), `dhokha` (धोखा - betrayal/scam), `danga` (दंगा - violence/riot), `giraftar` (गिरफ्तार - arrest), `jail` (जेल)

---

## 4. Matching and Normalization Rules

1. Case-insensitivity: Matching operates on lowercased, punctuation-stripped candidate words and normalized multi-word phrases.
2. Boundary protection: Tokens are matched as discrete lexical units or defined multi-word idioms (e.g. `heart attack`, `last to`, `first to`, `rock bottom`).
3. Normalization denominator: Token count $L = \max(1, \text{number of whitespace tokens})$.
