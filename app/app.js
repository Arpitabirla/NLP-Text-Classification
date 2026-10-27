/* ============================================================
   NLP Explorer — app.js
   Full NLP pipeline: Tokenization, Stemming, Lemmatization,
   BOW, TF-IDF — all implemented in pure JavaScript
   ============================================================ */

"use strict";

// ── SAMPLE TEXTS ─────────────────────────────────────────────────────────────
const SAMPLES = [
  // Technology
  "Artificial intelligence and machine learning are rapidly transforming how computers process and understand human language. Deep learning algorithms are enabling machines to perform tasks that were previously thought to require human intelligence.",
  // Environment
  "Climate change is causing rising sea levels and extreme weather events across the globe. Scientists warn that urgent action on renewable energy and carbon emissions is needed to protect biodiversity and ecosystems.",
  // Sports
  "The football team practiced rigorously for months, running drills and studying their opponents' strategies. The players worked tirelessly, training daily to improve their passing, dribbling, and shooting skills.",
  // Health
  "Regular exercise and a balanced nutritional diet are essential for maintaining good physical and mental health. Doctors recommend at least thirty minutes of moderate activity daily to reduce the risk of chronic diseases.",
];

// ── ENGLISH STOP WORDS (subset) ───────────────────────────────────────────────
const STOP_WORDS = new Set([
  "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
  "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
  "it","its","itself","they","them","their","theirs","themselves","what","which",
  "who","whom","this","that","these","those","am","is","are","was","were","be",
  "been","being","have","has","had","having","do","does","did","doing","a","an",
  "the","and","but","if","or","because","as","until","while","of","at","by","for",
  "with","about","against","between","into","through","during","before","after",
  "above","below","to","from","up","down","in","out","on","off","over","under",
  "again","further","then","once","here","there","when","where","why","how","all",
  "both","each","few","more","most","other","some","such","no","nor","not","only",
  "own","same","so","than","too","very","s","t","can","will","just","don","should",
  "now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn",
  "hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
  "weren","won","wouldn","also","would","could","may"
]);

// ── PORTER STEMMER (JS Implementation) ───────────────────────────────────────
// Simplified Porter Stemmer — covers the most common English suffixes
const PorterStemmer = (() => {
  const step2list = {
    "ational":"ate","tional":"tion","enci":"ence","anci":"ance","izer":"ize",
    "abli":"able","alli":"al","entli":"ent","eli":"e","ousli":"ous",
    "ization":"ize","ation":"ate","ator":"ate","alism":"al","iveness":"ive",
    "fulness":"ful","ousness":"ous","aliti":"al","iviti":"ive","biliti":"ble"
  };
  const step3list = {
    "icate":"ic","ative":"","alize":"al","iciti":"ic","ical":"ic","ful":"","ness":""
  };

  function isVowel(c) { return "aeiou".includes(c); }

  function measure(stem) {
    let count = 0, inVowel = false;
    for (let i = 0; i < stem.length; i++) {
      if (isVowel(stem[i])) { inVowel = true; }
      else if (inVowel) { count++; inVowel = false; }
    }
    return count;
  }

  function containsVowel(stem) {
    for (const c of stem) if (isVowel(c)) return true;
    return false;
  }

  function endsDoubleC(stem) {
    return stem.length >= 2 && stem[stem.length-1] === stem[stem.length-2] &&
           !"aeioulv".includes(stem[stem.length-1]);
  }

  function stem(word) {
    word = word.toLowerCase();
    if (word.length < 3) return word;

    // Step 1a
    if      (word.endsWith("sses")) word = word.slice(0,-2);
    else if (word.endsWith("ies"))  word = word.slice(0,-2);
    else if (!word.endsWith("ss") && word.endsWith("s")) word = word.slice(0,-1);

    // Step 1b
    let flag = false;
    if (word.endsWith("eed")) {
      const stem1 = word.slice(0,-3);
      if (measure(stem1) > 0) word = word.slice(0,-1);
    } else if (word.endsWith("ed") && containsVowel(word.slice(0,-2))) {
      word = word.slice(0,-2); flag = true;
    } else if (word.endsWith("ing") && containsVowel(word.slice(0,-3))) {
      word = word.slice(0,-3); flag = true;
    }
    if (flag) {
      if      (word.endsWith("at") || word.endsWith("bl") || word.endsWith("iz")) word += "e";
      else if (endsDoubleC(word) && !"lsz".includes(word[word.length-1])) word = word.slice(0,-1);
      else if (measure(word) === 1 && word.length >= 3) {
        const [a,b,c] = [word[word.length-3],word[word.length-2],word[word.length-1]];
        if (!isVowel(a) && isVowel(b) && !isVowel(c) && !"wxwy".includes(c)) word += "e";
      }
    }

    // Step 1c
    if (word.endsWith("y") && containsVowel(word.slice(0,-1))) word = word.slice(0,-1) + "i";

    // Step 2
    for (const [suf, rep] of Object.entries(step2list)) {
      if (word.endsWith(suf)) {
        const base = word.slice(0, -suf.length);
        if (measure(base) > 0) { word = base + rep; break; }
      }
    }

    // Step 3
    for (const [suf, rep] of Object.entries(step3list)) {
      if (word.endsWith(suf)) {
        const base = word.slice(0, -suf.length);
        if (measure(base) > 0) { word = base + rep; break; }
      }
    }

    // Step 4
    const step4suffs = ["ement","ment","ance","ence","ion","ou","ism","ate",
                        "iti","ous","ive","ize","al","er","ic"];
    for (const suf of step4suffs) {
      if (word.endsWith(suf)) {
        const base = word.slice(0, -suf.length);
        if (measure(base) > 1) {
          if (suf === "ion" && !"st".includes(base[base.length-1])) continue;
          word = base; break;
        }
      }
    }

    // Step 5a
    if (word.endsWith("e")) {
      const base = word.slice(0,-1);
      const m = measure(base);
      if (m > 1 || (m === 1 && !endsDoubleC(base))) word = base;
    }

    // Step 5b
    if (word.endsWith("ll") && measure(word.slice(0,-1)) > 1) word = word.slice(0,-1);

    return word;
  }

  return { stem };
})();

// ── SNOWBALL STEMMER (JS — simplified aggressive version) ─────────────────────
const SnowballStemmer = {
  stem(word) {
    word = word.toLowerCase();
    if (word.length < 3) return word;
    // More aggressive — also strip -ly, -er, -ness, -ment, -ful, -tion etc.
    const suffixes = [
      ["ification","ifi"],["ifications","ifi"],["ization","ize"],
      ["fulness","ful"],["ousness","ous"],["iveness","ive"],["nesses",""],
      ["ments",""],["ment",""],["ness",""],["tion","t"],["tions","t"],
      ["ings",""],["ing",""],["edly",""],["edly",""],["edly",""],
      ["edly","ed"],["ness",""],["ful",""],["ous",""],["ive",""],
      ["ize",""],["ise",""],["ate",""],["ity",""],["ely",""],
      ["ness",""],["ers",""],["ly",""],["er",""],["ed",""],
      ["es",""],["s",""],["y","i"],
    ];
    for (const [suf, rep] of suffixes) {
      if (word.endsWith(suf) && word.length - suf.length >= 3) {
        return word.slice(0, -suf.length) + rep || word.slice(0, -suf.length);
      }
    }
    return word;
  }
};

// ── LEMMATIZER (JS — rule-based dictionary approach) ──────────────────────────
const Lemmatizer = (() => {
  // Common irregular forms → base form
  const LEMMA_DICT = {
    "running":"run","runs":"run","ran":"run",
    "swimming":"swim","swims":"swim","swam":"swim",
    "eating":"eat","eats":"eat","ate":"eat",
    "playing":"play","plays":"play","played":"play",
    "studying":"study","studies":"study","studied":"study",
    "going":"go","goes":"go","went":"go",
    "coming":"come","comes":"come","came":"come",
    "taking":"take","takes":"take","took":"take",
    "making":"make","makes":"make","made":"make",
    "saying":"say","says":"say","said":"say",
    "getting":"get","gets":"get","got":"get",
    "using":"use","uses":"use","used":"use",
    "seeing":"see","sees":"see","saw":"see",
    "knowing":"know","knows":"know","knew":"know",
    "trying":"try","tries":"try","tried":"try",
    "having":"have","has":"have","had":"have",
    "being":"be","was":"be","were":"be","is":"be","are":"be",
    "doing":"do","does":"do","did":"do",
    "thinking":"think","thinks":"think","thought":"think",
    "bringing":"bring","brings":"bring","brought":"bring",
    "finding":"find","finds":"find","found":"find",
    "working":"work","works":"work","worked":"work",
    "giving":"give","gives":"give","gave":"give",
    "looking":"look","looks":"look","looked":"look",
    "helping":"help","helps":"help","helped":"help",
    "talking":"talk","talks":"talk","talked":"talk",
    "keeping":"keep","keeps":"keep","kept":"keep",
    "waiting":"wait","waits":"wait","waited":"wait",
    "starting":"start","starts":"start","started":"start",
    "stopping":"stop","stops":"stop","stopped":"stop",
    "turning":"turn","turns":"turn","turned":"turn",
    "leaving":"leave","leaves":"leave","left":"leave",
    "feeling":"feel","feels":"feel","felt":"feel",
    "calling":"call","calls":"call","called":"call",
    "forming":"form","forms":"form","formed":"form",
    "processing":"process","processes":"process","processed":"process",
    "enabling":"enable","enables":"enable","enabled":"enable",
    "transforming":"transform","transforms":"transform","transformed":"transform",
    "understanding":"understand","understands":"understand","understood":"understand",
    "performing":"perform","performs":"perform","performed":"perform",
    "requiring":"require","requires":"require","required":"require",
    "training":"train","trains":"train","trained":"train",
    "learning":"learn","learns":"learn","learned":"learn",
    "improving":"improve","improves":"improve","improved":"improve",
    "causing":"cause","causes":"cause","caused":"cause",
    "protecting":"protect","protects":"protect","protected":"protect",
    "maintaining":"maintain","maintains":"maintain","maintained":"maintain",
    "reducing":"reduce","reduces":"reduce","reduced":"reduce",
    "practicing":"practice","practices":"practice","practiced":"practice",
    "studying":"study","studies":"study","studied":"study",
    "machines":"machine","algorithms":"algorithm","languages":"language",
    "computers":"computer","diseases":"disease","doctors":"doctor",
    "scientists":"scientist","strategies":"strategy","players":"player",
    "emissions":"emission","events":"event","tasks":"task","skills":"skill",
    "ecosystems":"ecosystem","levels":"level","months":"month","drills":"drill",
    "happily":"happy","rapidly":"rapid","tirelessly":"tireless","rigorously":"rigorous",
    "beautiful":"beautiful","essential":"essential","moderate":"moderate",
    "physical":"physical","mental":"mental","balanced":"balance",
    "nutritional":"nutrition","chronic":"chronic",
    "children":"child","people":"person","feet":"foot","teeth":"tooth",
    "mice":"mouse","geese":"goose","oxen":"ox","men":"man","women":"woman",
    "better":"good","best":"good","worse":"bad","worst":"bad",
    "more":"more","most":"most",
  };

  function lemmatize(word) {
    const lower = word.toLowerCase();
    if (LEMMA_DICT[lower]) return LEMMA_DICT[lower];

    // Rule-based fallback
    if (lower.endsWith("ies") && lower.length > 4) return lower.slice(0,-3) + "y";
    if (lower.endsWith("ves") && lower.length > 4) return lower.slice(0,-3) + "f";
    if (lower.endsWith("ing") && lower.length > 5) {
      const base = lower.slice(0,-3);
      // double consonant (running -> run)
      if (base.length >= 2 && base[base.length-1] === base[base.length-2]) return base.slice(0,-1);
      // magic-e (making -> make)
      return base + "e";
    }
    if (lower.endsWith("ed") && lower.length > 4) {
      const base = lower.slice(0,-2);
      if (base.length >= 2 && base[base.length-1] === base[base.length-2]) return base.slice(0,-1);
      return base + (base.endsWith("e") ? "" : "e");
    }
    if (lower.endsWith("s") && !lower.endsWith("ss") && lower.length > 3) return lower.slice(0,-1);

    return lower;
  }

  return { lemmatize };
})();

// ── TOKENIZER ─────────────────────────────────────────────────────────────────
function tokenize(text) {
  // Split on whitespace + handle common punctuation
  return text.match(/\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b|\d+|[^\s\w]/g) || [];
}

function sentenceTokenize(text) {
  return text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);
}

function cleanTokens(tokens) {
  return tokens
    .map(t => t.toLowerCase())
    .filter(t => /^[a-z]+$/.test(t) && !STOP_WORDS.has(t) && t.length > 1);
}

// ── TF Computation ─────────────────────────────────────────────────────────────
function computeTF(tokens) {
  const freq = {};
  for (const t of tokens) freq[t] = (freq[t] || 0) + 1;
  const total = tokens.length;
  const tf = {};
  for (const [word, count] of Object.entries(freq)) tf[word] = count / total;
  return tf;
}

// IDF against a small "reference corpus" (our 4 samples + input)
function computeIDF(wordList, corpus) {
  const idf = {};
  const N = corpus.length;
  for (const word of wordList) {
    const df = corpus.filter(doc => doc.includes(word)).length;
    idf[word] = Math.log(N / (1 + df)) + 1;
  }
  return idf;
}

// ── BOW ────────────────────────────────────────────────────────────────────────
function computeBOW(tokens) {
  const freq = {};
  for (const t of tokens) freq[t] = (freq[t] || 0) + 1;
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1]);
}

// ── MAIN ANALYSIS ─────────────────────────────────────────────────────────────
function analyzeText() {
  const text = document.getElementById("main-input").value.trim();
  if (!text) {
    showToast("Please enter some text first!");
    return;
  }
  if (text.split(" ").length < 5) {
    showToast("Please enter at least 5-6 words for meaningful analysis.");
    return;
  }

  // Tokenize
  const rawTokens  = tokenize(text);
  const sentences  = sentenceTokenize(text);
  const cleanToks  = cleanTokens(rawTokens);

  // Stemming & Lemmatization
  const porterStems   = cleanToks.map(t => PorterStemmer.stem(t));
  const snowballStems = cleanToks.map(t => SnowballStemmer.stem(t));
  const lemmas        = cleanToks.map(t => Lemmatizer.lemmatize(t));

  // BOW
  const bowData = computeBOW(cleanToks);

  // TF-IDF (against samples corpus)
  const corpus = SAMPLES.map(s => cleanTokens(tokenize(s)));
  corpus.push(cleanToks);
  const tf  = computeTF(cleanToks);
  const idf = computeIDF(cleanToks, corpus.map(c => c.join(" ")));

  const tfidfData = Object.entries(tf)
    .map(([word, tfScore]) => ({
      word,
      tf   : tfScore,
      idf  : idf[word] || 1,
      tfidf: tfScore * (idf[word] || 1),
    }))
    .sort((a, b) => b.tfidf - a.tfidf);

  // ── Render all sections ──
  renderTokenization(rawTokens, sentences, cleanToks);
  renderStemmingTable(cleanToks, porterStems, snowballStems, lemmas);
  renderBOW(bowData, cleanToks);
  renderTFIDF(tfidfData);
  renderComparison(bowData, tfidfData);

  // Show all hidden sections with animation
  const hiddenSections = document.querySelectorAll(".hidden-section");
  hiddenSections.forEach((s, i) => {
    s.style.display = "block";
    s.style.animation = `fadeInUp 0.6s ease ${i * 0.1}s both`;
    s.classList.remove("hidden-section");
  });

  // Scroll to tokenization
  setTimeout(() => {
    document.getElementById("tokenization-section")
      .scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

// ── RENDER TOKENIZATION ────────────────────────────────────────────────────────
function renderTokenization(rawTokens, sentences, cleanToks) {
  // Word tokens
  document.getElementById("word-token-count").textContent = `${rawTokens.length} tokens`;
  const wordDiv = document.getElementById("word-tokens-display");
  wordDiv.innerHTML = rawTokens.map(t => {
    const isPunct = /^[^\w\s]$/.test(t);
    return `<span class="token-tag ${isPunct ? "punct-tag" : ""}">${escHtml(t)}</span>`;
  }).join("");

  // Sentence tokens
  document.getElementById("sent-token-count").textContent = `${sentences.length} sentence${sentences.length !== 1 ? "s" : ""}`;
  const sentDiv = document.getElementById("sent-tokens-display");
  sentDiv.innerHTML = sentences.map((s, i) =>
    `<div class="sent-item"><span class="sent-num">S${i+1}.</span>${escHtml(s)}</div>`
  ).join("");

  // Clean tokens
  document.getElementById("clean-token-count").textContent =
    `${cleanToks.length} tokens (stopwords removed)`;
  const cleanDiv = document.getElementById("clean-tokens-display");
  cleanDiv.innerHTML = cleanToks.map(t =>
    `<span class="token-tag clean-tag">${escHtml(t)}</span>`
  ).join("");
}

// ── RENDER STEMMING TABLE ─────────────────────────────────────────────────────
function renderStemmingTable(original, porter, snowball, lemmas) {
  const tbody = document.getElementById("stem-table-body");
  tbody.innerHTML = original.map((word, i) =>
    `<tr>
      <td class="td-num">${i + 1}</td>
      <td class="td-orig">${escHtml(word)}</td>
      <td class="td-porter">${escHtml(porter[i] || word)}</td>
      <td class="td-snow">${escHtml(snowball[i] || word)}</td>
      <td class="td-lemma">${escHtml(lemmas[i] || word)}</td>
    </tr>`
  ).join("");
}

// ── RENDER BOW ────────────────────────────────────────────────────────────────
function renderBOW(bowData, tokens) {
  const top = bowData.slice(0, 12);
  const max = top[0]?.[1] || 1;

  // Bar chart
  const chartDiv = document.getElementById("bow-bar-chart");
  chartDiv.innerHTML = top.map(([word, count]) => {
    const pct = Math.round((count / max) * 100);
    return `
      <div class="bar-row">
        <div class="bar-label">${escHtml(word)}</div>
        <div class="bar-track">
          <div class="bar-fill bar-fill-bow" data-pct="${pct}" style="width:0%">${count}</div>
        </div>
      </div>`;
  }).join("");

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill-bow").forEach(el => {
      setTimeout(() => el.style.width = el.dataset.pct + "%", 50);
    });
  });

  // BOW vector
  const vectorDiv = document.getElementById("bow-vector-display");
  vectorDiv.innerHTML = top.map(([word, count]) =>
    `<div class="bow-vector-item">
      <span class="bow-word">${escHtml(word)}</span>
      <span class="bow-count">${count}</span>
    </div>`
  ).join("");
}

// ── RENDER TF-IDF ─────────────────────────────────────────────────────────────
function renderTFIDF(tfidfData) {
  const top   = tfidfData.slice(0, 15);
  const maxTf = Math.max(...top.map(d => d.tfidf));

  // Table
  const tbody = document.getElementById("tfidf-table-body");
  tbody.innerHTML = top.map((d, i) => {
    const pct = Math.round((d.tfidf / maxTf) * 100);
    return `<tr>
      <td class="td-num">${i+1}</td>
      <td class="td-orig">${escHtml(d.word)}</td>
      <td class="td-tf">${d.tf.toFixed(4)}</td>
      <td class="td-idf">${d.idf.toFixed(4)}</td>
      <td class="td-tfidf">${d.tfidf.toFixed(4)}</td>
      <td>
        <div class="importance-bar">
          <div class="importance-fill" style="width:${pct}%"></div>
        </div>
      </td>
    </tr>`;
  }).join("");

  // TF-IDF bar chart
  const chartDiv = document.getElementById("tfidf-bar-chart");
  chartDiv.innerHTML = top.slice(0, 10).map(d => {
    const pct = Math.round((d.tfidf / maxTf) * 100);
    return `
      <div class="bar-row">
        <div class="bar-label">${escHtml(d.word)}</div>
        <div class="bar-track">
          <div class="bar-fill bar-fill-tfidf" data-pct="${pct}"
               style="width:0%">${d.tfidf.toFixed(3)}</div>
        </div>
      </div>`;
  }).join("");

  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill-tfidf").forEach(el => {
      setTimeout(() => el.style.width = el.dataset.pct + "%", 50);
    });
  });
}

// ── RENDER COMPARISON ─────────────────────────────────────────────────────────
function renderComparison(bowData, tfidfData) {
  // BOW ranking (top 8)
  const bowDiv = document.getElementById("bow-ranking");
  bowDiv.innerHTML = bowData.slice(0, 8).map(([word, count], i) =>
    `<div class="rank-item bow-rank">
      <div class="rank-num">${i+1}</div>
      <div class="rank-word">${escHtml(word)}</div>
      <div class="rank-score">${count} × </div>
    </div>`
  ).join("");

  // TF-IDF ranking (top 8)
  const tfidfDiv = document.getElementById("tfidf-ranking");
  tfidfDiv.innerHTML = tfidfData.slice(0, 8).map((d, i) =>
    `<div class="rank-item tfidf-rank">
      <div class="rank-num">${i+1}</div>
      <div class="rank-word">${escHtml(d.word)}</div>
      <div class="rank-score">${d.tfidf.toFixed(3)}</div>
    </div>`
  ).join("");
}

// ── SAMPLE LOADER ─────────────────────────────────────────────────────────────
function loadSample(index) {
  const input = document.getElementById("main-input");
  input.value = SAMPLES[index];
  updateCharCount();
  input.focus();

  // Flash animation on textarea
  input.style.borderColor = "var(--purple)";
  input.style.boxShadow = "0 0 0 3px rgba(129,140,248,0.2)";
  setTimeout(() => {
    input.style.borderColor = "";
    input.style.boxShadow = "";
  }, 800);
}

// ── UTILITIES ─────────────────────────────────────────────────────────────────
function updateCharCount() {
  const text = document.getElementById("main-input").value;
  document.getElementById("char-count").textContent =
    `${text.length} characters · ${text.trim().split(/\s+/).filter(Boolean).length} words`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  toast.style.cssText = `
    position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
    background:#1e293b; color:#f0f4ff; border:1px solid rgba(129,140,248,0.3);
    padding:12px 24px; border-radius:100px; font-size:0.875rem; font-weight:500;
    box-shadow:0 8px 30px rgba(0,0,0,0.4); z-index:1000;
    animation:fadeInUp 0.3s ease;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── EVENT LISTENERS ────────────────────────────────────────────────────────────
document.getElementById("main-input").addEventListener("input", updateCharCount);

// Allow Enter+Ctrl to analyze
document.getElementById("main-input").addEventListener("keydown", e => {
  if (e.ctrlKey && e.key === "Enter") analyzeText();
});

// Smooth scroll for nav links
document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    link.classList.add("active");
  });
});

// ── INIT ───────────────────────────────────────────────────────────────────────
// Pre-fill with a sample on load
window.addEventListener("DOMContentLoaded", () => {
  loadSample(0);
  updateCharCount();
});
