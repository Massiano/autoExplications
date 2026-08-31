import string, time, os, json, requests, nltk, threading
from collections import Counter
from typing import Set, Dict, List, Tuple, Callable, Optional

for r in ("wordnet", "omw-1.4"):
    nltk.download(r, quiet=True)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

LLMService = Callable[[str, str], str]
ControlledVocab = Set[str]

lemmatizer = WordNetLemmatizer()
_ALL_POS = (wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV)
_APOS = str.maketrans({"'": " ", "\u2019": " ", "\u2018": " "})

def lemmatize_en_variants(token):
    variants = {token} | {lemmatizer.lemmatize(token, pos=p) for p in _ALL_POS}
    return {v for v in variants if len(v) > 1 or len(token) == 1}

def canonical_en(token):
    v = lemmatizer.lemmatize(token, pos=wordnet.VERB)
    if v != token and len(v) > 1: return v
    n = lemmatizer.lemmatize(token, pos=wordnet.NOUN)
    return n if len(n) > 1 or len(token) == 1 else token

LANGS = {"en": {"lemma_variants": lemmatize_en_variants, "canonical": canonical_en}}

def lang_fns(language):
    if language not in LANGS: raise ValueError(f"language '{language}' not implemented; available: {list(LANGS)}")
    return LANGS[language]["lemma_variants"], LANGS[language]["canonical"]

def tokenize(text):
    return [t for t in (raw.strip(string.punctuation) for raw in text.lower().translate(_APOS).split()) if t and (len(t) > 1 or t in ("a", "i"))]

def seed_en():
    function_words = {"a", "an", "the", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "us", "them", "someone", "something", "people",
                      "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
                      "can", "could", "will", "would", "shall", "should", "may", "might", "must", "not", "no", "yes",
                      "and", "or", "but", "if", "then", "because", "so", "than", "as", "like",
                      "of", "to", "in", "on", "at", "by", "for", "from", "with", "about", "into", "out", "up", "down", "off", "over", "under", "near", "around", "inside", "away",
                      "all", "some", "many", "few", "much", "one", "two", "other", "another", "each", "same", "more", "most", "less", "any", "every", "whole", "until", "even", "everything", "others",
                      "what", "who", "which", "when", "where", "why", "how", "very", "also", "too", "only", "again", "often", "sometimes", "always", "never", "usually", "now", "here", "there"}
    nsm_primes = {"want", "need", "know", "think", "feel", "see", "hear", "say", "word", "true",
                  "do", "happen", "make", "move", "touch", "go", "come", "get", "give", "take", "use", "help", "keep", "stay", "turn", "look", "find", "put", "let",
                  "live", "die", "eat", "grow", "there", "have",
                  "good", "bad", "big", "small", "long", "short", "old", "new",
                  "kind", "part", "side", "way", "thing", "body", "place", "time", "moment", "before", "after", "day", "night", "far", "close", "above", "below"}
    return function_words | nsm_primes

SEEDS = {"en": seed_en}

SCHEME_DESCRIPTION = (
    "1 BOOTSTRAP: for each target word, ask the model k_bootstrap times for a simple explanation (rotating bootstrap_prompts), no filtering. "
    "2 SHELL: per target, keep words appearing in >= min_share of its bootstrap texts (lemma-aware intersection); union these cores across targets; add the fixed seed (function words + NSM-like primes); words outside seed are marked scaffold (untaught debt). Re-apply previously admitted rounds, then admit the admit_top_n most frequent outside-shell words from accumulated counts (frontier), excluding words already in shell. "
    "3 MINE: for each unverified target, up to k_mine attempts: mine_prompt asks to rewrite one of its bootstrap texts using only shell words, without the target word. Checks: target-word use (reject), outside-shell words (reject + count into frontier). "
    "4 VERIFY: surviving text goes to a cold guesser (guess_prompt); lemma match = strict; otherwise a judge (judge_prompt) grades strict/almost/no. strict wins immediately; almost is kept while hunting strict; in-shell but unguessed texts are retained separately. Verified targets join the shell (acquisition order = curriculum)."
)

DEFAULT_PROMPTS = {"en": {
    "bootstrap_prompts": [
        "Explain the word '{word}' to a school kid, in simple everyday language. Use only very simple, common words. Do not use the word itself. Output only the explanation.",
        "Describe '{word}' so a child who never heard the word would understand it. Keep every word as plain and common as possible. Do not use the word itself. Output only the description.",
        "Say what '{word}' is, using the simplest and most common words you can. Short sentences. Do not use the word itself. Output only the text.",
    ],
    "mine_prompt": "Rewrite the following text so that a reader could still guess the word '{word}' from it. Every single word of your output must come from the allowed list. Shorten freely, drop or replace anything not on the list, restructure completely if needed. Do not use the word itself. Output only the rewritten text.\nText: {source}\nAllowed words: {vocab_list}",
    "guess_prompt": "What single word is this text getting at? Text: '{text}'. Respond with ONLY the word.",
    "judge_prompt": "Target word: '{target}'. A reader guessed: '{guessed}'. Did the reader correctly identify the target? Answer with exactly one word: 'strict' if it is the same word or a form of it, 'almost' if it is a synonym or very close in meaning, 'no' otherwise.",
}}

def make_rate_limited(svc, min_interval_seconds=1.0):
    last = {"t": 0.0}; lock = threading.Lock()
    def throttled(prompt, model):
        with lock:
            wait = min_interval_seconds - (time.monotonic() - last["t"])
            if wait > 0: time.sleep(wait)
            last["t"] = time.monotonic()
        return svc(prompt, model)
    return throttled

def make_cached(svc):
    cache = {}; lock = threading.Lock()
    def cached(prompt, model):
        key = (prompt, model)
        with lock:
            if key in cache: return cache[key]
        result = svc(prompt, model)
        with lock: cache[key] = result
        return result
    return cached

def openrouter_raw(api_key, temperature, max_retries=6):
    def call(prompt, model):
        backoff = 5.0
        for attempt in range(max_retries + 1):
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={"model": model, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}, timeout=120)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = float(resp.headers.get("Retry-After", backoff))
                time.sleep(min(wait, 300)); backoff = min(backoff * 2, 300); continue
            if resp.status_code >= 400:
                raise RuntimeError(f"openrouter http {resp.status_code} for model '{model}': {resp.text[:300]}")
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if isinstance(err, dict) and err.get("code") in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(backoff); backoff = min(backoff * 2, 300); continue
                raise RuntimeError(f"openrouter error: {err}")
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                if attempt < max_retries: time.sleep(2); continue
                raise RuntimeError("empty completion")
            return content
        raise RuntimeError("retries exhausted")
    return call

def empty_state():
    return {"schemes": {}, "bootstrap_texts": {}, "shell": {"seed": [], "admitted": [], "verified_order": [], "scaffold": []}, "controlled_vocab": [], "verified_explications": {}, "grades": {}, "strict_but_unguessed": {}, "misguesses": {}, "violation_counts": {}}

class StopRequested(Exception): pass

class Pipeline:
    def __init__(self, config, state, save_fn, log_fn, stop_fn):
        self.cfg = config; self.state = {**empty_state(), **state}
        self.save = lambda: save_fn(self.state); self.log = log_fn; self.stop_fn = stop_fn
        self.language = config.get("language", "en")
        self.lemma_variants, self.canonical = lang_fns(self.language)
        api_key = os.environ["OPENROUTER_API_KEY"]
        self._calls = 0
        def counted(svc):
            def f(prompt, model):
                self._calls += 1
                return svc(prompt, model)
            return f
        self._count = counted
        self.llm_sample = self._count(make_rate_limited(openrouter_raw(api_key, config.get("temp_sample", 0.6)), config.get("min_interval", 1.0)))
        self.llm_guess = make_cached(self._count(make_rate_limited(openrouter_raw(api_key, 0.0), config.get("min_interval", 1.0))))
        default_model = config.get("model", "openai/gpt-4o-mini")
        mcfg = config.get("models", {})
        def as_list(v): return v if isinstance(v, list) else [v]
        self.role_models = {role: as_list(mcfg.get(role, default_model)) for role in ("sample", "guess", "judge")}
        self._role_idx = {role: 0 for role in self.role_models}
        self.model = default_model
        self.prompts = {**DEFAULT_PROMPTS[self.language], **config.get("prompts", {})}

    def next_model(self, role):
        ms = self.role_models[role]
        m = ms[self._role_idx[role] % len(ms)]
        self._role_idx[role] += 1
        return m

    def _elapsed(self):
        m, s = divmod(int(time.monotonic() - getattr(self, "_t0", time.monotonic())), 60)
        return f"{m}m{s:02d}s, {self._calls} calls"

    def check_stop(self):
        if self.stop_fn(): raise StopRequested()

    def vocab_from_text(self, text):
        out = set()
        for tok in tokenize(text): out |= self.lemma_variants(tok)
        return out

    def display_vocab(self, vocab):
        return sorted({self.canonical(t) for t in vocab})

    def find_outside(self, text, vocab):
        return {tok for tok in tokenize(text) if not (self.lemma_variants(tok) & vocab)}

    def uses_target(self, text, word):
        target = self.lemma_variants(word.lower())
        return any(self.lemma_variants(tok) & target for tok in tokenize(text))

    def bootstrap(self, target_words):
        prompts = self.prompts["bootstrap_prompts"]
        texts = self.state["bootstrap_texts"]; k = self.cfg.get("k_bootstrap", 5)
        for word in target_words:
            self.check_stop()
            if word in texts and texts[word]: continue
            texts[word] = [self.llm_sample(prompts[i % len(prompts)].format(word=word), self.next_model("sample")).strip() for i in range(k)]
            self.log(f"[bootstrap] {word} done ({self._elapsed()})")
            self.save()

    def build_vocab(self, target_words):
        seed = SEEDS[self.language]()
        min_share = self.cfg.get("min_share", 0.8)
        vocab = set(seed)
        for word, samples in self.state["bootstrap_texts"].items():
            target = self.lemma_variants(word.lower())
            counts = Counter()
            for t in samples:
                for c in {self.canonical(tok) for tok in tokenize(t)}: counts[c] += 1
            needed = max(1, int(round(min_share * len(samples))))
            for c, n in counts.items():
                if n >= needed: vocab |= (self.lemma_variants(c) - target)
        scaffold = {self.canonical(t) for t in vocab} - {self.canonical(t) for t in seed}
        for batch in self.state["shell"]["admitted"]:
            for w in batch["words"]: vocab |= self.lemma_variants(w)
        top_n = self.cfg.get("admit_top_n", 0)
        if top_n > 0:
            ranked = [c for c, n in Counter(self.state["violation_counts"]).most_common() if n >= 2 and not (self.lemma_variants(c) & vocab)][:top_n]
            for c in ranked: vocab |= self.lemma_variants(c)
            if ranked:
                self.state["shell"]["admitted"].append({"round": len(self.state["shell"]["admitted"]) + 1, "words": ranked})
                self.log(f"[vocab] admitted round {len(self.state['shell']['admitted'])}: {ranked}")
        self.state["shell"]["seed"] = sorted({self.canonical(t) for t in seed})
        self.state["shell"]["scaffold"] = sorted(scaffold)
        self.log(f"[shell] seed {len(self.state['shell']['seed'])} scaffold {len(scaffold)} admitted {sum(len(b['words']) for b in self.state['shell']['admitted'])}")
        return vocab

    def guess(self, text):
        prompt = self.prompts["guess_prompt"].format(text=text)
        return self.llm_guess(prompt, self.next_model("guess")).strip().lower().strip(string.punctuation)

    def judge(self, target, guessed):
        prompt = self.prompts["judge_prompt"].format(target=target, guessed=guessed)
        v = self.llm_guess(prompt, self.next_model("judge")).strip().lower().strip(string.punctuation)
        return v if v in ("strict", "almost", "no") else "no"

    def grade(self, target, guessed):
        if self.lemma_variants(guessed) & self.lemma_variants(target.lower()): return "strict"
        return self.judge(target, guessed)

    def mine_word(self, word, vocab):
        vocab_list = ", ".join(self.display_vocab(vocab))
        sources = self.state["bootstrap_texts"].get(word, [])
        k = self.cfg.get("k_mine", 5)
        misguesses = []; best_grade = "no"; best_text = ""
        vc = self.state["violation_counts"]
        for i in range(k):
            self.check_stop()
            source = sources[i % len(sources)] if sources else ""
            prompt = self.prompts["mine_prompt"].format(word=word, source=source, vocab_list=vocab_list)
            text = self.llm_sample(prompt, self.next_model("sample")).strip()
            if self.uses_target(text, word): continue
            outside = self.find_outside(text, vocab)
            if outside:
                for v in outside: c = self.canonical(v); vc[c] = vc.get(c, 0) + 1
                continue
            if best_grade == "no": best_grade, best_text = "unguessed", text
            guessed = self.guess(text)
            g = self.grade(word, guessed)
            if g == "strict": return "strict", text, misguesses
            if g == "almost" and best_grade != "almost":
                best_grade, best_text = "almost", text; continue
            misguesses.append(guessed)
        return best_grade, best_text, misguesses

    def run(self, target_words):
        self._t0 = time.monotonic(); self._calls = 0
        self.state["schemes"] = {"bootstrap": "multi", "vocab": "intersect", "mine": "rewrite", "min_share": self.cfg.get("min_share", 0.8), "admit_top_n": self.cfg.get("admit_top_n", 0), "language": self.language, "models": self.role_models}
        self.state["scheme_description"] = SCHEME_DESCRIPTION
        self.state["prompts"] = self.prompts
        self.save()
        self.bootstrap(target_words)
        vocab = self.build_vocab(target_words)
        self.save()
        for word in target_words:
            self.check_stop()
            if word in self.state["verified_explications"]: continue
            grade, text, misguesses = self.mine_word(word, vocab)
            self.state["misguesses"][word] = self.state["misguesses"].get(word, []) + misguesses
            if grade in ("strict", "almost"):
                self.state["verified_explications"][word] = text
                self.state["grades"][word] = grade
                if word not in self.state["shell"]["verified_order"]: self.state["shell"]["verified_order"].append(word)
                self.state["strict_but_unguessed"].pop(word, None)
                vocab |= self.lemma_variants(word.lower())
                self.log(f"[mine] {word}: {grade} ({self._elapsed()})")
            elif grade == "unguessed" and text:
                self.state["strict_but_unguessed"][word] = text
                self.log(f"[mine] {word}: unguessed ({self._elapsed()})")
            else:
                self.log(f"[mine] {word}: failed ({self._elapsed()})")
            self.state["controlled_vocab"] = sorted(self.display_vocab(vocab))
            self.save()
        self.state["failed_words"] = [w for w in target_words if w not in self.state["verified_explications"]]
        self.state.setdefault("run_stats", []).append({"seconds": round(time.monotonic() - self._t0), "llm_calls": self._calls, "verified_total": len(self.state["verified_explications"])})
        self.log(f"[stats] {self._calls} llm calls, {round(time.monotonic() - self._t0)}s")
        self.state["top_violations"] = Counter(self.state["violation_counts"]).most_common(50)
        self.save()

def extract_ltwf_targets(path):
    data = json.load(open(path))
    drop = {"to", "of", "the", "a", "an", "in", "on", "at", "for", "with", "by", "is", "as", "same", "than", "if", "then", "about"}
    seedish = {c for c in seed_en()} | {"person", "one", "two", "three", "maybe", "its", "your", "my", "his", "her", "our", "their", "mine", "me", "him"}
    order = []
    for lesson, groups in data["lessons"].items():
        for g in groups:
            head = [w for w in g[0].lower().split() if w not in drop]
            if len(head) == 1 and head[0].isalpha():
                w = head[0]
                if w not in seedish and w not in order: order.append(w)
    return order
