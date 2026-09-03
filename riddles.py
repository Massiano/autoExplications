import os, json, time, threading, string, re, glob
import pipeline

RIDDLE_PROMPTS = {
    "forbidden_prompt": "List the giveaway terms for the {media_type} '{title}': every word of the title, main character names, actor/author/creator names, iconic invented terms, and place names unique to it. Output ONLY a comma-separated list of lowercase single words (split multi-word names into their words). No explanations.",
    "retell_prompt": "Retell the {media_type} '{title}' as a riddle, without giving away which {media_type} it is. Every single word of your output must come from the allowed word list below. Do not use any forbidden word. Do not use the title or any names. Keep it short, 3 to 6 sentences, plot and feel, so someone who knows the {media_type} could guess it.\nForbidden words: {forbidden}\nAllowed words: {vocab_list}\nOutput only the riddle text.",
    "recall_prompt": "This is a riddle describing a {media_type}. Which {media_type} is it? Text: '{text}'. Respond with ONLY the title, nothing else.",
    "recall_judge_prompt": "Target {media_type}: '{title}'. A reader guessed: '{guess}'. Is that the same {media_type} (ignore subtitles, articles, translations)? Answer exactly one word: yes or no.",
    "suitability_prompt": "Assess the {media_type} '{title}' as material for a vocabulary-limited guessing riddle aimed at adult foreign-language learners. Respond ONLY with a JSON object, no markdown fences, with keys: popularity (1-5, how widely known globally), content_ok (true/false, false if sexual/graphic-violence/otherwise inappropriate for a general learning app), content_note (short string), retellability (1-5, how well the plot can be conveyed in very simple concrete words), retell_note (short string), verdict (one of: good, ok, poor).",
}

def lesson_sort_key(k):
    m = re.match(r"(\d+)(\w*)", k)
    return (int(m.group(1)), m.group(2))

def vocab_upto(wordlist, upto_lesson, extra_words, lemma_variants):
    vocab = set()
    for w in pipeline.seed_en(): vocab |= lemma_variants(w)
    if wordlist:
        data = json.load(open(f"wordlists/{wordlist}.json"))
        keys = sorted(data["lessons"].keys(), key=lesson_sort_key)
        if upto_lesson in keys: keys = keys[:keys.index(upto_lesson) + 1]
        for k in keys:
            for group in data["lessons"][k]:
                for variant in group:
                    for tok in pipeline.tokenize(variant): vocab |= lemma_variants(tok)
    for w in extra_words:
        for tok in pipeline.tokenize(w): vocab |= lemma_variants(tok)
    return vocab

def parse_forbidden(text):
    toks = [t.strip().lower().strip(string.punctuation) for t in re.split(r"[,\n]", text)]
    return sorted({t for t in toks if t and t.isalpha() and len(t) > 1})

def validate(text, vocab, forbidden, lemma_variants, canonical):
    toks = pipeline.tokenize(text)
    fb = set(forbidden)
    forbidden_hits = sorted({t for t in toks if t in fb or canonical(t) in fb})
    outside = sorted({canonical(t) for t in toks if t not in fb and not (lemma_variants(t) & vocab)})
    return outside, forbidden_hits

def strip_json(s):
    s = s.strip()
    if s.startswith("```"): s = re.sub(r"^```[a-z]*\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    return m.group(0) if m else s

RIDDLE_DIR = None
riddle_jobs = {}

def rpath(rid): return os.path.join(RIDDLE_DIR, rid + ".json")

def save_riddle(r):
    tmp = rpath(r["id"]) + ".tmp"
    with open(tmp, "w") as f: json.dump(r, f, indent=1)
    os.replace(tmp, rpath(r["id"]))

def load_riddles():
    out = []
    for p in sorted(glob.glob(os.path.join(RIDDLE_DIR, "*.json")), reverse=True):
        try: out.append(json.load(open(p)))
        except Exception: pass
    return out

def run_riddle(r, cfg):
    lemma_variants, canonical = pipeline.lang_fns(cfg.get("language", "en"))
    svc = pipeline.make_rate_limited(pipeline.openrouter_raw(os.environ["OPENROUTER_API_KEY"], cfg.get("temp", 0.7)), cfg.get("min_interval", 1.0))
    cold = pipeline.make_rate_limited(pipeline.openrouter_raw(os.environ["OPENROUTER_API_KEY"], 0.0), cfg.get("min_interval", 1.0))
    P = {**RIDDLE_PROMPTS, **cfg.get("prompts", {})}
    gen_model, guess_model, judge_model = cfg.get("gen_model", "openai/gpt-4o-mini"), cfg.get("guess_model") or cfg.get("gen_model", "openai/gpt-4o-mini"), cfg.get("judge_model") or cfg.get("gen_model", "openai/gpt-4o-mini")
    title, media = r["title"], r["media_type"]
    max_viol, k = int(cfg.get("max_violations", 0)), int(cfg.get("k_attempts", 5))
    def log(msg): r["log"].append(f"{time.strftime('%H:%M:%S')} {msg}"); save_riddle(r)
    try:
        if cfg.get("assess", True):
            raw = svc(P["suitability_prompt"].format(media_type=media, title=title), judge_model)
            try: r["suitability"] = json.loads(strip_json(raw))
            except Exception: r["suitability"] = {"verdict": "unparsed", "raw": raw[:400]}
            log(f"suitability: {r['suitability'].get('verdict')}")
        fb_raw = svc(P["forbidden_prompt"].format(media_type=media, title=title), gen_model)
        r["forbidden"] = sorted(set(parse_forbidden(fb_raw)) | set(parse_forbidden(title)) | set(cfg.get("extra_forbidden", [])))
        log(f"forbidden: {len(r['forbidden'])} terms")
        vocab = vocab_upto(cfg.get("wordlist"), cfg.get("upto_lesson"), cfg.get("learner_words", []), lemma_variants)
        r["vocab_size"] = len({canonical(t) for t in vocab})
        vocab_list = ", ".join(sorted({canonical(t) for t in vocab}))
        best = None
        for i in range(k):
            text = svc(P["retell_prompt"].format(media_type=media, title=title, forbidden=", ".join(r["forbidden"]), vocab_list=vocab_list), gen_model).strip()
            outside, hits = validate(text, vocab, r["forbidden"], lemma_variants, canonical)
            att = {"text": text, "violations": outside, "forbidden_hits": hits, "valid": len(hits) == 0 and len(outside) <= max_viol}
            r["attempts"].append(att)
            log(f"attempt {i + 1}: {len(outside)} outside, {len(hits)} forbidden -> {'valid' if att['valid'] else 'invalid'}")
            if att["valid"]:
                guess = cold(P["recall_prompt"].format(media_type=media, text=text), guess_model).strip().strip(string.punctuation)
                att["guess"] = guess
                ok = title.lower() in guess.lower() or guess.lower() in title.lower()
                if not ok:
                    verdict = cold(P["recall_judge_prompt"].format(media_type=media, title=title, guess=guess), judge_model).strip().lower()
                    ok = verdict.startswith("yes")
                att["guessed"] = ok
                log(f"attempt {i + 1} recall: '{guess}' -> {'HIT' if ok else 'miss'}")
                if ok: best = att; break
                if best is None: best = att
            save_riddle(r)
        r["best"] = best
        r["status"] = "done" if best and best.get("guessed") else ("weak" if best else "failed")
        log(f"final: {r['status']}")
    except Exception as e:
        r["status"] = "error"; r["error"] = str(e); log("ERROR " + str(e))
    save_riddle(r)

def register(app, exp_dir):
    global RIDDLE_DIR
    from flask import request, jsonify, send_from_directory
    RIDDLE_DIR = os.path.join(exp_dir, "_riddles")
    os.makedirs(RIDDLE_DIR, exist_ok=True)

    @app.route("/riddles")
    def riddle_page(): return send_from_directory(".", "riddle_dashboard.html")

    @app.route("/api/riddles", methods=["GET"])
    def riddle_list():
        return jsonify([{k: r.get(k) for k in ("id", "title", "media_type", "status", "vocab_size")} | {"suitability": (r.get("suitability") or {}).get("verdict"), "guessed": bool((r.get("best") or {}).get("guessed"))} for r in load_riddles()])

    @app.route("/api/riddles", methods=["POST"])
    def riddle_create():
        body = request.get_json(force=True)
        title = (body.get("title") or "").strip()
        if not title: return jsonify({"error": "no title"}), 400
        rid = time.strftime("r_%Y%m%d_%H%M%S_") + re.sub(r"\W+", "", title.lower())[:20]
        r = {"id": rid, "title": title, "media_type": body.get("media_type", "movie"), "status": "running", "config": body, "attempts": [], "log": [], "forbidden": [], "suitability": None, "best": None}
        save_riddle(r)
        threading.Thread(target=run_riddle, args=(r, body), daemon=True).start()
        return jsonify({"id": rid})

    @app.route("/api/riddles/<rid>")
    def riddle_get(rid):
        if not os.path.exists(rpath(rid)): return jsonify({"error": "not found"}), 404
        return jsonify(json.load(open(rpath(rid))))

    @app.route("/api/riddles/<rid>/delete", methods=["POST"])
    def riddle_delete(rid):
        if os.path.exists(rpath(rid)): os.remove(rpath(rid))
        return jsonify({"deleted": rid})

    @app.route("/api/riddle_lessons/<wl>")
    def riddle_lessons(wl):
        try: data = json.load(open(f"wordlists/{wl}.json"))
        except Exception as e: return jsonify({"error": str(e)}), 404
        return jsonify(sorted(data["lessons"].keys(), key=lesson_sort_key))
