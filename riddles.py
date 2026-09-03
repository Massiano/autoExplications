import os, json, time, threading, string, re, glob
import pipeline

DEFAULT_SETTINGS = {
    "wordlist": "ltwf", "upto_lesson": "12H", "gen_model": "openai/gpt-4o-mini", "guess_model": "", "judge_model": "", "max_violations": 0, "k_attempts": 3, "min_interval": 1.0, "temp": 0.7, "max_tokens": 2000,
    "prompts": {
        "forbidden_prompt": "List the giveaway terms for the {media_type} '{title}': every word of the title, main character names, actor/author/creator names, iconic invented terms, and place names unique to it. Output ONLY a comma-separated list of lowercase single words (split multi-word names into their words). No explanations.",
        "retell_prompt": "Retell the {media_type} '{title}' as a riddle, without giving away which {media_type} it is. Every single word of your output must come from the allowed word list below. Do not use any forbidden word. Do not use the title or any names. Keep it short, 3 to 6 sentences, plot and feel, so someone who knows the {media_type} could guess it.\nForbidden words: {forbidden}\nAllowed words: {vocab_list}\nOutput only the riddle text.",
        "recall_prompt": "This is a riddle describing a {media_type}. Which {media_type} is it? Text: '{text}'. Respond with ONLY the title, nothing else.",
        "recall_judge_prompt": "Target {media_type}: '{title}'. A reader guessed: '{guess}'. Is that the same {media_type} (ignore subtitles, articles, translations)? Answer exactly one word: yes or no.",
        "discover_prompt": "List {n} {media_type}s for a guessing game aimed at adult foreign-language learners: widely known internationally, family-friendly, with plots that can be retold in very simple words. {criteria}\nOutput ONLY one title per line. No numbering, no years, no explanations.",
        "suitability_prompt": "Assess the {media_type} '{title}' as material for a vocabulary-limited guessing riddle aimed at adult foreign-language learners. Respond ONLY with a JSON object, no markdown fences, with keys: popularity (1-5, how widely known globally), content_ok (true/false, false if sexual/graphic-violence/otherwise inappropriate for a general learning app), content_note (short string), retellability (1-5, how well the plot can be conveyed in very simple concrete words), retell_note (short string), verdict (one of: good, ok, poor).",
    },
}

DIR = None
LOCK = threading.Lock()
BUSY = {}
BATCHES = {}

def _p(name): return os.path.join(DIR, name + ".json")

def _load(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except Exception: return default
    return default

def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def apply_keys():
    for env, val in _load(_p("_keys"), {}).items():
        if val and env not in os.environ: os.environ[env] = val

def key_status():
    stored = _load(_p("_keys"), {})
    out = {}
    for name, p in pipeline.PROVIDERS.items():
        env = p["key_env"]
        out[env] = {"provider": name, "set": bool(os.environ.get(env) or stored.get(env)), "source": "env" if env in os.environ and env not in stored else ("saved" if stored.get(env) else "")}
    return out

def settings(): return {**DEFAULT_SETTINGS, **_load(_p("_settings"), {}), "prompts": {**DEFAULT_SETTINGS["prompts"], **_load(_p("_settings"), {}).get("prompts", {})}}

def load_riddle(rid): return _load(_p(rid), None)
def save_riddle(r): r["updated"] = time.strftime("%Y-%m-%d %H:%M:%S"); _save(_p(r["id"]), r)

def list_riddles():
    out = []
    for path in glob.glob(_p("r_*")):
        r = _load(path, None)
        if r: out.append(r)
    out.sort(key=lambda r: r.get("created", ""), reverse=True)
    return out

def lesson_sort_key(k):
    m = re.match(r"(\d+)(\w*)", str(k))
    return (int(m.group(1)) if m else 0, m.group(2) if m else str(k))

def wordlist_lessons(name):
    data = json.load(open(f"wordlists/{name}.json"))
    if "lessons" in data: return sorted(data["lessons"].keys(), key=lesson_sort_key)
    return []

def build_vocab(name, upto, extra, lemma_variants):
    vocab = set()
    for w in pipeline.seed_en(): vocab |= lemma_variants(w)
    if name:
        data = json.load(open(f"wordlists/{name}.json"))
        if "lessons" in data:
            keys = sorted(data["lessons"].keys(), key=lesson_sort_key)
            if upto in keys: keys = keys[:keys.index(upto) + 1]
            words = [v for k in keys for g in data["lessons"][k] for v in g]
        else:
            words = data if isinstance(data, list) else data.get("words", [])
    else: words = []
    for w in list(words) + list(extra or []):
        for tok in pipeline.tokenize(str(w)): vocab |= lemma_variants(tok)
    return vocab

def parse_words(text):
    toks = [t.strip().lower().strip(string.punctuation) for t in re.split(r"[,\n]", text or "")]
    return sorted({t for t in toks if t and t.replace("-", "").isalpha()})

def strip_json(s):
    s = re.sub(r"^```[a-z]*\s*|\s*```$", "", s.strip())
    m = re.search(r"\{.*\}", s, re.S)
    return m.group(0) if m else s

def llm(temp, cfg):
    return pipeline.make_rate_limited(pipeline.openrouter_raw(os.environ.get("OPENROUTER_API_KEY", ""), temp, max_tokens=cfg.get("max_tokens")), float(cfg.get("min_interval", 1.0)))

def cfg_of(r):
    s = settings()
    return {**s, **{k: v for k, v in (r.get("config") or {}).items() if v not in ("", None)}, "prompts": {**s["prompts"], **(r.get("config") or {}).get("prompts", {})}}

def log(r, msg): r.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")

def step_assess(r):
    cfg = cfg_of(r)
    raw = llm(0.3, cfg)(cfg["prompts"]["suitability_prompt"].format(media_type=r["media_type"], title=r["title"]), cfg.get("judge_model") or cfg["gen_model"])
    try: r["suitability"] = json.loads(strip_json(raw))
    except Exception: r["suitability"] = {"verdict": "unparsed", "raw": raw[:400]}
    log(r, f"assess: {r['suitability'].get('verdict')}")

def step_forbidden(r):
    cfg = cfg_of(r)
    raw = llm(0.3, cfg)(cfg["prompts"]["forbidden_prompt"].format(media_type=r["media_type"], title=r["title"]), cfg["gen_model"])
    r["forbidden"] = sorted(set(parse_words(raw)) | set(parse_words(r["title"])) | set(r.get("extra_forbidden") or []))
    log(r, f"forbidden: {len(r['forbidden'])} terms")

def step_generate(r):
    cfg = cfg_of(r)
    if not r.get("forbidden"): step_forbidden(r)
    lv, can = pipeline.lang_fns("en")
    vocab = build_vocab(cfg["wordlist"], cfg["upto_lesson"], r.get("learner_words"), lv)
    r["vocab_size"] = len({can(t) for t in vocab})
    vocab_list = ", ".join(sorted({can(t) for t in vocab}))
    best = None
    for i in range(int(cfg["k_attempts"])):
        text = llm(float(cfg["temp"]), cfg)(cfg["prompts"]["retell_prompt"].format(media_type=r["media_type"], title=r["title"], forbidden=", ".join(r["forbidden"]), vocab_list=vocab_list), cfg["gen_model"]).strip()
        outside, hits = _validate_text(text, vocab, r["forbidden"], lv, can)
        att = {"text": text, "outside": outside, "forbidden_hits": hits, "valid": not hits and len(outside) <= int(cfg["max_violations"])}
        r.setdefault("attempts", []).append(att)
        log(r, f"generate {i + 1}: {len(outside)} outside, {len(hits)} forbidden")
        if att["valid"]: best = att; break
        if best is None or len(outside) + 3 * len(hits) < len(best["outside"]) + 3 * len(best["forbidden_hits"]): best = att
    r["text"] = best["text"]
    r["validation"] = {"outside": best["outside"], "forbidden_hits": best["forbidden_hits"], "valid": best["valid"]}
    r["recall"] = None
    r["status"] = "valid" if best["valid"] else "generated"

def _validate_text(text, vocab, forbidden, lv, can):
    toks = pipeline.tokenize(text)
    fb = set(forbidden or [])
    hits = sorted({t for t in toks if t in fb or can(t) in fb})
    outside = sorted({can(t) for t in toks if t not in fb and not (lv(t) & vocab)})
    return outside, hits

def step_validate(r):
    cfg = cfg_of(r)
    lv, can = pipeline.lang_fns("en")
    vocab = build_vocab(cfg["wordlist"], cfg["upto_lesson"], r.get("learner_words"), lv)
    r["vocab_size"] = len({can(t) for t in vocab})
    outside, hits = _validate_text(r.get("text", ""), vocab, r.get("forbidden"), lv, can)
    valid = bool(r.get("text")) and not hits and len(outside) <= int(cfg["max_violations"])
    r["validation"] = {"outside": outside, "forbidden_hits": hits, "valid": valid}
    if r["status"] not in ("published", "rejected"): r["status"] = "valid" if valid else ("generated" if r.get("text") else "draft")
    log(r, f"validate: {len(outside)} outside, {len(hits)} forbidden -> {'valid' if valid else 'invalid'}")

def step_recall(r):
    cfg = cfg_of(r)
    gm = cfg.get("guess_model") or cfg["gen_model"]
    guess = llm(0.0, cfg)(cfg["prompts"]["recall_prompt"].format(media_type=r["media_type"], text=r.get("text", "")), gm).strip().strip(string.punctuation)
    hit = r["title"].lower() in guess.lower() or guess.lower() in r["title"].lower()
    if not hit:
        v = llm(0.0, cfg)(cfg["prompts"]["recall_judge_prompt"].format(media_type=r["media_type"], title=r["title"], guess=guess), cfg.get("judge_model") or gm).strip().lower()
        hit = v.startswith("yes")
    r["recall"] = {"guess": guess, "hit": hit, "model": gm}
    if hit and r["status"] == "valid": r["status"] = "tested"
    log(r, f"recall: '{guess}' -> {'HIT' if hit else 'miss'}")

STEPS = {"assess": step_assess, "forbidden": step_forbidden, "generate": step_generate, "validate": step_validate, "recall": step_recall}

def run_steps(rid, steps):
    r = load_riddle(rid)
    if not r: BUSY.pop(rid, None); return
    try:
        for s in steps:
            BUSY[rid] = s
            STEPS[s](r)
            save_riddle(r)
    except Exception as e:
        log(r, f"ERROR {s}: {e}"); save_riddle(r)
    BUSY.pop(rid, None)

def register(app, exp_dir):
    global DIR
    from flask import request, jsonify, send_from_directory
    DIR = os.path.join(exp_dir, "_riddles")
    os.makedirs(DIR, exist_ok=True)
    apply_keys()

    @app.route("/api/riddle_keys", methods=["GET"])
    def get_keys(): return jsonify(key_status())

    @app.route("/api/riddle_keys", methods=["POST"])
    def set_keys():
        body = request.get_json(force=True)
        stored = _load(_p("_keys"), {})
        valid_envs = {p["key_env"] for p in pipeline.PROVIDERS.values()}
        for env, val in body.items():
            if env not in valid_envs: continue
            val = (val or "").strip()
            if val == "": stored.pop(env, None); os.environ.pop(env, None)
            else: stored[env] = val; os.environ[env] = val
        _save(_p("_keys"), stored)
        return jsonify(key_status())

    @app.route("/riddles")
    def riddles_page(): return send_from_directory(".", "riddle_dashboard.html")

    @app.route("/api/riddle_settings", methods=["GET"])
    def get_settings(): return jsonify(settings())

    @app.route("/api/riddle_settings", methods=["POST"])
    def set_settings():
        _save(_p("_settings"), request.get_json(force=True)); return jsonify(settings())

    @app.route("/api/riddle_wordlists")
    def rwordlists():
        out = {}
        for path in glob.glob("wordlists/*.json"):
            name = os.path.basename(path).rsplit(".", 1)[0]
            try: out[name] = wordlist_lessons(name)
            except Exception: out[name] = []
        return jsonify(out)

    @app.route("/api/riddle_wordlists", methods=["POST"])
    def upload_wordlist():
        body = request.get_json(force=True)
        name = re.sub(r"\W+", "_", body.get("name", "")).strip("_")
        if not name: return jsonify({"error": "no name"}), 400
        content = body.get("content", "")
        try: data = json.loads(content)
        except Exception: data = parse_words(content)
        _save(f"wordlists/{name}.json", data)
        return jsonify({"name": name})

    @app.route("/api/riddles", methods=["GET"])
    def rlist():
        out = []
        for r in list_riddles():
            out.append({k: r.get(k) for k in ("id", "title", "media_type", "status", "text", "vocab_size", "updated")} | {"busy": BUSY.get(r["id"]), "suit": (r.get("suitability") or {}).get("verdict"), "valid": (r.get("validation") or {}).get("valid"), "recall_hit": (r.get("recall") or {}).get("hit"), "n_outside": len((r.get("validation") or {}).get("outside", [])), "n_fb": len((r.get("validation") or {}).get("forbidden_hits", []))})
        return jsonify(out)

    @app.route("/api/riddles", methods=["POST"])
    def rcreate():
        body = request.get_json(force=True)
        title = (body.get("title") or "").strip()
        if not title: return jsonify({"error": "no title"}), 400
        rid = time.strftime("r_%Y%m%d_%H%M%S_") + re.sub(r"\W+", "", title.lower())[:24]
        r = {"id": rid, "title": title, "media_type": body.get("media_type", "movie"), "status": "draft", "text": "", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {}, "learner_words": [], "extra_forbidden": [], "forbidden": [], "attempts": [], "log": [], "suitability": None, "validation": None, "recall": None}
        save_riddle(r)
        steps = body.get("steps") or []
        if steps: threading.Thread(target=run_steps, args=(rid, steps), daemon=True).start()
        return jsonify({"id": rid})

    @app.route("/api/riddles/<rid>", methods=["GET"])
    def rget(rid):
        r = load_riddle(rid)
        if not r: return jsonify({"error": "not found"}), 404
        return jsonify(r | {"busy": BUSY.get(rid)})

    @app.route("/api/riddles/<rid>", methods=["POST"])
    def rupdate(rid):
        r = load_riddle(rid)
        if not r: return jsonify({"error": "not found"}), 404
        body = request.get_json(force=True)
        for k in ("text", "title", "media_type", "status", "learner_words", "extra_forbidden", "forbidden", "config", "notes"):
            if k in body: r[k] = body[k]
        save_riddle(r)
        return jsonify(r | {"busy": BUSY.get(rid)})

    @app.route("/api/riddles/<rid>/delete", methods=["POST"])
    def rdelete(rid):
        if os.path.exists(_p(rid)): os.remove(_p(rid))
        BUSY.pop(rid, None)
        return jsonify({"deleted": rid})

    @app.route("/api/riddles/<rid>/run", methods=["POST"])
    def rrun(rid):
        if BUSY.get(rid): return jsonify({"error": "busy"}), 409
        steps = [s for s in request.get_json(force=True).get("steps", []) if s in STEPS]
        if not steps: return jsonify({"error": "no steps"}), 400
        BUSY[rid] = steps[0]
        threading.Thread(target=run_steps, args=(rid, steps), daemon=True).start()
        return jsonify({"running": steps})

    @app.route("/api/riddles/try", methods=["POST"])
    def rtry():
        body = request.get_json(force=True)
        step = body.get("step")
        if step not in STEPS and step != "discover": return jsonify({"error": "bad step"}), 400
        s = settings()
        cfg = {**s, **{k: v for k, v in body.get("config", {}).items() if v not in ("", None)}, "prompts": {**s["prompts"], **body.get("config", {}).get("prompts", {})}}
        t0 = time.time()
        try:
            if step == "discover":
                prompt = cfg["prompts"]["discover_prompt"].format(n=int(body.get("n", 8)), media_type=body.get("media_type", "movie"), criteria=body.get("criteria", ""))
                return jsonify({"raw": llm(0.8, cfg)(prompt, cfg["gen_model"]), "seconds": round(time.time() - t0, 1)})
            r = {"id": "_try", "title": body.get("title", "Finding Nemo"), "media_type": body.get("media_type", "movie"), "status": "draft", "text": body.get("text", ""), "config": body.get("config", {}), "learner_words": body.get("learner_words", []), "extra_forbidden": [], "forbidden": body.get("forbidden", []), "attempts": [], "log": [], "suitability": None, "validation": None, "recall": None}
            if step in ("generate", "validate", "recall") and not r["forbidden"] and step != "validate": step_forbidden(r)
            if step == "recall" and not r["text"]: step_generate(r)
            STEPS[step](r)
            return jsonify({k: r.get(k) for k in ("text", "forbidden", "suitability", "validation", "recall", "attempts", "log", "vocab_size")} | {"seconds": round(time.time() - t0, 1)})
        except Exception as e:
            return jsonify({"error": str(e), "seconds": round(time.time() - t0, 1)}), 502

    @app.route("/api/riddles/discover", methods=["POST"])
    def rdiscover():
        body = request.get_json(force=True)
        s = settings()
        media = body.get("media_type", "movie")
        n = min(int(body.get("n", 15)), 50)
        prompt = s["prompts"]["discover_prompt"].format(n=n, media_type=media, criteria=body.get("criteria", "").strip())
        try: raw = llm(0.8, s)(prompt, s["gen_model"])
        except Exception as e: return jsonify({"error": str(e)}), 502
        titles = []
        for line in raw.splitlines():
            t = re.sub(r"^\s*[\d\.\)\-\*]+\s*", "", line).strip().strip('"').strip()
            t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
            if t and len(t) < 90 and not t.lower().startswith(("here", "sure", "these")): titles.append(t)
        existing = {r["title"].casefold() for r in list_riddles()}
        created = []
        for t in titles:
            if t.casefold() in existing: continue
            existing.add(t.casefold())
            rid = time.strftime("r_%Y%m%d_%H%M%S_") + re.sub(r"\W+", "", t.lower())[:24] + f"_{len(created)}"
            r = {"id": rid, "title": t, "media_type": media, "status": "draft", "text": "", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {}, "learner_words": [], "extra_forbidden": [], "forbidden": [], "attempts": [], "log": [f"{time.strftime('%H:%M:%S')} discovered"], "suitability": None, "validation": None, "recall": None}
            save_riddle(r); created.append(rid)
        auto_reject = bool(body.get("auto_reject", True))
        if body.get("assess", True) and created:
            def work():
                for rid in created:
                    run_steps(rid, ["assess"])
                    r = load_riddle(rid)
                    su = r.get("suitability") or {}
                    if auto_reject and (su.get("verdict") == "poor" or su.get("content_ok") is False):
                        r["status"] = "rejected"; log(r, "auto-rejected by assessment"); save_riddle(r)
            threading.Thread(target=work, daemon=True).start()
        return jsonify({"created": created, "skipped": len(titles) - len(created)})

    @app.route("/api/riddles/batch", methods=["POST"])
    def rbatch():
        body = request.get_json(force=True)
        ids = [i for i in body.get("ids", []) if load_riddle(i)]
        steps = [s for s in body.get("steps", []) if s in STEPS]
        if not ids or not steps: return jsonify({"error": "need ids and steps"}), 400
        def work():
            for rid in ids:
                if not BUSY.get(rid): run_steps(rid, steps)
        threading.Thread(target=work, daemon=True).start()
        return jsonify({"batched": ids, "steps": steps})
