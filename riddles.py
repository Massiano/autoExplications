import os, json, time, threading, string, re, glob
import pipeline

DEFAULT_SETTINGS = {
    "wordlist": "ltwf", "upto_lesson": "12H", "gen_model": "openai/gpt-4o-mini", "guess_model": "", "judge_model": "", "max_violations": 0, "k_attempts": 3, "min_interval": 1.0, "temp": 0.7, "max_tokens": 2000,
    "llm_params": {"top_p": "", "frequency_penalty": "", "presence_penalty": "", "seed": "", "stop": "", "max_retries": 3, "timeout": 120},
    "extra_params": {},
    "prompts": {
        "forbidden_prompt": "List the giveaway terms for the {media_type} '{title}': every word of the title, main character names, actor/author/creator names, iconic invented terms, and place names unique to it. Output ONLY a comma-separated list of lowercase single words (split multi-word names into their words). No explanations.",
        "retell_prompt": "Retell the {media_type} '{title}' as a riddle, without giving away which {media_type} it is. Every single word of your output must come from the allowed word list below. Do not use any forbidden word. Do not use the title or any names. Keep it short, 3 to 6 sentences, plot and feel, so someone who knows the {media_type} could guess it.\nForbidden words: {forbidden}\nAllowed words: {vocab_list}\nOutput only the riddle text.",
        "recall_prompt": "This is a riddle describing a {media_type}. Which {media_type} is it? Text: '{text}'. Respond with ONLY the title, nothing else.",
        "recall_judge_prompt": "Target {media_type}: '{title}'. A reader guessed: '{guess}'. Is that the same {media_type} (ignore subtitles, articles, translations)? Answer exactly one word: yes or no.",
        "discover_prompt": "List {n} {media_type}s for a guessing game aimed at adult foreign-language learners: widely known internationally, family-friendly, with plots that can be retold in very simple words. {criteria}\nOutput ONLY one title per line. No numbering, no years, no explanations.",
        "annot_topic_prompt": "Assign each word exactly ONE broad thematic topic label (lowercase, one word or hyphenated, e.g. food, family, travel, body, work, nature, emotion, time, number, function-word). Words: {words}\nRespond ONLY with a JSON object mapping each word to its topic. No fences, no comments.",
        "annot_level_prompt": "Estimate the CEFR level (A1, A2, B1, B2, C1, C2) at which a foreign learner of English typically acquires each word. Words: {words}\nRespond ONLY with a JSON object mapping each word to its level.",
        "annot_concreteness_prompt": "Rate the concreteness of each word from 1 (fully abstract) to 5 (fully concrete, picturable object or action). Words: {words}\nRespond ONLY with a JSON object mapping each word to an integer 1-5.",
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    data = load_wordlist(name)
    if isinstance(data, dict) and "lessons" in data: return sorted(data["lessons"].keys(), key=lesson_sort_key)
    return []

def build_vocab(name, upto, extra, lemma_variants):
    vocab = set()
    for w in pipeline.seed_en(): vocab |= lemma_variants(w)
    words = wordlist_words(name, upto) if name else []
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
    lp = {**DEFAULT_SETTINGS["llm_params"], **(cfg.get("llm_params") or {})}
    params = {k: (float(lp[k]) if k in ("top_p", "frequency_penalty", "presence_penalty") else (int(lp[k]) if k == "seed" else ([s.strip() for s in str(lp[k]).split(",") if s.strip()] if k == "stop" else lp[k]))) for k in ("top_p", "frequency_penalty", "presence_penalty", "seed", "stop") if str(lp.get(k, "")).strip() != ""}
    params.update(cfg.get("extra_params") or {})
    return pipeline.make_rate_limited(pipeline.openrouter_raw(os.environ.get("OPENROUTER_API_KEY", ""), temp, max_retries=int(lp.get("max_retries", 3)), max_tokens=cfg.get("max_tokens"), params=params), float(cfg.get("min_interval", 1.0)))

def list_dir(): return os.path.join(DIR, "_lists")

def wordlist_path(name):
    up = os.path.join(list_dir(), name + ".json")
    return up if os.path.exists(up) else f"wordlists/{name}.json"

def load_wordlist(name): return json.load(open(wordlist_path(name)))

def wordlist_words(name, upto=None):
    data = load_wordlist(name)
    if isinstance(data, dict) and "lessons" in data:
        keys = sorted(data["lessons"].keys(), key=lesson_sort_key)
        if upto in keys: keys = keys[:keys.index(upto) + 1]
        return [v for k in keys for g in data["lessons"][k] for v in g]
    return data if isinstance(data, list) else data.get("words", [])

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

def parse_source(content, loader, column=None):
    content = content.strip()
    if loader == "auto":
        try:
            d = json.loads(content)
            return d if (isinstance(d, dict) and "lessons" in d) else [str(w) for w in (d if isinstance(d, list) else d.get("words", []))]
        except Exception:
            loader = "csv" if ("," in content.splitlines()[0] or ";" in content.splitlines()[0]) and len(content.splitlines()) > 1 else "plain"
    if loader == "ltwf": return json.loads(content)
    if loader == "plain": return parse_words(content)
    if loader == "csv":
        import csv, io
        sniff = csv.Sniffer()
        try: dialect = sniff.sniff(content[:2000])
        except Exception: dialect = csv.excel
        rows = list(csv.reader(io.StringIO(content), dialect))
        header = rows[0]
        idx = 0
        if column not in (None, ""):
            if str(column).isdigit(): idx = int(column)
            elif column in header: idx = header.index(column)
        has_header = column in header if column else not all(c.replace("-", "").isalpha() and c.islower() for c in [r[idx] for r in rows[:5] if len(r) > idx])
        vals = [r[idx].strip() for r in (rows[1:] if has_header else rows) if len(r) > idx and r[idx].strip()]
        return sorted({v.lower() for v in vals if re.fullmatch(r"[a-zA-Z][a-zA-Z' -]*", v)})
    raise ValueError("unknown loader")

def annot_path(name): return os.path.join(DIR, "_annot", re.sub(r"\W+", "_", name) + ".json")
def load_annot(name): return _load(annot_path(name), {})
ANNOT_JOBS = {}

def canonical_words(name):
    lv, can = pipeline.lang_fns("en")
    out = []
    for w in wordlist_words(name):
        for tok in pipeline.tokenize(str(w)):
            c = can(tok)
            if c and c not in out: out.append(c)
    return sorted(set(out))

def run_annotators(name, annotators, model, cfg):
    words = canonical_words(name)
    ann = load_annot(name)
    job = ANNOT_JOBS[name] = {"done": 0, "total": len(words) * len(annotators), "error": ""}
    errs = []
    if "pos" in annotators:
        try:
            import nltk
            try: nltk.pos_tag(["test"])
            except LookupError: nltk.download("averaged_perceptron_tagger_eng", quiet=True); nltk.download("averaged_perceptron_tagger", quiet=True)
            for w, tag in nltk.pos_tag(words): ann.setdefault(w, {})["pos"] = tag
            _save(annot_path(name), ann)
        except Exception as e: errs.append(f"pos: {e}")
        job["done"] += len(words)
    pk = {"topic": "annot_topic_prompt", "level": "annot_level_prompt", "concreteness": "annot_concreteness_prompt"}
    svc = llm(0.2, cfg)
    for a in annotators:
        if a not in pk: continue
        try:
            todo = [w for w in words if a not in ann.get(w, {})]
            for i in range(0, len(todo), 40):
                batch = todo[i:i + 40]
                raw = svc(cfg["prompts"][pk[a]].format(words=", ".join(batch)), model)
                try: m = json.loads(strip_json(raw))
                except Exception: m = {}
                for w in batch:
                    v = m.get(w, m.get(w.capitalize()))
                    if v is not None: ann.setdefault(w, {})[a] = v
                job["done"] += len(batch)
                _save(annot_path(name), ann)
            job["done"] += len(words) - len(todo)
        except Exception as e: errs.append(f"{a}: {str(e)[:200]}"); job["done"] = job["total"]
    _save(annot_path(name), ann)
    if errs: job["error"] = "; ".join(errs); threading.Timer(20, lambda: ANNOT_JOBS.pop(name, None)).start()
    else: ANNOT_JOBS.pop(name, None)

def apply_filters(rows, filters):
    def ok(r):
        for f in filters or []:
            v = r["annot"].get(f["col"]) if f["col"] != "word" else r["word"]
            s, fv = str(v).lower() if v is not None else "", str(f.get("val", "")).lower()
            op = f.get("op", "contains")
            if op == "contains" and fv not in s: return False
            if op == "eq" and s != fv: return False
            if op == "ne" and s == fv: return False
            if op in ("lte", "gte"):
                try:
                    if op == "lte" and float(v) > float(fv): return False
                    if op == "gte" and float(v) < float(fv): return False
                except Exception: return False
            if op == "set" and v is None: return False
            if op == "unset" and v is not None: return False
        return True
    return [r for r in rows if ok(r)]

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

WCTL = {}

def wpath(wid): return os.path.join(DIR, "_workloads", wid + ".json")
def load_wl(wid): return _load(wpath(wid), None)
def save_wl(w): w["updated"] = time.strftime("%Y-%m-%d %H:%M:%S"); _save(wpath(w["id"]), w)

def list_wls():
    out = [w for w in (_load(p, None) for p in glob.glob(os.path.join(DIR, "_workloads", "w_*.json"))) if w]
    out.sort(key=lambda w: w.get("created", ""), reverse=True)
    return out

def wlog(w, msg): w.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}"); w["log"] = w["log"][-400:]

def wl_counts(w):
    c = {"total": len(w["riddle_ids"]), "pending": len(w["pending"]), "rejected": 0, "generated": 0, "valid": 0, "tested": 0, "draft": 0, "published": 0}
    for rid in w["riddle_ids"]:
        r = load_riddle(rid)
        if r and r["status"] in c: c[r["status"]] += 1
    return c

def run_workload(wid):
    w = load_wl(wid)
    ctl = WCTL.setdefault(wid, {"pause": False, "stop": False})
    spec = w["spec"]
    try:
        if spec.get("mode", "discover") == "discover" and not w["riddle_ids"]:
            wlog(w, "discovering titles"); save_wl(w)
            s = settings()
            raw = llm(0.8, s)(s["prompts"]["discover_prompt"].format(n=int(spec.get("n", 15)), media_type=spec.get("media_type", "movie"), criteria=spec.get("criteria", "")), s["gen_model"]); w["calls"] += 1
            existing = {r["title"].casefold() for r in list_riddles()}
            for line in raw.splitlines():
                t = re.sub(r"^\s*[\d\.\)\-\*]+\s*", "", line).strip().strip('"').strip()
                t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
                if not t or len(t) > 90 or t.lower().startswith(("here", "sure", "these")) or t.casefold() in existing: continue
                existing.add(t.casefold())
                rid = time.strftime("r_%Y%m%d_%H%M%S_") + re.sub(r"\W+", "", t.lower())[:24] + f"_{len(w['riddle_ids'])}"
                save_riddle({"id": rid, "title": t, "media_type": spec.get("media_type", "movie"), "status": "draft", "text": "", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "workload": wid, "config": {}, "learner_words": [], "extra_forbidden": [], "forbidden": [], "attempts": [], "log": [], "suitability": None, "validation": None, "recall": None})
                w["riddle_ids"].append(rid)
            w["pending"] = list(w["riddle_ids"])
            wlog(w, f"discovered {len(w['riddle_ids'])} titles"); save_wl(w)
        steps, target = spec.get("steps") or ["assess", "generate", "recall"], int(spec.get("target_tested") or 0)
        while w["pending"]:
            while ctl["pause"] and not ctl["stop"]:
                if w["status"] != "paused": w["status"] = "paused"; wlog(w, "paused"); save_wl(w)
                time.sleep(1)
            if ctl["stop"]: w["status"] = "stopped"; wlog(w, "stopped"); save_wl(w); WCTL.pop(wid, None); return
            if w["status"] != "running": w["status"] = "running"; save_wl(w)
            rid = w["pending"][0]
            r = load_riddle(rid)
            if not r: w["pending"].pop(0); save_wl(w); continue
            interrupted = False
            for s_ in steps:
                if ctl["stop"] or ctl["pause"]: interrupted = True; break
                try:
                    STEPS[s_](r); w["calls"] += (int(cfg_of(r).get("k_attempts", 3)) + 1 if s_ == "generate" else (0 if s_ == "validate" else 1))
                    save_riddle(r)
                    if s_ == "assess" and spec.get("auto_reject", True):
                        su = r.get("suitability") or {}
                        if su.get("verdict") == "poor" or su.get("content_ok") is False:
                            r["status"] = "rejected"; log(r, "auto-rejected"); save_riddle(r); break
                except Exception as e:
                    wlog(w, f"{r['title']}: ERROR {s_}: {str(e)[:150]}"); log(r, f"ERROR {s_}: {e}"); save_riddle(r); break
            if interrupted: save_wl(w); continue
            w["pending"].pop(0)
            wlog(w, f"{r['title']} -> {load_riddle(rid)['status']}")
            save_wl(w)
            if target and wl_counts(w)["tested"] >= target:
                wlog(w, f"target of {target} tested reached"); break
        w["status"] = "done"; wlog(w, "workload complete"); save_wl(w)
    except Exception as e:
        w["status"] = "error"; wlog(w, "FATAL " + str(e)[:200]); save_wl(w)
    WCTL.pop(wid, None)

def register(app, exp_dir):
    global DIR
    from flask import request, jsonify, send_from_directory
    DIR = os.path.join(exp_dir, "_riddles")
    os.makedirs(DIR, exist_ok=True)
    apply_keys()
    for w in list_wls():
        if w["status"] in ("running", "paused"): w["status"] = "interrupted"; wlog(w, "interrupted by restart"); save_wl(w)

    @app.route("/api/workloads", methods=["GET"])
    def wl_list():
        return jsonify([{k: w.get(k) for k in ("id", "name", "status", "calls", "created", "updated")} | {"counts": wl_counts(w), "spec": w["spec"]} for w in list_wls()])

    @app.route("/api/workloads", methods=["POST"])
    def wl_create():
        body = request.get_json(force=True)
        wid = time.strftime("w_%Y%m%d_%H%M%S")
        spec = {"mode": body.get("mode", "discover"), "criteria": body.get("criteria", ""), "media_type": body.get("media_type", "movie"), "n": int(body.get("n", 15)), "steps": [s for s in body.get("steps", ["assess", "generate", "recall"]) if s in STEPS], "auto_reject": bool(body.get("auto_reject", True)), "target_tested": int(body.get("target_tested") or 0)}
        ids = [i for i in body.get("ids", []) if load_riddle(i)] if spec["mode"] == "existing" else []
        w = {"id": wid, "name": body.get("name") or (spec["criteria"][:40] or spec["media_type"] + " batch"), "status": "running", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "spec": spec, "config_snapshot": settings(), "riddle_ids": ids, "pending": list(ids), "calls": 0, "log": []}
        if spec["mode"] == "existing":
            for rid in ids:
                r = load_riddle(rid); r["workload"] = wid; save_riddle(r)
        save_wl(w)
        threading.Thread(target=run_workload, args=(wid,), daemon=True).start()
        return jsonify({"id": wid})

    @app.route("/api/workloads/<wid>", methods=["GET"])
    def wl_get(wid):
        w = load_wl(wid)
        if not w: return jsonify({"error": "not found"}), 404
        riddles_ = []
        for rid in w["riddle_ids"]:
            r = load_riddle(rid)
            if r: riddles_.append({"id": rid, "title": r["title"], "status": r["status"], "guess": (r.get("recall") or {}).get("guess"), "hit": (r.get("recall") or {}).get("hit"), "suit": (r.get("suitability") or {}).get("verdict"), "n_out": len((r.get("validation") or {}).get("outside", [])), "pending": rid in w["pending"]})
        return jsonify({k: w.get(k) for k in ("id", "name", "status", "calls", "spec", "log", "created", "updated")} | {"counts": wl_counts(w), "riddles": riddles_})

    @app.route("/api/workloads/<wid>/ctl", methods=["POST"])
    def wl_ctl(wid):
        w = load_wl(wid)
        if not w: return jsonify({"error": "not found"}), 404
        act = request.get_json(force=True).get("action")
        if act == "pause" and wid in WCTL: WCTL[wid]["pause"] = True
        elif act == "resume":
            if wid in WCTL: WCTL[wid]["pause"] = False
            elif w["status"] in ("interrupted", "paused", "stopped", "error") and w["pending"]:
                w["status"] = "running"; wlog(w, "resumed"); save_wl(w)
                threading.Thread(target=run_workload, args=(wid,), daemon=True).start()
        elif act == "stop":
            if wid in WCTL: WCTL[wid]["stop"] = True
            else: w["status"] = "stopped"; save_wl(w)
        elif act == "delete":
            if wid in WCTL: WCTL[wid]["stop"] = True
            if os.path.exists(wpath(wid)): os.remove(wpath(wid))
            return jsonify({"deleted": wid})
        return jsonify({"status": load_wl(wid)["status"] if load_wl(wid) else "deleted"})

    @app.route("/api/riddle_keys", methods=["GET"])
    def get_keys(): return jsonify(key_status())

    @app.route("/api/provider_test/<prov>", methods=["POST"])
    def provider_test(prov):
        if prov not in pipeline.PROVIDERS: return jsonify({"error": "unknown provider"}), 404
        p = pipeline.PROVIDERS[prov]
        model = (request.get_json(force=True) or {}).get("model", "").strip()
        if not model:
            try:
                import requests as rq
                resp = rq.get(p["base_url"] + "/models", headers={"Authorization": "Bearer " + os.environ.get(p["key_env"], "")}, timeout=20).json()
                ids = [m.get("id") or m.get("name", "") for m in (resp.get("data") or resp.get("models") or [])]
                ids = [i.split("/")[-1] if prov == "gemini" else i for i in ids if i]
                pref = [i for h in ("flash", "instant", "haiku", "small", "turbo", r"(^|[^a-z])mini", "8b") for i in ids if re.search(h, i.lower())]
                model = (pref or ids)[0]
            except Exception as e:
                return jsonify({"ok": False, "error": f"could not list models: {str(e)[:200]}"})
            if prov != "openrouter": model = f"{prov}::{model}"
        t0 = time.time()
        try:
            out = pipeline.openrouter_raw(os.environ.get("OPENROUTER_API_KEY", ""), 0.0, max_retries=0, max_tokens=8)("Reply with the single word: ok", model)
            return jsonify({"ok": True, "seconds": round(time.time() - t0, 2), "reply": out[:40], "model": model})
        except Exception as e:
            return jsonify({"ok": False, "seconds": round(time.time() - t0, 2), "error": str(e)[:300], "model": model})

    @app.route("/api/lex/sources", methods=["GET"])
    def lex_sources():
        out = []
        seen = set()
        for path in sorted(glob.glob(os.path.join(list_dir(), "*.json"))):
            n = os.path.basename(path).rsplit(".", 1)[0]; seen.add(n)
            out.append({"name": n, "origin": "uploaded", "words": len(canonical_words(n)), "lessons": len(wordlist_lessons(n)), "annotated": len(load_annot(n)), "busy": ANNOT_JOBS.get(n)})
        for path in sorted(glob.glob("wordlists/*.json")):
            n = os.path.basename(path).rsplit(".", 1)[0]
            if n in seen: continue
            out.append({"name": n, "origin": "built-in", "words": len(canonical_words(n)), "lessons": len(wordlist_lessons(n)), "annotated": len(load_annot(n)), "busy": ANNOT_JOBS.get(n)})
        return jsonify(out)

    @app.route("/api/lex/sources", methods=["POST"])
    def lex_upload():
        body = request.get_json(force=True)
        name = re.sub(r"\W+", "_", body.get("name", "")).strip("_")
        if not name: return jsonify({"error": "no name"}), 400
        try: data = parse_source(body.get("content", ""), body.get("loader", "auto"), body.get("column"))
        except Exception as e: return jsonify({"error": f"loader failed: {e}"}), 400
        n = len(data.get("lessons", {})) if isinstance(data, dict) else len(data)
        if not n: return jsonify({"error": "loader produced nothing"}), 400
        if body.get("preview"): return jsonify({"name": name, "preview": (data if isinstance(data, list) else sorted(data["lessons"].keys(), key=lesson_sort_key))[:60], "count": n})
        os.makedirs(list_dir(), exist_ok=True)
        _save(os.path.join(list_dir(), name + ".json"), data)
        return jsonify({"name": name, "count": n})

    @app.route("/api/lex/sources/<name>/delete", methods=["POST"])
    def lex_delete(name):
        p = os.path.join(list_dir(), name + ".json")
        if os.path.exists(p): os.remove(p)
        if os.path.exists(annot_path(name)): os.remove(annot_path(name))
        return jsonify({"deleted": name})

    @app.route("/api/lex/table/<name>")
    def lex_table(name):
        ann = load_annot(name)
        words = canonical_words(name)
        cols = sorted({c for a in ann.values() for c in a})
        rows = [{"word": w, "annot": ann.get(w, {})} for w in words]
        return jsonify({"cols": cols, "rows": rows, "busy": ANNOT_JOBS.get(name)})

    @app.route("/api/lex/annotate", methods=["POST"])
    def lex_annotate():
        body = request.get_json(force=True)
        name, annotators = body.get("list"), [a for a in body.get("annotators", []) if a in ("topic", "level", "concreteness", "pos")]
        if not name or not annotators: return jsonify({"error": "need list and annotators"}), 400
        if ANNOT_JOBS.get(name): return jsonify({"error": "busy"}), 409
        s = settings()
        model = body.get("model") or s["gen_model"]
        ANNOT_JOBS[name] = {"done": 0, "total": 1, "error": ""}
        threading.Thread(target=run_annotators, args=(name, annotators, model, s), daemon=True).start()
        return jsonify({"started": annotators})

    @app.route("/api/lex/annot/<name>", methods=["POST"])
    def lex_annot_edit(name):
        body = request.get_json(force=True)
        ann = load_annot(name)
        for w, kv in body.get("set", {}).items():
            for c, v in kv.items():
                if v in ("", None): ann.get(w, {}).pop(c, None)
                else: ann.setdefault(w, {})[c] = v
        _save(annot_path(name), ann)
        return jsonify({"ok": True})

    @app.route("/api/lex/export", methods=["POST"])
    def lex_export():
        body = request.get_json(force=True)
        src, name = body.get("list"), re.sub(r"\W+", "_", body.get("name", "")).strip("_")
        if not src or not name: return jsonify({"error": "need list and name"}), 400
        ann = load_annot(src)
        rows = apply_filters([{"word": w, "annot": ann.get(w, {})} for w in canonical_words(src)], body.get("filters"))
        words = [r["word"] for r in rows]
        if not words: return jsonify({"error": "empty result"}), 400
        os.makedirs(list_dir(), exist_ok=True)
        _save(os.path.join(list_dir(), name + ".json"), words)
        for w in words:
            if ann.get(w): pass
        _save(annot_path(name), {w: ann[w] for w in words if w in ann})
        return jsonify({"name": name, "count": len(words)})

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
        for path in glob.glob("wordlists/*.json") + glob.glob(os.path.join(list_dir(), "*.json")):
            name = os.path.basename(path).rsplit(".", 1)[0]
            try: out[name] = wordlist_lessons(name)
            except Exception: out[name] = []
        return jsonify(out)

    @app.route("/api/riddle_wordlists", methods=["POST"])
    def upload_wordlist():
        body = request.get_json(force=True)
        name = re.sub(r"\W+", "_", body.get("name", "")).strip("_")
        if not name: return jsonify({"error": "no name"}), 400
        try: data = parse_source(body.get("content", ""), "auto")
        except Exception as e: return jsonify({"error": str(e)}), 400
        os.makedirs(list_dir(), exist_ok=True)
        _save(os.path.join(list_dir(), name + ".json"), data)
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
