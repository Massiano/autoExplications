import os, json, threading, traceback, time, glob, shutil, sys, platform
from flask import Flask, request, jsonify, send_from_directory
import pipeline

EXP_DIR = os.environ.get("EXP_DIR", "experiments")
os.makedirs(EXP_DIR, exist_ok=True)
print(f"[boot] EXP_DIR={os.path.abspath(EXP_DIR)} (env {'set' if 'EXP_DIR' in os.environ else 'NOT SET - ephemeral default'})", flush=True)

app = Flask(__name__)
runners = {}
lock = threading.Lock()

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/api/<path:_>", methods=["OPTIONS"])
def options(_): return ""

def exp_path(eid, name): return os.path.join(EXP_DIR, eid, name)

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return default

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)

class Runner:
    def __init__(self, eid):
        self.eid = eid
        self.stop_flag = False
        self.status = "idle"
        self.error = ""
        self.thread = None

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(exp_path(self.eid, "log.txt"), "a") as f: f.write(line + "\n")

    def start(self):
        if self.thread and self.thread.is_alive(): return False
        self.stop_flag = False; self.error = ""; self.status = "running"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        t0 = time.monotonic()
        try:
            cfg = load_json(exp_path(self.eid, "config.json"), {})
            state = load_json(exp_path(self.eid, "state.json"), {})
            targets = cfg["target_words"]
            rounds = int(cfg.get("rounds", 1))
            for rnd in range(1, rounds + 1):
                state = load_json(exp_path(self.eid, "state.json"), {})
                p = pipeline.Pipeline(cfg, state, lambda s: save_json(exp_path(self.eid, "state.json"), s), self.log, lambda: self.stop_flag)
                before_v = len(state.get("verified_explications", {}))
                before_a = sum(len(b.get("words", [])) for b in state.get("shell", {}).get("admitted", []))
                self.log(f"round {rnd}/{rounds} start: {len(targets)} targets, models {p.role_models}")
                p.run(targets)
                after = load_json(exp_path(self.eid, "state.json"), {})
                after_v = len(after.get("verified_explications", {}))
                after_a = sum(len(b.get("words", [])) for b in after.get("shell", {}).get("admitted", []))
                self.log(f"round {rnd}: +{after_v - before_v} verified, +{after_a - before_a} admitted")
                if after_v == before_v and after_a == before_a:
                    self.log("no progress this round, stopping early"); break
                if after_v >= len(targets):
                    self.log("all targets verified"); break
            self.status = "done"
            self.log(f"run complete in {time.monotonic() - t0:.0f}s")
        except pipeline.StopRequested:
            self.status = "stopped"; self.log(f"stopped by request after {time.monotonic() - t0:.0f}s")
        except Exception as e:
            self.status = "error"; self.error = str(e)
            self.log("ERROR: " + str(e))
            self.log(traceback.format_exc())

def get_runner(eid):
    with lock:
        if eid not in runners: runners[eid] = Runner(eid)
        return runners[eid]

BOOT_TIME = time.time()

@app.route("/api/diag")
def diag():
    d = {}
    d["exp_dir"] = {"configured": os.environ.get("EXP_DIR", "(not set, default 'experiments')"), "resolved": os.path.abspath(EXP_DIR), "exists": os.path.isdir(EXP_DIR)}
    try:
        du = shutil.disk_usage(EXP_DIR)
        d["exp_dir"]["disk"] = {"total_mb": du.total // 2**20, "used_mb": du.used // 2**20, "free_mb": du.free // 2**20}
    except Exception as e: d["exp_dir"]["disk"] = str(e)
    try:
        root = shutil.disk_usage("/")
        d["root_disk"] = {"total_mb": root.total // 2**20, "free_mb": root.free // 2**20}
        d["exp_dir"]["on_separate_device"] = d["exp_dir"].get("disk", {}).get("total_mb") != d["root_disk"]["total_mb"] if isinstance(d["exp_dir"].get("disk"), dict) else None
    except Exception as e: d["root_disk"] = str(e)
    try:
        st_exp = os.stat(os.path.abspath(EXP_DIR)); st_root = os.stat("/")
        d["exp_dir"]["device_id"] = st_exp.st_dev; d["root_device_id"] = st_root.st_dev
        d["exp_dir"]["is_own_mount"] = st_exp.st_dev != st_root.st_dev
    except Exception as e: d["mount_check"] = str(e)
    try:
        probe = os.path.join(EXP_DIR, ".diag_probe")
        with open(probe, "w") as f: f.write(str(time.time()))
        with open(probe) as f: f.read()
        os.remove(probe)
        d["exp_dir"]["write_test"] = "ok"
    except Exception as e: d["exp_dir"]["write_test"] = f"FAILED: {e}"
    try:
        mounts = [l for l in open("/proc/mounts") if "/data" in l or "overlay" in l.split()[2:3]]
        d["mounts"] = [l.strip() for l in mounts][:10]
    except Exception as e: d["mounts"] = str(e)
    exps = []
    try:
        for eid in sorted(os.listdir(EXP_DIR)):
            p = os.path.join(EXP_DIR, eid)
            if not os.path.isdir(p): continue
            files = {}
            for fn in os.listdir(p):
                fp = os.path.join(p, fn)
                files[fn] = {"bytes": os.path.getsize(fp), "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(fp)))}
            exps.append({"id": eid, "files": files})
    except Exception as e: exps = str(e)
    d["experiments_on_disk"] = exps
    d["runners"] = {eid: {"status": r.status, "error": r.error, "thread_alive": bool(r.thread and r.thread.is_alive())} for eid, r in runners.items()}
    d["provider_keys"] = {name: cfg["key_env"] + (" set" if cfg["key_env"] in os.environ else " NOT set") for name, cfg in pipeline.PROVIDERS.items()}
    d["env"] = {"EXP_DIR_set": "EXP_DIR" in os.environ, "OPENROUTER_API_KEY_set": "OPENROUTER_API_KEY" in os.environ, "PORT": os.environ.get("PORT", "(default)"), "railway_vars": {k: v for k, v in os.environ.items() if k.startswith("RAILWAY_") and "TOKEN" not in k and "SECRET" not in k}}
    d["process"] = {"python": sys.version.split()[0], "platform": platform.platform(), "pid": os.getpid(), "cwd": os.getcwd(), "uptime_s": int(time.time() - BOOT_TIME), "threads": threading.active_count()}
    try:
        import resource
        d["process"]["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception: pass
    d["wordlists"] = sorted(os.path.basename(p) for p in glob.glob("wordlists/*.json"))
    try:
        import requests as _rq
        t = time.monotonic()
        r = _rq.get("https://openrouter.ai/api/v1/models", timeout=10)
        d["openrouter_reachable"] = {"status": r.status_code, "ms": int((time.monotonic() - t) * 1000)}
    except Exception as e: d["openrouter_reachable"] = str(e)
    return jsonify(d)

@app.route("/api/provider_models/<prov>")
def provider_models(prov):
    import pipeline as pl, requests as _rq
    if prov not in pl.PROVIDERS: return jsonify({"error": f"unknown provider (known: {list(pl.PROVIDERS)})"}), 400
    p = pl.PROVIDERS[prov]
    key = os.environ.get(p["key_env"], "")
    if not key: return jsonify({"error": f"{p['key_env']} not set"}), 400
    try:
        r = _rq.get(p["base_url"] + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
        data = r.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    models = data.get("data", data.get("models", []))
    ids = sorted(m.get("id", m.get("name", "?")) for m in models if isinstance(m, dict))
    return jsonify({"provider": prov, "count": len(ids), "ids": ids})

@app.route("/api/openrouter_models")
def openrouter_models():
    import requests as _rq
    try:
        r = _rq.get("https://openrouter.ai/api/v1/models", timeout=15)
        models = r.json().get("data", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    out = []
    for m in models:
        p = m.get("pricing", {})
        try: free = float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0
        except Exception: free = False
        out.append({"id": m.get("id"), "free": free, "context": m.get("context_length"),
                    "prompt_per_m": round(float(p.get("prompt", 0)) * 1e6, 3) if p.get("prompt") else 0,
                    "completion_per_m": round(float(p.get("completion", 0)) * 1e6, 3) if p.get("completion") else 0})
    out.sort(key=lambda x: (not x["free"], x["id"]))
    return jsonify({"count": len(out), "free_count": sum(1 for x in out if x["free"]), "models": out})

test_jobs = {}

@app.route("/api/test_models", methods=["POST"])
def test_models():
    body = request.get_json(force=True)
    models = [m.strip() for m in body.get("models", []) if m.strip()][:8]
    words = ([w.strip() for w in body.get("words", []) if w.strip()] or ["fire"])[:5]
    prompt_tpl = body.get("prompt") or None
    guess_model = body.get("guess_model", "").strip() or None
    if not models: return jsonify({"error": "no models"}), 400
    import pipeline as pl
    if prompt_tpl is None: prompt_tpl = pl.DEFAULT_PROMPTS["en"]["bootstrap_prompts"][0]
    jid = f"job_{int(time.time() * 1000)}"
    job = {"done": False, "total": len(models) * len(words), "completed": 0, "log": [], "results": [], "prompt": prompt_tpl}
    test_jobs[jid] = job
    def work():
        svc = pl.openrouter_raw(os.environ["OPENROUTER_API_KEY"], 0.6, max_retries=1)
        for model in models:
            for word in words:
                r = {"model": model, "word": word}
                t = time.monotonic()
                try:
                    text = svc(prompt_tpl.format(word=word), model).strip()
                    r["explication"] = text; r["ms"] = int((time.monotonic() - t) * 1000)
                    gm = guess_model or model
                    guess = svc(pl.DEFAULT_PROMPTS["en"]["guess_prompt"].format(text=text), gm).strip()
                    r["guess"] = guess; r["guess_model"] = gm
                    r["ok"] = word.lower() in guess.lower()
                    job["log"].append(f"{model} / {word}: {'OK' if r['ok'] else 'MISS (' + guess + ')'} {r['ms']}ms")
                except Exception as e:
                    r["error"] = str(e)
                    job["log"].append(f"{model} / {word}: ERROR {e}")
                job["results"].append(r)
                job["completed"] += 1
        job["done"] = True
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job": jid})

@app.route("/api/test_models/<jid>")
def test_models_status(jid):
    job = test_jobs.get(jid)
    if not job: return jsonify({"error": "unknown job"}), 404
    return jsonify(job)

@app.route("/api/wordlists")
def wordlists():
    out = {}
    for p in glob.glob("wordlists/*.json"):
        name = os.path.basename(p).rsplit(".", 1)[0]
        try: out[name] = len(pipeline.extract_ltwf_targets(p))
        except Exception: out[name] = None
    return jsonify(out)

@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    out = []
    for eid in sorted(os.listdir(EXP_DIR)):
        cfg = load_json(exp_path(eid, "config.json"), {})
        state = load_json(exp_path(eid, "state.json"), {})
        r = runners.get(eid)
        out.append({"id": eid, "status": r.status if r else "idle", "error": r.error if r else "",
                    "n_targets": len(cfg.get("target_words", [])), "model": cfg.get("model", ""),
                    "verified": len(state.get("verified_explications", {})),
                    "bootstrapped": len(state.get("bootstrap_texts", {}))})
    return jsonify(out)

@app.route("/api/experiments", methods=["POST"])
def create_experiment():
    body = request.get_json(force=True)
    eid = body.get("id") or time.strftime("exp_%Y%m%d_%H%M%S")
    if os.path.exists(exp_path(eid, "config.json")): return jsonify({"error": "exists"}), 400
    cfg = {k: body[k] for k in body if k != "id"}
    if "wordlist" in cfg and "target_words" not in cfg:
        all_targets = pipeline.extract_ltwf_targets(f"wordlists/{cfg['wordlist']}.json")
        cfg["target_words"] = all_targets[:int(cfg.get("n_targets", 30))]
    os.makedirs(os.path.join(EXP_DIR, eid), exist_ok=True)
    save_json(exp_path(eid, "config.json"), cfg)
    save_json(exp_path(eid, "state.json"), pipeline.empty_state())
    return jsonify({"id": eid})

@app.route("/api/experiments/<eid>/delete", methods=["POST"])
def delete_experiment(eid):
    r = runners.get(eid)
    if r and r.thread and r.thread.is_alive(): return jsonify({"error": "still running, stop it first"}), 400
    p = os.path.join(EXP_DIR, eid)
    if not os.path.isdir(p): return jsonify({"error": "not found"}), 404
    shutil.rmtree(p)
    runners.pop(eid, None)
    return jsonify({"deleted": eid})

@app.route("/api/experiments/<eid>/start", methods=["POST"])
def start(eid):
    ok = get_runner(eid).start()
    return jsonify({"started": ok, "status": get_runner(eid).status})

@app.route("/api/experiments/<eid>/stop", methods=["POST"])
def stop(eid):
    r = get_runner(eid); r.stop_flag = True
    return jsonify({"stopping": True})

@app.route("/api/experiments/<eid>/state")
def state(eid):
    return jsonify(load_json(exp_path(eid, "state.json"), {}))

@app.route("/api/experiments/<eid>/config")
def config(eid):
    return jsonify(load_json(exp_path(eid, "config.json"), {}))

@app.route("/api/experiments/<eid>/log")
def logtail(eid):
    p = exp_path(eid, "log.txt")
    if not os.path.exists(p): return ""
    with open(p) as f: lines = f.readlines()
    return "".join(lines[-int(request.args.get("n", 100)):])

@app.route("/")
def index():
    return send_from_directory(".", "explication_dashboard.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), threaded=True)
