import os, json, threading, traceback, time, glob
from flask import Flask, request, jsonify, send_from_directory
import pipeline

EXP_DIR = os.environ.get("EXP_DIR", "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

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
                self.log(f"round {rnd}/{rounds} start: {len(targets)} targets, model {p.model}")
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
