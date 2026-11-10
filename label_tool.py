"""Local browser tool for the two manual sheets. No API calls.

    python label_tool.py        then open http://127.0.0.1:8765

Part 1 fills human_label in manual_labels.csv (100 rows).
Part 2 fills error_type in error_samples.csv (30 rows). It stays locked until part 1
is finished, because that sheet shows the judge's labels and reasons.

Every answer is saved to the CSV immediately, so you can stop and resume at any time.
This tool never reads manual_labels_key.csv.
"""
import argparse
import csv
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config as C

SHEETS = {
    "labels": {"file": "manual_labels.csv", "column": "human_label",
               "values": ["supported", "contradicted", "unsupported"]},
    "errors": {"file": "error_samples.csv", "column": "error_type",
               "values": ["wrong passage retrieved", "one hop missing", "passage misread",
                          "answer from memory", "judge error"]},
}
LOCK = threading.Lock()
csv.field_size_limit(10_000_000)


def read_sheet(folder, name):
    path = Path(folder) / SHEETS[name]["file"]
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames, rows


def write_sheet(folder, name, fields, rows):
    """Write to a temp file first, so a crash can't leave a half-written sheet."""
    path = Path(folder) / SHEETS[name]["file"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def labels_done(folder):
    _, rows = read_sheet(folder, "labels")
    column = SHEETS["labels"]["column"]
    return all(r[column].strip() for r in rows)


def load_state(folder):
    state = {"errors_unlocked": labels_done(folder)}
    for name, sheet in SHEETS.items():
        _, rows = read_sheet(folder, name)
        if name == "errors" and not state["errors_unlocked"]:
            rows = []  # don't send judge labels and reasons to the page yet
        state[name] = {"column": sheet["column"], "values": sheet["values"], "rows": rows}
    return state


def save_answer(folder, name, row_id, value):
    """Returns an error message, or None if saved."""
    sheet = SHEETS.get(name)
    if sheet is None or value not in sheet["values"]:
        return "Invalid sheet or value."
    if name == "errors" and not labels_done(folder):
        return "Finish part 1 first."
    fields, rows = read_sheet(folder, name)
    for row in rows:
        if row["id"] == row_id:
            row[sheet["column"]] = value
            write_sheet(folder, name, fields, rows)
            return None
    return f"Unknown id: {row_id}"


class Handler(BaseHTTPRequestHandler):
    folder = None

    def log_message(self, *args):
        pass

    def reply(self, code, body, content_type="application/json"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/":
                self.reply(200, PAGE, "text/html")
            elif self.path == "/api/state":
                with LOCK:
                    self.reply(200, json.dumps(load_state(self.folder)))
            else:
                self.reply(404, "{}")
        except OSError as e:
            self.reply(500, json.dumps({"error": f"Could not read the sheet: {e}"}))

    def do_POST(self):
        if self.path != "/api/save":
            return self.reply(404, "{}")
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                error = save_answer(self.folder, body["sheet"], body["id"], body["value"])
        except OSError:
            error = "Could not write the CSV. Is it open in Excel? Close it and try again."
        except (KeyError, TypeError, ValueError):
            error = "Bad request."
        if error:
            return self.reply(400, json.dumps({"error": error}))
        self.reply(200, '{"ok": true}')


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manual labelling</title>
<style>
:root { color-scheme: light dark; --bg:#f7f7f5; --card:#fff; --text:#1f1f1f; --muted:#666;
        --line:#ddd; --accent:#1f5fbf; --on:#fff; --bad:#b3261e; --mark:#fff6d6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#17181a; --card:#212226; --text:#e8e8e6; --muted:#a0a0a0;
          --line:#3a3b40; --accent:#6ea2f2; --on:#10131a; --bad:#f2968f; --mark:#3a3420; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:16px/1.55 system-ui, "Segoe UI", sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px 16px 48px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:baseline; }
h1 { font-size:18px; margin:0; }
#progress { color:var(--muted); font-variant-numeric:tabular-nums; }
#bar { height:6px; background:var(--line); border-radius:3px; margin:10px 0 16px; overflow:hidden; }
#fill { height:100%; background:var(--accent); width:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:14px 16px; margin-bottom:12px; }
.tag { font-size:12px; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
       color:var(--muted); margin-bottom:4px; }
#answer { background:var(--mark); }
#answerText { font-size:18px; font-weight:600; white-space:pre-wrap; overflow-wrap:anywhere; }
#passages p { margin:0 0 12px; overflow-wrap:anywhere; }
#passages p:last-child { margin:0; }
#meta dl { display:grid; grid-template-columns:max-content 1fr; gap:2px 12px; margin:0; }
#meta dt { color:var(--muted); }
#meta dd { margin:0; overflow-wrap:anywhere; }
#choices { display:grid; gap:8px; margin-bottom:12px; }
button { font:inherit; color:inherit; background:var(--card); border:1px solid var(--line);
         border-radius:6px; padding:8px 12px; cursor:pointer; text-align:left; }
button:hover:not(:disabled) { border-color:var(--accent); }
button:disabled { opacity:.45; cursor:default; }
button.on { background:var(--accent); color:var(--on); border-color:var(--accent); }
button.on small { color:inherit; }
button small { color:var(--muted); display:block; font-size:13px; }
kbd { font:600 12px ui-monospace, Consolas, monospace; border:1px solid currentColor;
      border-radius:3px; padding:0 4px; margin-right:6px; }
nav { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
#status { min-height:24px; color:var(--bad); font-weight:600; margin-top:8px; }
#done { display:none; }
</style></head><body><main>
<header><h1 id="title"></h1><span id="progress"></span></header>
<div id="bar"><div id="fill"></div></div>
<div id="work">
  <div class="card"><div class="tag">Question</div><div id="question"></div></div>
  <div class="card" id="answer"><div class="tag">Model's answer</div><div id="answerText"></div></div>
  <div id="choices"></div>
  <nav>
    <button id="prev"><kbd>&larr;</kbd>Previous</button>
    <button id="next"><kbd>&rarr;</kbd>Next</button>
    <button id="skip"><kbd>N</kbd>Next unanswered</button>
  </nav>
  <div id="status" role="alert"></div>
  <div class="card" id="meta" hidden><div class="tag">Details</div><dl id="metaList"></dl></div>
  <div class="card"><div class="tag">Passages shown to the model</div><div id="passages"></div></div>
</div>
<div class="card" id="done"></div>
</main><script>
const HELP = {
  "supported": "The passages state or clearly imply the answer.",
  "contradicted": "The passages say something that conflicts with the answer.",
  "unsupported": "The passages don't contain enough information to back the answer.",
  "wrong passage retrieved": "The passages don't contain the facts needed; retrieval went wrong.",
  "one hop missing": "One needed fact is in the passages, but the other hop is missing.",
  "passage misread": "The needed facts are there, but the model drew the wrong conclusion.",
  "answer from memory": "The answer doesn't come from the passages at all.",
  "judge error": "The answer is actually fine; the judge got it wrong."
};
const KEYS = { labels: ["s", "c", "u"], errors: ["1", "2", "3", "4", "5"] };
const $ = id => document.getElementById(id);
let state, mode, i = 0, saving = false;

const sheet = () => state[mode];
const filled = r => r[sheet().column].trim() !== "";
const firstOpen = () => sheet().rows.findIndex(r => !filled(r));

async function load() {
  const res = await fetch("/api/state");
  state = await res.json();
  if (state.error) { $("status").textContent = state.error; return; }
  mode = state.errors_unlocked ? "errors" : "labels";
  i = Math.max(firstOpen(), 0);
  render();
}

function render() {
  const s = sheet(), rows = s.rows, count = rows.filter(filled).length, r = rows[i];
  $("title").textContent = mode === "labels"
    ? "Part 1 of 2: judge from the passages only, not your own knowledge"
    : "Part 2 of 2: why did this answer go wrong?";
  $("progress").textContent = `Row ${i + 1} of ${rows.length} (${r.id}) · ${count} answered`;
  $("fill").style.width = (100 * count / rows.length) + "%";
  $("question").textContent = r.question;
  $("answerText").textContent = r.answer;
  $("passages").replaceChildren(...r.passages.split(/\n\s*\n/).map(t => {
    const p = document.createElement("p"); p.textContent = t; return p;
  }));
  $("choices").replaceChildren(...s.values.map((v, n) => {
    const b = document.createElement("button");
    const k = document.createElement("kbd"); k.textContent = KEYS[mode][n].toUpperCase();
    const h = document.createElement("small"); h.textContent = HELP[v];
    b.append(k, v, h);
    b.className = r[s.column].trim().toLowerCase() === v ? "on" : "";
    b.onclick = () => choose(v);
    return b;
  }));
  $("meta").hidden = mode !== "errors";
  if (mode === "errors") {
    const items = [["Model", r.model], ["Retriever", r.retriever], ["Gold answer", r.gold],
      ["Judge label", r.judge_label], ["Judge reason", r.judge_reason],
      ["Gold passages in top 5", r.gold_passages_in_top5],
      ["Answer words in context", r.answer_in_context]];
    $("metaList").replaceChildren(...items.flatMap(([a, b]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = a; dd.textContent = b; return [dt, dd];
    }));
  }
  $("prev").disabled = i === 0;
  $("next").disabled = i === rows.length - 1;
  $("skip").disabled = count === rows.length;
  $("work").style.display = "block";
  $("done").style.display = "none";
}

async function choose(value) {
  if (saving) return;
  saving = true;
  const r = sheet().rows[i];
  $("status").textContent = "";
  try {
    const res = await fetch("/api/save", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sheet: mode, id: r.id, value }) });
    const out = await res.json();
    if (out.error) { $("status").textContent = "NOT saved. " + out.error; return; }
    r[sheet().column] = value;
    advance();
  } catch (e) {
    $("status").textContent = "NOT saved. Is label_tool.py still running?";
  } finally { saving = false; }
}

function advance() {
  const rows = sheet().rows;
  if (i < rows.length - 1) { i++; go(); return; }
  const open = firstOpen();
  if (open >= 0) { i = open; go(); return; }
  finished();
}

function go() { render(); window.scrollTo(0, 0); }

async function finished() {
  $("work").style.display = "none";
  const d = $("done"); d.style.display = "block";
  if (mode === "labels") {
    d.textContent = "Part 1 is finished and saved. ";
    const b = document.createElement("button");
    b.textContent = "Start part 2";
    b.onclick = load;
    d.append(b);
  } else {
    d.textContent = "Both sheets are finished and saved. Close this tab, stop label_tool.py, "
      + "then run: python run_analysis.py";
  }
}

document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.altKey || e.metaKey || !state || $("work").style.display === "none") return;
  const k = e.key.toLowerCase(), n = KEYS[mode].indexOf(k);
  if (n >= 0) { e.preventDefault(); choose(sheet().values[n]); }
  else if (k === "arrowleft" && i > 0) { i--; go(); }
  else if (k === "arrowright" && i < sheet().rows.length - 1) { i++; go(); }
  else if (k === "n" && firstOpen() >= 0) { i = firstOpen(); go(); }
});
$("prev").onclick = () => { i--; go(); };
$("next").onclick = () => { i++; go(); };
$("skip").onclick = () => { i = firstOpen(); go(); };
load();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Fill the manual sheets in a browser.")
    ap.add_argument("--dir", default=C.RESULTS_DIR / "analysis", help="folder holding the sheets")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    Handler.folder = args.dir
    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open {url}\nEverything saves as you go. To stop: close this terminal.")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
