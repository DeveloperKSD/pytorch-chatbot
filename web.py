
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bot.factory import build_dialog, model_exists, train_and_save
from bot.memory import Session

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sam</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--text:#1c1c1c;--mute:#777;--me:#2f6fed;--bot:#ececea}
@media(prefers-color-scheme:dark){:root{--bg:#161618;--card:#202023;--text:#eee;--mute:#999;--me:#3b78f0;--bot:#2c2c30}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif;
display:flex;justify-content:center}main{width:100%;max-width:640px;height:100vh;display:flex;flex-direction:column;padding:12px}
h1{font-size:18px;margin:4px 0 8px}#log{flex:1;overflow:auto;background:var(--card);border-radius:12px;padding:12px}
.m{max-width:85%;margin:6px 0;padding:8px 12px;border-radius:14px;white-space:pre-wrap;line-height:1.35}
.me{background:var(--me);color:#fff;margin-left:auto}.bot{background:var(--bot)}
.meta{font-size:11px;color:var(--mute);margin:-2px 0 4px 6px}
form{display:flex;gap:8px;margin-top:10px}input{flex:1;padding:12px;border-radius:10px;border:1px solid #8884;
background:var(--card);color:var(--text);font:inherit}button{padding:0 18px;border:0;border-radius:10px;background:var(--me);color:#fff;font:inherit}
</style></head><body><main><h1>Sam - learning chatbot</h1><div id="log"></div>
<form id="f"><input id="t" autocomplete="off" placeholder="Say something..." autofocus><button>Send</button></form></main>
<script>
const sid=Math.random().toString(36).slice(2),log=document.getElementById('log');
function add(text,cls,meta){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;log.appendChild(d);
if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;log.appendChild(m)}log.scrollTop=log.scrollHeight}
add("Hi! I'm Sam. Try: weather in Mumbai, what is 9+10, tell me a joke, how do you work","bot");
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const i=document.getElementById('t'),v=i.value.trim();if(!v)return;
i.value='';add(v,'me');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid,message:v})});
const j=await r.json();add(j.reply,'bot',j.route+(j.intent?' / '+j.intent:'')+' / '+j.confidence.toFixed(2))}catch(_){add('Connection problem.','bot')}};
</script></body></html>"""


def make_handler(dm):
    sessions = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/chat":
                return self._send(404, "not found", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 10_000)
                payload = json.loads(self.rfile.read(length) or b"{}")
                sid = str(payload.get("sid", ""))[:40] or "anon"
                message = str(payload.get("message", ""))[:1000]
            except (ValueError, TypeError):
                return self._send(400, json.dumps({"error": "bad request"}), "application/json")
            if len(sessions) > 1000:      # crude memory guard
                sessions.clear()
            session = sessions.setdefault(sid, Session())
            r = dm.reply(message, session)
            body = json.dumps({"reply": r.text, "route": r.route, "intent": r.intent,
                               "confidence": float(r.confidence)})
            self._send(200, body, "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--intents", default="intents.json")
    a = ap.parse_args()
    if not model_exists(a.model_dir):
        print("No model found - training one first...")
        train_and_save(a.intents, a.model_dir)
    dm = build_dialog(a.model_dir, a.intents)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(dm))
    print(f"Open http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
