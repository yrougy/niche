import json, time, urllib.request

URL = "http://127.0.0.1:8002/completion"

def req(prompt, n, temp=0.0, eos=True):
    body = {"prompt": prompt, "max_tokens": n, "temperature": temp, "stream": False,
            "ignore_eos": eos}
    data = json.dumps(body).encode()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    t = d.get("timings", {})
    return t

LINES = [
    "Translate to French: The dog runs fast. Le chien court vite.",
    "Translate to French: The cat sleeps. Le chat dort.",
    "Translate to French: The horse is big. Le cheval est grand.",
    "Translate to French: The bird sings. L'oiseau chante.",
    "Translate to French: The tree grows. L'arbre pousse.",
]
REP = "\n".join(LINES * 16) + "\n"

print("warmup", req("Hello.", 16))

for i in range(3):
    t = req("Q: What is the capital of France? A:", 700)
    print("tg bench  r%d: tg=%.2f t/s  n=%s  (%.1fs)  acc=%s len=%s" % (
        i+1, t.get("predicted_per_second", 0), t.get("predicted_n"),
        t.get("predicted_ms", 0)/1000, t.get("draft", {}).get("accept_pct", "?"),
        t.get("draft", {}).get("mean", "?")))

for i in range(2):
    t = req(REP, 500)
    print("rep 500   r%d: tg=%.2f t/s  n=%s  acc=%s len=%s" % (
        i+1, t.get("predicted_per_second", 0), t.get("predicted_n"),
        t.get("draft", {}).get("accept_pct", "?"), t.get("draft", {}).get("mean", "?")))

t = req("Translate to French: The cat sleeps. Le chat dort.\nTranslate to French:", 300)
print("short     r1: tg=%.2f t/s  n=%s" % (t.get("predicted_per_second", 0), t.get("predicted_n")))

# fixed determinism probe: same text OFF vs ON
import sys
det = {"prompt": "Translate to French: The cat sleeps. Le chat dort.\nTranslate to French:",
       "max_tokens": 200, "temperature": 0.0, "stream": False, "ignore_eos": False,
       "seed": 42, "n_predict": 200}
data = json.dumps(det).encode()
r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read())
tag = sys.argv[1] if len(sys.argv) > 1 else "x"
open("/tmp/ab-text-%s.txt" % tag, "w").write(d.get("content", "") + "\n---\n" + d.get("reasoning_content", "") + "\n")
print("determinism text -> /tmp/ab-text-%s.txt" % tag)
