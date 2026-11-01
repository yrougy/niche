import json, os, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "8003"))
URL = "http://127.0.0.1:%d/completion" % PORT

def req(prompt, n, temp=0.0, eos=True):
    body = {"prompt": prompt, "max_tokens": n, "temperature": temp, "stream": False,
            "ignore_eos": eos}
    data = json.dumps(body).encode()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return d.get("timings", {})

LINES = [
    "Translate to French: The dog runs fast. Le chien court vite.",
    "Translate to French: The cat sleeps. Le chat dort.",
    "Translate to French: The horse is big. Le cheval est grand.",
    "Translate to French: The bird sings. L'oiseau chante.",
    "Translate to French: The tree grows. L'arbre pousse.",
]
REP = "\n".join(LINES * 16) + "\n"

tag = sys.argv[1] if len(sys.argv) > 1 else "x"
print("[%s] warmup %s" % (tag, req("Hello.", 16).get("predicted_per_second", 0)))
for i in range(3):
    t = req("Q: What is the capital of France? A:", 700)
    print("[%s] tg bench r%d: tg=%.2f t/s n=%s (%.1fs)" % (
        tag, i+1, t.get("predicted_per_second", 0), t.get("predicted_n"),
        t.get("predicted_ms", 0)/1000))
for i in range(2):
    t = req(REP, 500)
    print("[%s] rep 500  r%d: tg=%.2f t/s n=%s" % (
        tag, i+1, t.get("predicted_per_second", 0), t.get("predicted_n")))
t = req("Translate to French: The cat sleeps. Le chat dort.\nTranslate to French:", 300)
print("[%s] short    r1: tg=%.2f t/s n=%s" % (tag, t.get("predicted_per_second", 0), t.get("predicted_n")))
