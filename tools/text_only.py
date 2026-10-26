import json, sys, urllib.request
URL = "http://127.0.0.1:8002/completion"
det = {"prompt": "Translate to French: The cat sleeps. Le chat dort.\nTranslate to French:",
       "max_tokens": 200, "temperature": 0.0, "stream": False, "ignore_eos": False,
       "seed": 42, "n_predict": 200}
data = json.dumps(det).encode()
r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=300).read())
tag = sys.argv[1] if len(sys.argv) > 1 else "x"
open("/tmp/ab-text-%s.txt" % tag, "w").write(d.get("content", "") + "\n---\n" + d.get("reasoning_content", "") + "\n")
print("determinism text -> /tmp/ab-text-%s.txt" % tag)
