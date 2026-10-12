# PROGRESS — port de l'expert cache Strata dans llama.cpp (qwen35moe)

Branch : `experimental/ornith-expert-cache` dans `/data/o35/llama.cpp` (base `master` @ `23b0202a1`).
**Cinq commits faits** (`5bfb641e4`, `1a1811266`, `de0531d8f`, `1a6b4a0b0`, `47a1dbbfc`). cwd = `/data/o35/o35`.
Dernière mise à jour : 2026-10-12, **fallback CPU par expert porté (exigence 6)** + mesures froid/chaud
+ **deuxième prompt (prompt B) testé : divergence mesurée, chemin CPU prouvé exact sur prompt A**.

## Où j'en suis (une ligne)

**Le bug de texte corrompu du mode statique est RÉSOLU et les deux commits sont faits**
(`5bfb641e4` instrumentation de routage, `1a1811266` seed statique + plages LRU par couche +
metrics). **Le complément RAM (exigences 4 et 5) est porté, buildé et mesuré** :
`populate`/`pin`/`release` dans `llama_mmap`, `mmap_range()` dans `llama_model`,
`cache_complement()` dans le cache MoE, option `--moe-cache-pin-mib`. Run froid (page cache
évicté) : **majflt = 0 sur tout le run**, ~87 MiB seulement dans la fenêtre de génération,
chiffre **insensible** à pin/populate/release/LRU/taille de KV. **Run à page cache chaude
(pin17) : la fenêtre de génération lit 0,00 MiB** → exigence 5 remplie. Output
**bit-identique** (`pin7` == clean128, `pin8` == clean512, `pin5` == clean128).
**Commits faits : `de0531d8f`** (12 fichiers, 300 insertions) et **`1a6b4a0b0`** (3 fichiers,
45 insertions : peuplement du fichier entier). `IMPLEMENTATION_NOTES.md` **écrit**. La question des
lectures froides est **résolue** (queue du fichier + évictions sous pression, pas les plages
d'experts) : **207,52 → 112,69 MiB**. **Le fallback CPU par expert (exigence 6) est PORTÉ, buildé et
mesuré** (`47a1dbbfc`, 11 fichiers, 217 insertions / 57 deletions) : **63 % des calculs d'experts sur le
CPU, `uploaded = 0,00 MiB` pendant le décode, output bit-identique, les 7 diffs de `verify.sh` = 0**
(`cpufb` == `clean128`). Voir la section « Fallback CPU porté ». **Deuxième prompt testé (prompt B)** :
la sortie **n'est pas bit-identique** (`cleanB` vs `cpufbB` = **121 lignes de diff**, préfixe commun
**1514 caractères sur 9027** = **16,8 %** de la génération identique) ; **la cause est la différence de
kernel entre le calcul CPU (`MMQ`) et le calcul CUDA (`MMVQ`) sur les mêmes octets quantifiés** — la sonde
a mesuré un écart relatif de **0,3 %**, ce n'est **pas un bug** — et **`baseB` n'est pas la baseline
numérique** (ses couches denses tournent sur le CPU) : la preuve que le calcul CPU est exact est
**`allcpuA` == `base128` (diff de contenu = 0)** avec **tous** les experts sur le CPU.
Voir la section « Deuxième prompt (prompt B) ». **Premier essai sans MTP sur un run long** (prompt B,
`-n 1024`, froid, `pin 1990`) : **16,26 t/s** avec le fallback CPU et **0,05 MiB lu sur le disque pendant
tout le décode** (chemin tout-GPU : 20,13 t/s mais **115 MiB** lues) — **exigence 5 vérifiée sur un run
long**. Voir la section « Premier essai sans MTP ».

## Diagnostic du bug (résolu cette session)

### Cause racine : éviction cross-couche dans un même split

`ggml_backend.cpp:1835-1838` : le callback de copie n'est appelé **que** pour les poids host, et
s'il renvoie true la copie standard est sautée. `ggml_backend_sched_compute_splits` **copie toutes
les entrées d'un split avant de calculer ce split**.

Conséquence : deux couches du même groupe dans le même split partagent le bank. L'upload de la
couche 38 écrasait les bytes que la couche 34 allait lire → texte corrompu
(`Here's a33333` / `Here's333333`).

Le même mécanisme explique la **CUDA illegal memory access** : l'éviction remet
`slot_map[il_prev][e] = -1`, `get_rows` renvoie `-1`, `mul_mat_id` lit la ligne `-1` du bank.

### Correctif : plages de slots disjointes par couche

- `lo[il]` = premier slot de scratch de la couche, `hi[il]` = un passé le dernier.
- Les `qi` premiers slots `[lo[il], lo[il]+qi)` sont les **résidents seedés**, jamais évincés.
- Les slots `[lo[il]+qi, hi[il])` sont le LRU **propre à cette couche**.
- `plan(il, ...)` ne prend `head[chain[il]]` que dans cette plage ; `touch()` ne touche que
  `slots[e] >= lo[il]`. L'éviction cross-couche est **structurellement impossible**.
- Capacité : `uniq.size() > hi[il] - lo[il]` → `plan()` renvoie false → pas de GGML_ABORT.

### Second bug trouvé : `init_chains()`

`init_chains()` parcourait **toutes** les couches (`lo.size() == n_layer == 41`). Les couches
hors groupe ont `lo=0, hi=n_slots, chain=0` → la fonction réinitialisait la tête/queue de la
chaîne 0 et réécrivait `prev/next` sur `[0, n_slots)`, **détruisant les chaînes par couche du
groupe** → CUDA illegal access.
**Fix : passer `g.layers`** (`src/llama-moe-cache.cpp:604`), pas la liste des couches du modèle.

## Expériences vérifiées (cette session)

Commande de base : `-cmoe -ngl 99 --moe-cache-mib 1024 --moe-profile /tmp/prof2.csv
--moe-cache-static -c 2048 --temp 0 -s 42`, prompt `Q: What is the capital of France? A:`.

| # | config | résultat |
|---|---|---|
| lru3 | `LLAMA_MOE_STATIC_Q=1 LLAMA_MOE_SEED_MAX_LAYERS=2 -n 8` (corrompu avant) | **CORRECT == base8** |
| lru4 | budget par défaut (q=20/19/19), `-n 8` | **CORRECT**, hit 39,58 % |
| lru128 | budget par défaut, `-n 128` | **CORRECT == base128**, hit 48,77 % |
| g0b | `SCRATCH_DIV=3 STATIC_Q=18 SEED_MIN_LAYERS=37` (crash CUDA avant) | **CORRECT**, hit 38,12 % |
| lru512 | budget par défaut, `-n 512` | **identique à base512**, hit 48,94 % |
| lruonly8 | mode LRU (sans `--moe-cache-static`), `-n 8` | **== base8** |
| lruonly128 | mode LRU, `-n 128` | diverge de base128 (divergence de layout, amont) mais **bit-identique à `/tmp/lru_master.txt`** → chemin LRU inchangé |
| clean8 | build **nettoyé des TEMP**, `-n 8` | **CORRECT == base8**, hit 39,58 % |
| clean128 | build nettoyé, `-n 128` | **CORRECT == base128**, hit 48,77 % |
| clean512 | build nettoyé, `-n 512` | **CORRECT == base512**, hit 48,94 % |
| lru8 | mode LRU, `-n 8` | **== base8**, hit 41,19 % |
| lru128 | mode LRU, `-n 128` | **bit-identique à `/tmp/lru_master.txt`**, hit 52,95 % |

**Re-vérification après les deux commits : les 5 runs sont bit-identiques aux baselines**
(`diff` = 0 ligne, en filtrant la ligne `build : b11553-5bfb641e4` qui change avec le commit).

## Commits faits

- `5bfb641e4` `moex: record which experts each MoE layer selects`
  (`src/llama-moe-profile.{h,cpp}`, `src/CMakeLists.txt`, `src/llama-graph.{h,cpp}`,
  `src/llama-context.{h,cpp}`, `src/llama-cparams.h`, `include/llama.h`, common/*,
  `--moe-profile-output`). **Build vérifié seul** (arbre sans les modifs de cache → compile).
- `1a1811266` `moex: seed the expert cache from recorded routing`
  (`src/llama-moe-cache.{h,cpp}` + les options `--moe-profile`, `--moe-cache-static`, `--moe-stats`).
- `de0531d8f` `Keep MoE experts readable from RAM during decode` — **12 fichiers, 300 insertions,
  6 deletions** : `src/llama-mmap.{h,cpp}` (populate/pin/release + `POPULATE_SUPPORTED` /
  `PIN_SUPPORTED` + `align_full()`), `src/llama-model.{h,cpp}` (`mmap_range()`),
  `src/llama-moe-cache.{h,cpp}` (`cache_complement()`, `pin_bytes`), `include/llama.h`,
  `src/llama-cparams.h`, `src/llama-context.cpp`, `common/common.{h,cpp}`, `common/arg.cpp`
  (option `--moe-cache-pin-mib`).
- `1a6b4a0b0` `Keep the whole GGUF readable from RAM during decode` — **3 fichiers, 45 insertions,
  2 deletions** : `src/llama-model.{h,cpp}` (`n_mappings()`, `mapping(i)`),
  `src/llama-moe-cache.cpp` (peuplement des **gaps** entre les plages d'experts dans
  `cache_complement()`, champ `rest_bytes` dans le log). **Les 6 diffs de `verify.sh` = 0.**
- `47a1dbbfc` `Compute MoE experts missing from the cache on the CPU` — **11 fichiers, 217 insertions,
  57 deletions** : `common/arg.cpp` (option `--moe-cache-cpu-fallback`), `common/common.{h,cpp}`,
  `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp`, `src/llama-graph.{h,cpp}`
  (`ids_cpu` dans `build_lora_mm_id`, `ids_cpu_rows` dans `build_moe_cache_slots`),
  `src/llama-moe-cache.{h,cpp}` (slot zéro, carte CPU par couche, `copy()` court-circuitée),
  `ggml/src/ggml-cpu/ggml-cpu.c` (saut des ids `-1` + mise à zéro des lignes non écrites).
  **Les 7 diffs de `verify.sh` = 0** (le run `cpufb` a été ajouté).

Git n'a pas d'identité configurée sur la machine (`yves@caspar.(none)` est refusé) → commits
avec `git -c user.name="yves" -c user.email="yves@caspar.local"` (aucun fichier de config modifié).

## Parallelisme du build (mesuré cette session)

`-j8` **est bien honoré** : sur un rebuild complet (453 étapes, `touch include/llama.h`) la
sonde `tools/measure_parallel.sh` compte **max 8** processus de compilation concurrents,
**moyenne 7,24**. Les phases mono-cœur observées sont : rebuild d'un seul fichier (1 job),
**étapes de link** (`g++`/`ld` = 1 processus), et `ggml-cuda.cu` (**253 Ko, une seule
translation unit** → un seul `nvcc`). `ggml-cuda` est découpé en **144** `.cu` → parallélisable.
**Unity build NON activé** (aucune clé `unity` dans `CMakeCache.txt`).
`GGML_CCACHE=ON` mais **ccache absent** → chaque rebuild repart de zéro ; le warning se coupe
avec `-D GGML_CCACHE=OFF`. `ninja` n'est pas installé (generator = Unix Makefiles).

## Mesure du chemin de décode (nouvel outil, cette session)

`tools/majflt.py` + `tools/run_majflt.sh` sur le run statique `-n 128` :

- `majflt` monte à **293 565** pendant le chargement du modèle ;
- **+85** faultes majeures **pendant le décode** ;
- `read_bytes` **+246 KB pendant le décode**.

→ **Ces chiffres étaient faux** : `tools/majflt.py` lisait `f[6],f[7]` = flags/minflt. Avec
`tools/majflt2.py` (champs corrects) **majflt = 0 pendant tout le décode**, et la campagne
pin7 → pin22 (voir plus bas) montre qu'à **page cache chaude la fenêtre de génération ne lit
rien** → **exigence 5 remplie** grâce au complément RAM (`de0531d8f`).

## Commandes

```
# build (Makefile generator, pas de ninja ; arch 61)
cd /data/o35/llama.cpp && /snap/bin/cmake --build build --config Release -j8 2>&1 | grep -Ei "error|warning" | head -20; echo "BUILD_DONE"

# vérification complète : 7 runs prompt A + 3 runs prompt B, masters dans /data/o35/o35/masters
# — bash, pas sh ; enlever les marqueurs avant
rm -f /tmp/*.done && bash /data/o35/o35/tools/verify.sh

# run (-v est OBLIGATOIRE pour voir LLAMA_LOG_INFO ; --no-display-prompt, pas --no-display)
cd /data/o35/llama.cpp && timeout 1200 ./build/bin/llama-cli -v \
  -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 --moe-cache-mib 1024 \
  --moe-profile /tmp/prof2.csv --moe-cache-static -c 2048 -n 128 --temp 0 -s 42 \
  -p "Q: What is the capital of France? A:" --no-display-prompt > /tmp/out.txt 2>/tmp/out.log

# comparaison numérique
diff <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/base128.txt) <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/out.txt)

# fautes / corruption
grep -Ei "ABORT|CUDA error|stale|clobber|mismatch|hit rate" /tmp/out.log

# evict du page cache, run froid mesuré (majflt + secteurs lus sur sda), puis fenêtrage
python3 /data/o35/o35/tools/evict.py 13
STATIC=1 sh /data/o35/o35/tools/run_pin.sh pin7 --moe-cache-pin-mib 1990 > /tmp/pin7.samples 2>&1
python3 /data/o35/o35/tools/phase2.py /tmp/pin7.samples /tmp/pin7.log

# run froid avec /proc/PID/io (python est le parent) + résidence du fichier + swap + blocs de GiB
python3 /data/o35/o35/tools/evict.py 13 && python3 /data/o35/o35/tools/runio3.py pin22 1024 1990 > /tmp/pin22.samples 2>&1
python3 /data/o35/o35/tools/phase4.py /tmp/pin22.samples /tmp/pin22.log
python3 /data/o35/o35/tools/resident.py          # résidence par GiB en une passe
# 4ᵉ argument de runio3.py : "lru" retire --moe-cache-static, un chiffre remplace -c

# run froid avec strace : attribution des read() par fichier
python3 /data/o35/o35/tools/evict.py 13 && sh /data/o35/o35/tools/runstrace.sh pin23 1024 1990
python3 /data/o35/o35/tools/strace_analyze.py /tmp/pin23.strace /tmp/pin23.log

# run froid + échantillonnage 20 ms (résidence par fenêtre de 64 MiB en rotation)
python3 /data/o35/o35/tools/evict.py 13 && python3 /data/o35/o35/tools/runio6.py pin27 1024 1990 > /tmp/pin27.samples 2>&1

# run froid + résidence du fichier ENTIER par bloc de 1 GiB (cadence 130 ms)
python3 /data/o35/o35/tools/evict.py 13 && python3 /data/o35/o35/tools/runio7.py pin29 1024 1990 > /tmp/pin29.samples 2>&1

# run froid avec le fallback CPU : 4ᵉ argument "fb" de runio7.py ajoute --moe-cache-cpu-fallback
python3 /data/o35/o35/tools/evict.py 13 && python3 /data/o35/o35/tools/runio7.py cpufb 1024 1990 fb > /tmp/cpufb.samples 2>&1
python3 /data/o35/o35/tools/phase6.py /tmp/cpufb.samples /tmp/cpufb.log   # index corrects, résidence par bloc

# run chaud avec le fallback CPU (page cache plein, pas d'evict)
cd /data/o35/llama.cpp && timeout 1200 ./build/bin/llama-cli -v \
  -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 --moe-cache-mib 1024 \
  --moe-cache-static --moe-cache-cpu-fallback --moe-profile /tmp/prof2.csv -c 2048 -n 1024 \
  --temp 0 -s 42 -p "Q: What is the capital of France? A:" --no-display-prompt \
  > /tmp/cpufb512.txt 2>/tmp/cpufb512.log
grep -o "eval time =.*tokens" /tmp/cpufb512.log | tail -1
diff <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/clean512.txt) <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/cpufb512.txt)

# analyse du profil
cd /data/o35/o35 && python3 tools/analyze_moe_profile.py /tmp/prof2.csv --expert-used 8 --sizes 10,20,30,40,50
```

Références texte (prompt `Q: What is the capital of France? A:`, `-s 42 --temp 0`) :
`/tmp/base8.txt`, `/tmp/base128.txt`, `/tmp/base512.txt` (sans cache), `/tmp/lru_master.txt`
(LRU, référence du chemin LRU — diverge de base128 pour cause de layout, amont).

## Géométrie des groupes à 1024 MiB (vérifiée)

- **groupe 0 = 37 couches `[0..33, 35, 36, 37]`, iq3_xxs, 1036 slots**
- **groupe 1 = 2 couches `[34, 38]`, iq4_xs, 55 slots**
- **groupe 2 = 1 couche `[39]`, iq4_xs, 27 slots**
- MTP `blk.40` non caché. Total 1118 slots, 1020 MiB résidents.
- Budget par défaut : groupe 0 → q=20, groupe 1 → q=19, groupe 2 → q=19.
- `max_batch_cached = 1` pour tous les groupes → **cache réservé au décode**, prefill non caché.

## Code : emplacements exacts (`/data/o35/llama.cpp/src/llama-moe-cache.cpp`, 1067 lignes)

- `moe_cache_lru` :22 — membres `slot_map`, `key_of`, `prev`, `next`, `lo`, `hi`, `chain`,
  `head`, `tail`, `seen`, `seen_gen`, `uniq`, `static_mode`, `n_resident`.
  `init()` :47 (`lo/hi/chain/head/tail` initialisés), `seed()` :68 (`lo[il]++`, `n_resident++`),
  **`init_chains(g.layers)` :82** (ne parcourir que les couches du groupe), `touch(s, il)` :103,
  `plan(il, ids, ...)` :124 (capacité `hi[il]-lo[il]`, `head[chain[il]]`, `n_scratch`).
- `llama_moe_cache_load_profile()` :214 (header `layer,expert,count,weight_sum\n`, tri count
  décroissant puis index croissant ; `weight_sum` lu mais ignoré).
- `impl` :290 — `group.max_batch_cached` par groupe :293 ; `max_batch = 32` :315 ;
  `max_batch_scratch = 1` :318 ; `static_mode` :321 ; **`pin_bytes` :324**.
- **`cache_complement()` :354-476** (voir la section dédiée).
- **bloc statique :689-762** — garde `static_mode && !model.hparams.no_alloc` :689,
  `n_scratch`, garde `n_scratch < n_expert_used*n_layers`, `budget`/`q`/`extra`
  (bug `extra` corrigé : 0 quand q==0), répartition `qi`/`sci` et plages `lo[il]`/`hi[il]`,
  `g.max_batch_cached = mb`, **`g.lru.init_chains(g.layers)`**, upload des seedés sur tous les
  slots `key_of[s] >= 0`, logs `uploaded %.2f MiB` et `static MoE cache: ...`.
  **`cache_complement(model, ranked)` :761** — appelé une fois, après la boucle par groupe.
- `get_slot_map()` :797 — `n_tokens > g.max_batch_cached` **ou**
  `n_tokens*n_expert_used > hi[il]-lo[il]` → nullptr.
- `copy()` :850-924 — lookup GET_ROWS, GGML_ABORT si top-k dans le même split, `plan()`,
  uploads de miss fusionnés :909, stats, puis
  `ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src))` :922
  (la copie suivante synchronise le backend avant de modifier à nouveau le slot map).
- `log_stats()` :927, `export_stats()` :985 (+ `--moe-stats`).
- Constructeurs : `impl(..., int64_t pin_mib)` :479, wrapper `llama_moe_cache(..., int64_t pin_mib)`
  :1026 — **un seul site de construction : `src/llama-context.cpp:497`**.

**Tous les blocs TEMP sont supprimés** (diags `n_diag*`, `n_copy_calls`, vérifs byte-à-byte,
`scratch`/`next_scratch()`/`init_scratch_lru()`, `#include <cstdlib>`, knobs `LLAMA_MOE_*`).
Le build est vert après suppression.

## Mécanique du scheduler (vérifiée en lisant le code)

- `ggml_backend.cpp:1835-1838` — callback de copie uniquement pour les poids host.
- `llama-graph.cpp:2472-2509` (`build_moe_cache_slots`) — `slots = ggml_get_rows(ctx0, slot_map, ids)` ;
  `slot_map` est un poids host → le split démarre à ce nœud **seulement si `split->n_inputs > 0`** ;
  `ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il))`.
- `llama-context.cpp:2696-2771` (`sched_copy_experts`) — d'abord le chemin slot map
  (`moe_cache->copy()`), puis le chemin tensor complet d'experts
  (`node->op == GGML_OP_MUL_MAT_ID && node->src[0] == dst`) utilisé par `-cmoe` simple.
- `USE_GRAPHS = 1` sur ce build ; CUDA graphs désactivés sur sm_61
  (`disabling CUDA graphs due to GPU architecture`).

## Géométrie du modèle (ground truth)

`/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (arch `qwen35moe`) : 256 experts, 8 utilisés,
41 blocs (blk.0..39 tronc + blk.40 MTP), n_embd 2048, n_ff_exp 512, fichier 11,88 GB,
experts 10,06 GB (95,6 %), **0,913 MiB par expert**, tensors gate/up/down séparés.
**Règle : expert `e` du tensor `w` = `(uint8_t*)w->data + e*w->nb[2]`, taille `w->nb[2]`.**

## Courbe de couverture (`/tmp/prof2.csv`, 1026 tokens, 82 % des paires vues)

```
   size%    experts slots/ly     hit%  hit%(q)    mass%
    10.0       1024       25    51.74    50.75    56.66
    20.0       2048       51    71.46    71.13    75.37
    30.0       3072       76    83.26    82.56    86.15
    40.0       4096      102    90.77    89.81    92.74
    50.0       5120      128    95.36    94.30    96.45
```

Strata mesure `h_expert = 0,6447` (leave-one-out) contre `h_layer = 0,0456` : « seulement 4,6 %
des paires (layer, token) ont leurs dix experts résidents → pas de kernel groupé par couche ».

## Le complément RAM du cache (porté cette session, exigences 4 et 5)

**But** : les experts des couches cachées que le profil n'a pas mis en banque (le complément,
**8 618,06 MiB** sur 10 240 tensors d'experts) ne doivent **jamais** être lus sur le HDD pendant
le décode. Ils sont faultés dans le page cache, les plus chauds sont **mlockés** (inévictables),
et les pages des résidents sont **relâchées** après leur upload en banque.

**Ordre exact dans `cache_complement()`** : (1) `release()` des plages résidentes → libère le page
cache ; (2) `populate()` du complément, plages contiguës fusionnées par mapping (une `madvise` par
bloc) ; (3) `pin()` des experts les plus chauds du complément dans l'ordre `ranked[il]`, un
`mlock` par expert (0,913 MiB par appel), arrêt à la limite du budget ou si `pin()` échoue.
Gardes : `llama_mmap::POPULATE_SUPPORTED` / `PIN_SUPPORTED`, et seulement
`static_mode && !model.hparams.no_alloc`.

**`unmap_fragment` est DANGEREUX et n'est PAS utilisé** : `copy_experts()` lit les tensors d'experts
host directement dans le mmap pendant le prefill (`llama-context.cpp:2696-2771`) → unmapper =
SIGSEGV. `MADV_DONTNEED` garde le mapping valide ; une lecture ultérieure refaulte depuis le disque
(le prefill paie, le décode ne paie pas — l'exigence 5 ne concerne que le décode).

**Contrainte dure** : `RLIMIT_MEMLOCK` soft = hard = **2 088 734 720 octets = 1,945 GiB**
(`python3 -c "import resource; print(resource.getrlimit(resource.RLIMIT_MEMLOCK))"`, `ulimit` est
indisponible dans mvdan/sh) → mlock ne peut pas verrouler 10 GB, seulement ~2 130 experts
(20 % du complément). Le noyau **6.8.0-138-generic** (≥ 5.11) ne compte plus RLIMIT_MEMLOCK pour
`mlock` (comptabilité cgroup) → `--moe-cache-pin-mib 1990` passe au-delà de la limite.
`MADV_POPULATE_READ` (Linux 5.14+) est disponible et ne demande aucun privilège.

**Nouveau code** :
- `src/llama-mmap.h` après `unmap_fragment` : `populate(first,last)`, `pin(first,last)`→bool,
  `release(first,last)`, `POPULATE_SUPPORTED` (`MADV_POPULATE_READ`), `PIN_SUPPORTED`.
- `src/llama-mmap.cpp` : `align_full()` (page-alignée **complète**, la `align_range()` :531 saute la
  première page partielle → inutilisable ici), Linux `MADV_POPULATE_READ`/`mlock`/`MADV_DONTNEED`,
  Windows `VirtualLock` + no-op, fallback `throw`, forwarders.
- `src/llama-model.h` : `struct llama_mmap;` (llama-model.h **n'inclut pas** llama-mmap.h) et
  `llama_mmap * mmap_range(const ggml_tensor * t, size_t & first, size_t & last) const`.
- `src/llama-model.cpp:2382` : `mmap_range()` cherche `t->data` dans `pimpl->mappings`
  (déclaré :1242, rempli :2013-2014, `init_mappings(true,...)` :1870).
- `src/llama-moe-cache.cpp` : `#include "llama-mmap.h"`, `pin_bytes`, `cache_complement()`.
- Option `--moe-cache-pin-mib` (int64_t, défaut 0 = populate seul, jamais de lock) : `include/llama.h`,
  `src/llama-cparams.h`, `src/llama-context.cpp` (3 sites), `common/common.h`, `common/common.cpp`,
  `common/arg.cpp`. Les lambdas de `common_arg` n'ont **pas** d'overload int64_t/size_t → `int`.

**Bug trouvé par la mesure** : `idx` était `std::vector<int32_t>` (un index par `(il,e)`) alors que
chaque couche a **3 tensors d'experts** (gate/up/down) → seules 1/3 des plages résidentes étaient
marquées. Le premier run rapportait `314,09 MiB de pages résidentes relâchées` au lieu de
`727,94 MiB`. Corrigé en `std::vector<std::vector<int32_t>> idx`.

**Mesures (page cache évicté par `evict.py 13` : 12,3 GB → 1,68 GB)** :

| run | config | majflt | disque total | disque dans la fenêtre de génération |
|---|---|---|---|---|
| pin5 | statique, populate seul, `-n 128` | **0** | 8 603 MiB | **88,65 MiB** |
| pin6 | LRU (option pin ignorée), `-n 128` | **0** | 8 572 MiB | 89,71 MiB |
| pin4 | statique, populate seul, `-n 128` | **0** | 9 556 MiB | 89,08 MiB |

Output de pin5 **bit-identique à `/tmp/clean128.txt`** (seules la ligne de spinner et la ligne de
timing diffèrent). La fenêtre de génération est alignée sur les timestamps du log (`M.SS.mmm`) avec
`tools/phase2.py` : le delta de 89 MiB est **le même pour le LRU** → bruit ambiant, pas le cache.

## La campagne de mesure pin7 → pin22 (cette session)

**Outils écrits cette session** (tous dans `/data/o35/o35/tools/`) :
- `runio.py nom n pin` — **python est le parent de llama-cli** → `/proc/PID/io` lisible sous
  `ptrace_scope=1`. Colonnes : t, minflt, majflt, rss, read_bytes, rchar, write_bytes, disk.
- `runio2.py` — ajoute la **résidence du fichier** (`mincore` en ctypes sur le GGUF mappé en
  `MAP_SHARED PROT_READ`).
- `runio3.py` — ajoute `pswpin/pswpout` (`/proc/vmstat`) et la **résidence par bloc de 1 GiB** ;
  4ᵉ argument : `"lru"` retire `--moe-cache-static`, un chiffre remplace la valeur de `-c`.
- `resident.py` — rapport de résidence par GiB en une passe.
- `phase2.py` / `phase3.py` / `phase4.py` — découpent les samples en phases (fenêtre de génération
  alignée sur `eval time`) : 5, 8 et 7 colonnes.
- `run3.sh` — evict → pin7 (`-n 128`) → evict → pin8 (`-n 512`).

**Champ RSS trouvé** : dans `/proc/PID/stat` après le token `)`, **`f[21] = rss`** (vérifié contre
`VmRSS`) ; `f[19] = vsize`.

Mesures, page cache évicté (`evict.py 13`), fenêtre de génération alignée par `phase2.py` :

| run | config | lectures dans la fenêtre | majflt | notes |
|---|---|---|---|---|
| pin7 | statique + pin 1990, `-n 128` | 50,47 MiB | 0 | output == clean128, hit 48,77 % |
| pin8 | statique + pin 1990, `-n 512` | 68,46 MiB | 0 | output == clean512, hit 48,94 % |
| pin9–pin13 | idem, `-n 512`, sampler python | 80–89 MiB | 0 | RSS 10,67 GiB au début du décode |
| pin14 | statique + pin 1990, `-n 1024` | 86,63 MiB | 0 | 176 tokens seulement (EOS) |
| pin15 | statique, populate seul, `-n 1024` | 86,50 MiB | 0 | |
| pin16 | statique + pin 1990 + mincore, `-n 1024` | 87,80 MiB | 0 | résidence **11,07 GiB complète** |
| **pin17** | **page cache CHAUD**, pin 1990, `-n 1024` | **0,00 MiB** | 0 | disque 0,04 MiB, résidence complète |
| pin18 | + compteurs swap | 87,80 MiB | 0 | `pswpin` ≈ 1 par échantillon → swap hors cause |
| pin19 | **LRU**, `-n 1024` | 88,52 MiB | 0 | hit 51,70 % |
| pin20 | pin 1990, `-c 512` | 86,82 MiB | 0 | hit 48,94 % |
| pin21 | pin 1990, `release()` désactivé (TEMP) | 81,48 MiB | 0 | à peine changé |
| pin22 | pin 1990, résidence par GiB | 13–16 MiB par 0,7 s | 0 | les 11 blocs à 1024 MiB résidents |

**Résultat clé** : à **page cache chaude la fenêtre de génération ne lit rien** (0,00 MiB lus,
0,04 MiB disque, majflt 0) → **exigence 5 remplie**. Le disque ambiant **sans aucun processus** est
mesuré à **0,05 MiB en 10 s** → les ~87 MiB des runs froids sont des lectures réelles.

**Question ouverte — RÉSOLUE** (sondes pin23 → pin29, voir la section suivante) : les lectures
froides ne sont **pas** des cycles drop+refill des plages d'experts (leur résidence ne baisse
jamais), ce sont les **pages du GGUF hors plages d'experts** (queue du fichier, MTP `blk.40`,
attention, embeddings) qui n'étaient pas peuplées, plus les évictions du noyau sous la pression de
mémoire. Le pining ne réduit pas le nombre de lectures ; son bénéfice mesurable est la **garantie
que les experts les plus chauds survivent** à la pression de mémoire.

## Les sondes pin23 → pin29 (cette session)

**Outils ajoutés** : `runstrace.sh nom n pin` (strace sur `read/pread64/readv/preadv`, `-yy -tt`),
`strace_analyze.py samples log` (somme des octets lus par fichier, fenêtre alignée sur `eval time`),
`runio5.py` (échantillonnage 20 ms, parent python), `runio6.py` (8 fenêtres `mincore` de 64 MiB en
rotation), **`runio7.py`** (`mincore` du fichier entier agrégé par bloc de 1 GiB, cadence 130 ms —
une passe complète coûte 131 ms, le comptage 2 ms).

| run | config | lectures dans la fenêtre | résidence |
|---|---|---|---|
| pin23/pin24 | froid + strace | **0 `read()` sur le GGUF** dans la fenêtre | — |
| pin25 | froid + strace + 20 ms | 253,94 MiB au total (dont la fin du peuplement), **lecture arrêtée à t≈102 s** | — |
| pin26 | + `mincore` 8×64 MiB | cassé (type d'argument `mincore`) | — |
| pin27 | froid + 8 fenêtres 64 MiB, 20 ms | 64,68 MiB, disque 66,06 MiB | **les 8 fenêtres à 64 MiB résidentes en permanence** (couvre 0–7,5 GiB, la queue 7,5–11,07 GiB non sondée) |
| pin28 | froid + `mincore` fichier entier, 130 ms | 207,52 MiB, disque 208,16 MiB | blocs 0–9 à 1024 MiB, **bloc 10 : 918→1024**, **bloc 11 : 0→68** |
| **pin29** | froid + **peuplement du fichier entier** (`rest` = 1986,48 MiB) | **112,69 MiB**, disque 113,43 MiB | blocs 0–10 résidents au début du décode, bloc 11 à 0 puis 16 |

**Ce que les sondes prouvent** :
- `strace` : **aucun appel `read()` sur le GGUF pendant la génération** (20,98 MiB / 5372 appels,
  tous au chargement). Dans la fenêtre : seulement `/tmp/prof2.csv` 0,15 MiB.
- `majflt` **n'est pas** un indicateur de lecture disque sur ce noyau : le test `/tmp/faulttest2.py`
  (100 MiB froids du GGUF mappé, après evict) donne `minflt += 1604`, **`majflt += 0`**,
  `read_bytes += 100,25 MiB`. Seuls `/proc/PID/io` (`read_bytes`) et `/proc/diskstats` sont fiables.
- La résidence des **plages d'experts ne baisse jamais** pendant la génération → le complément
  `populate`+`pin` fonctionne.
- Les lectures froides se concentrent dans les **1,2 première seconde** de génération
  (27+27+27+9,8 MiB) puis tombent à ~1–4 MiB par échantillon.
- RSS chute de 11,44 → 10,32 GiB à t≈112 s : le noyau **évicte les pages mappées** sous la pression
  (11,2 GiB de page cache + ~1 GiB d'anon sur 15 GiB de RAM).
- **Peupler le fichier entier** (la queue du GGUF hors plages d'experts) **réduit les lectures
  froides de 207,52 → 112,69 MiB** (pin28 → pin29). Le coût : +1986,48 MiB lus au chargement
  (chargement 90 s → 100 s), et la queue est évictée juste avant le décode (bloc 11 à 0 au début).

**Conclusion sur l'exigence 5** : à page cache chaude, 0 MiB lu pendant la génération. À froid,
le residuel est de ~112 MiB concentré sur la première seconde, et il vient de la **queue du fichier**
non peuplée + des évictions sous pression, pas des plages d'experts. La seule levier restant est de
piner plus de complément, borné par `RLIMIT_MEMLOCK` = 1,945 GiB.

## Prochaine étape exacte

1. ~~Re-vérification du build nettoyé~~ **fait** : clean8 == base8, clean128 == base128,
   clean512 == base512, lru8 == base8, lru128 == `/tmp/lru_master.txt`.
2. ~~Commit n°1 (instrumentation)~~ **fait** : `5bfb641e4`.
3. ~~Commit n°2 (seed statique + metrics)~~ **fait** : `1a1811266`.
4. ~~Commit du complément RAM~~ **FAIT** : `de0531d8f` (12 fichiers, 300 insertions, 6 deletions).
5. ~~Re-vérification avec le build du complément~~ **FAITE** : `verify.sh` **contient maintenant**
   le run `pin128` (`--moe-cache-pin-mib 1990 -n 128`) diffé contre `clean128`.
   **Les 6 diffs = 0** : clean8==base8, clean128==base128, clean512==base512, lru8==base8,
   lru128==lru_master, **pin128==clean128** (hit rate 48,77 % identique, `727,94 MiB de pages
   résidentes relâchées` = `release()` actif, `mlock` arrêté à 1986,96 MiB / 6635 experts).
   **Ne pas oublier `rm -f /tmp/*.done` avant `verify.sh`** (les marqueurs font sauter les runs).
6. ~~Résoudre la question ouverte~~ **FAIT** (pin23 → pin29) : les lectures froides ne sont pas des
   cycles sur les plages d'experts (résidence constante), elles viennent de la **queue du fichier
   hors plages d'experts** + des évictions sous pression. `runio7.py` (peuplement du fichier entier)
   **réduit les lectures de 207,52 → 112,69 MiB**.
7. ~~Fallback CPU par expert (exigence 6)~~ **FAIT** (`47a1dbbfc`) : la granularité **de couche** est
   mesurée **sans intérêt** (chaleur uniforme : 8384 sélections par couche sur 40, 2,50 % chacune) ;
   la granularité **par expert** est portée (slot zéro + carte CPU + `ggml_add`). Voir la section
   « Fallback CPU porté ».
8. `IMPLEMENTATION_NOTES.md` : chaque mécanisme réutilisé/adapté/rejeté → file:line Strata. **FAIT**
   (+ le slot zéro, sans analogue Strata).
9. **Batch de 2-4 tokens en décode : sans objet, vérifié.** `max_batch_cached = 4` est autorisé en mode
   fallback, mais `llama-cli` en décode glouton traite **1 token par ubatch** (les `n_tokens = 1, 2, 4, 8,
   …` du log sont les réservations de graphes, pas des batches réels) → la limite de 4 ne s'applique
   jamais. Il n'y a **pas de chemin de décode multi-token actif** (`model has unused tensor
   blk.40.nextn.eh_proj.weight ... ignoring` = MTP absent). Le gain n'existe que pour un décode
   parallèle/spéculatif, donc **la priorité est d'activer MTP**, pas d'augmenter le batch.
10. ~~Isoler l'effet du `pin` sur le run froid~~ **FAIT** : `cpufb` **avec** `pin 1990` lit
    **170,02 MiB (`read_bytes`) / 172,42 MiB (disk)** dans la fenêtre de décode, **sans** `pin`
    **290,99 / 291,66 MiB** → **−119 MiB (−41 %)**, t/s 13,62 contre 13,40 (sans effet). Le résidu de
    lecture est la **queue du fichier** (bloc de GiB 10, +118 MiB), concentrée sur les 2 premières
    secondes du décode.
11. **Prouver que la divergence `cleanB`/`cpufbB` n'est pas un bug** — **FAIT cette session** :
    sonde dans `src/llama-graph.cpp` (`LLAMA_MOE_PROBE=1`, patch `tools/moe_probe.patch`) comparant,
    pour chaque expert **résident**, le `mul_mat_id` **CPU** sur les poids hôtes au résultat du
    **slot GPU**. Mesuré sur prompt B : **max abs 0,020653, mean abs pondéré 0,003236 sur
    1 968 128 éléments, max rel 7,455 (couche 0), moyenne des max-rel 1,41** → **0,3 % de
    l'amplitude typique**, pas du 1e-6 → **différence de noyau MMQ CPU vs MMVQ CUDA**, pas un bug.
    **FAIT** : sonde supprimée (`git checkout -- src/llama-graph.h src/llama-graph.cpp
    src/llama-context.cpp`), rebuild propre à `47a1dbbfc`, `verify.sh` → **7 diffs à 0**. Le patch de
    la sonde est sauvegardé dans `tools/moe_probe.patch` (`git apply` pour le rejouer).
12. **Ajouter prompt B à `tools/verify.sh` comme régression — FAIT cette session** : `runB` /
    `checkB` (comparaison via `tools/cmp.py`) ; les masters sont copiés dans
    **`/data/o35/o35/masters/`** (`cleanB.txt`, `cpufbB.txt`, `allcpuB.txt`, `baseB.txt`) parce que
    `runB` **écrase** `/tmp/$name.txt` à chaque run — comparer un run à son propre fichier `/tmp`
    donnait **0 par construction**, une vérification **vide de sens**. Chaque mode doit donner
    **diff 0 contre son master** ; `cpufbB` vs `cleanB` = **121** (préfixe 1514), `cpufbB` vs
    `allcpuB` = **130** (préfixe 795), `allcpuB` vs `cleanB` = **137** (préfixe 795). Les valeurs
    **8 et 28** attendues auparavant étaient **fausses** (texte brut non filtré) et produisaient
    deux **REGRESSION FAIL** ; la vérification complète repasse maintenant **toute verte**.
13. **Activer MTP** (`blk.40.nextn.eh_proj.weight` est ignoré) pour que des batches de 2-4 tokens
    existent et que `max_batch_cached = 4` serve ; mesurer le t/s à `-n 1024`.
14. **Premier essai sans MTP — FAIT cette session** (prompt B, `-n 1024`, froid, `pin 1990`, nouvel outil
    `tools/runtrial.py`) : **16,26 t/s** et **0,05 MiB** lus sur le disque pendant les 62,9 s de décode
    (chemin tout-GPU : **20,13 t/s** mais **114,81 MiB**) ; préfill **plus rapide** en fallback (24,20 contre
    17,22). Sortie cohérente. **L'exigence 5 est vérifiée sur un run long.**

## Fallback CPU (exigence 6) — analyse de cette session

**Mesure décisive** : la chaleur par couche dans `/tmp/prof2.csv` est **uniforme** — les 40 couches
ont **8384 sélections chacune** (2,50 % du total 335 184), la 41ᵉ (`blk.40`, MTP) 8208. Les couches
les plus chaudes et les plus froides sont **identiques**. Donc la granularité **de couche** ne donne
aucun gain : cacher les 8 couches les plus chaudes couvre 20,01 % des sélections, cacher 32 couches
80,04 %, et le budget de 1024 MiB (1036 slots) permet de loger **8 experts par couche × 129 couches**
— il y a 40 couches, donc **320 slots = 292 MiB** suffiraient pour loger les 8 experts de toutes les
couches. Le mode statique actuel alloue 20 slots résidents + 8 scratch par couche, hit 48,94 %.

**Coût dominant mesuré** : `uploaded = 26137,55 MiB` de PCIe pour 28608 misses sur une génération de
9,4 s → **2,78 Go/s de PCIe**, soit la quasi-totalité de la bande passante. Le calcul CPU des experts
non résidents est le levier, pas le choix des couches.

**Conception du fallback par expert (MVP B)** :
- Deux cartes par couche : `slot_map` (existant, slots GPU) et une carte d'indices host
  `ids_cpu[il][e] = e` si l'expert n'est **pas** résident, sinon `-1`.
- Deux `ggml_mul_mat_id` par tenseur : un sur le bank GPU (`slots`), un sur le tenseur d'experts
  host (`ids_cpu`) → le scheduler place le second sur le CPU (poids host), puis `ggml_add`.
- **Prérequis technique** : `ggml_mul_mat_id` doit **sauter** les ids `-1`. Côté CUDA,
  `mm_ids_helper` (`ggml/src/ggml-cuda/mmid.cu:33-108`) compte `expert_used < expert`, donc `-1`
  casse la compaction des colonnes et l'`expert_bounds` → il faut un garde. Côté CPU,
  `ggml-cpu/ops.cpp` doit ignorer `-1` de même.
- Rejeté : masquer les poids à zéro et laisser le GPU calculer (numériquement faux, coût identique).
- Contrainte sm_61 : les kernels MoE préfill de Strata sont `sm_75+` → inutilisables ; le chemin
  MMQ de llama.cpp est le seul disponible (pas de fusion MMVQ sur Pascal).

## Fallback CPU porté (exigence 6) — session du 12 octobre

**Commit `47a1dbbfc`** — 11 fichiers, 217 insertions / 57 deletions. Option
`--moe-cache-cpu-fallback` (`common/arg.cpp`, `common/common.{h,cpp}`, `include/llama.h`,
`src/llama-cparams.h`, `src/llama-context.cpp`).

**Le slot zéro** (conception **sans analogue** dans Strata) : `ggml_mul_mat_id` a un `n_expert_used`
**fixe par token**, on ne peut pas « sauter » un expert côté GPU. Chaque groupe reçoit
`n_extra = 2` slots de plus (`llama-moe-cache.cpp:671`) : le slot `n_slots` est un **bank d'octets à
zéro** (le buffer GPU est clearing à 0, `:724`), `n_slots+1` est la padding que CUDA lit au-delà de la
fin. Les experts non résidents pointent sur ce slot zéro (`:779-792`) → le GPU ne contribue rien.

**Carte CPU par couche** : `l.cpu_map` (I32 `[1, n_expert]`, host, créée à `:691-695`),
`cpu[e] = -1` si l'expert est résident, `= e` sinon. Dans le graphe (`src/llama-graph.cpp:2522`)
`ids_cpu_rows = ggml_get_rows(ctx0, cpu_map, ids)` **forcé sur `backend_cpu`**, reshapé
`[8, n_tokens]` ; `res_cpu = ggml_mul_mat_id(ctx0, w, cur, ids_cpu)` **forcé sur `backend_cpu`**
(`:1630`) puis `res = ggml_add(res, res_cpu)`. Quatre sites d'appel (`:2288`, `:2307`, `:2320`,
`:2421`). `build_moe_cache_slots` retourne tôt si `slot_map == nullptr` (`:2494`) → **le préfill n'est
pas touché**.

**Côté CPU** (`ggml/src/ggml-cpu/ggml-cpu.c`) : `if (i02 < 0) continue;` dans le groupage des lignes
(`:1655`) — les ids `-1` ne sont jamais assignés à une matrice — puis une boucle qui **met à zéro** les
lignes de `dst` correspondant aux ids négatifs (`:1667-1681`), car `ggml_add` additionne une ligne non
écrite. **Aucun kernel CUDA n'est modifié** (le `-1` n'atteint jamais le GPU).

**Mode statique + fallback** : **pas de LRU, pas de scratch** ; les 27 slots semés par couche restent
résidents tout le run ; `q = g.n_slots / n_layers` (27 pour le groupe 0), `g.max_batch_cached = 4`
(constante `:325`, bloc `:760-795`). `copy()` est court-circuitée pour `g.cpu_fallback` (`:992-1005`) :
compte hits/misses, **n'upload rien**, copie seulement la carte de slots. `get_slot_map` (`:895-905`)
saute la limite `n_tokens*n_expert_used > hi-lo` dans ce mode.

**Preuve de validité numérique** : des octets à zéro donnent des poids à zéro pour **tous** les types de
quant du modèle (`iq2_xxs`, `iq3_xxs`, `iq4_xs`, `iq2_s`, `q2_K`) — vérifié par `/tmp/zeroq.c` qui
déquantifie un bloc d'octets nuls pour chaque type.

**Cause racine du bug (résolu cette session)** : dans la boucle de mise à zéro, la ligne de `dst` était
indexée `dst->data + iid1*nb1 + id*nb2`, alors que le calcul écrit `dst->data + i1*nb1 + i2*nb2` avec
**`i1 = id`** (indice dans `n_expert_used`) et **`i2 = iid1`** (ligne de `src1`) — `ggml-cpu.c:1534`.
**Les deux strides étaient inversés.** Pour `dst = [512, 8, 2]` : `nb1 = 2048` octets, `nb2 = 16384`,
et `iid1=1, id=7` donnait `1*2048 + 7*16384 = 116320` octets dans un `dst` de `512*8*2*4 = 32768`
octets → **écriture hors des limites du tas**, logits absurdes, EOS après 1 token. Correctif :
`dst->data + id*nb1 + iid1*nb2`.

**Deux hypothèses fausses, éliminées** :
- `ggml_backend_dev_supports_op` refuserait un `mul_mat_id` CPU dont `src0` est en `CUDA_Host` :
  **faux**. `CUDA_Host` n'est **pas** dans `ggml_backend_cpu_get_extra_buffer_types()`
  (`ggml-cpu/ggml-cpu.cpp:42` → AMX, repack, spacemit seulement) ;
  `ggml_backend_cpu_device_supports_op` (`:424`) n'a **aucun cas `GGML_OP_MUL_MAT_ID`** →
  `default: return true` ; `CUDA_Host.is_host = ggml_backend_cpu_buffer_type()->iface.is_host`
  (`ggml-cuda.cu:1317`) → `ggml_backend_buft_is_host` vrai →
  `ggml_backend_cpu_device_supports_buft` (`:490`) vrai → **aucune copie de `w`**. Le calcul CPU
  s'exécute bien sur le CPU.
- « anomalie des graphes de 2 tokens » : **fausse piste**. `graph_reserve` réserve des graphes pour
  `n_tokens = 1, 2, 4, 8, …` et le calcul CPU s'y exécute légitimement. Le garde `!hparams.no_alloc`
  ne filtre **pas** les builds de `graph_reserve` (`no_alloc` est un champ de `llama_graph_params`,
  `hparams.no_alloc` un champ de `llama_hparams` du contexte de mesure mémoire — deux choses
  indépendantes). Le `sumabs_neg` non nul de certaines couches était la sonde lisant **hors limites**
  avec les mêmes strides inversés ; après le correctif `sumabs_neg = 0,0000` partout. MTP **absent**
  (`model has unused tensor blk.40.nextn.eh_proj.weight ... ignoring`).

**Résultats mesurés, page cache chaud** :
- `-n 128` : 128 tokens, **17,78 t/s**, **diff = 0** vs `/tmp/clean128.txt`.
- `-n 1024` : 176 tokens (EOS), **17,22 t/s**, **diff = 0** vs `/tmp/clean512.txt` (1715 octets
  identiques). Baseline statique `clean512` = **19,39 t/s** → **coût 11 %**.
- `uploaded = 0,00 MiB` pendant tout le décode ; hit rate **30,24 %** (contre 48,77 % en statique pur —
  en mode fallback il n'y a **ni LRU ni scratch**, seules les 27 slots semés par couche comptent comme
  hits) ; **99 341 calculs d'experts sur le CPU** contre 58 520 lignes du bank GPU (63 % du total) ;
  `1115/10496` experts résidents (1016,88 MiB), `40/41` couches cachées ; groupes : 1035 résidents /
  8437 au CPU (quota 27), 54/458 (quota 27), 26/230 (quota 26) ; `uploaded 929,88 + 56,53 + 30,47 MiB`
  de semis au chargement.

**Résultats mesurés, page cache froid** (`runio7.py cpufb 1024 1990 fb`) :
- fenêtre de décode 138,76 → 151,61 s : **`read_bytes + 170,02 MiB`, `disk + 172,42 MiB`, `majflt = 0`** ;
  baseline `pin29` (statique + pin 1990) : **173,30 / 174,11 MiB** → le fallback **n'ajoute aucune
  lecture disque**.
- la résidence ne bouge que dans le **bloc de GiB 10** (`906 → 1024`, +118 MiB) : la lecture vient de la
  **queue du fichier**, pas des plages d'experts ; les lectures sont **concentrées sur les 2 premières
  secondes** du décode (27 MiB par sample jusqu'à t = 140,66), puis **0,00 MiB** de t = 143,24 à la fin.
- `strace` dans la fenêtre : **0,15 MiB** de `read()` seulement (`tools/strace_analyze.py`) → le GGUF
  n'est pas lu par `read()`, tout passe par des page faults sur le mmap.
- t/s froid = **13,62** (coût des page faults en début de décode).
- **Le `pin` sert en mode fallback** : le même run froid **sans** `--moe-cache-pin-mib`
  (`runio7.py cpufb0 1024 0 fb`) lit **290,99 MiB (`read_bytes`) / 291,66 MiB (disk)** dans la fenêtre de
  décode (97,36 → 110,42 s), contre **170,02 / 172,42 MiB** avec `pin 1990` → **−119 MiB (−41 %)** ;
  t/s froid 13,40 contre 13,62 (sans effet). Le `pin` ne change pas le t/s, il **réduit les lectures**
  en protégeant 1,945 GiB du complément contre les évictions.

**Outil nouveau** : `tools/phase6.py` — fenêtrage avec les **index corrects** de la ligne de sample
(`f[4] = read_bytes`, `f[5] = secteurs lus × 512`, `f[6:] = résidence par bloc de GiB`), résidence par
bloc + distribution temporelle des lectures. `phase4.py` filtre `len(f) == 7` (inutilisable avec 12
blocs) et `phase5.py` lit `f[6]` comme disque (c'est `ms_reading`) → **les deux donnent des chiffres
faux**.

## Deuxième prompt (prompt B) — divergence, session du 12 octobre

Prompt B : `Q: Explain how a diesel engine works. A:`, `-c 2048 -n 256 --temp 0 -s 42
--no-display-prompt`. Quatre runs (`/tmp/runB.sh`, `/tmp/runAllCPU.sh`) :

| run | flags | t/s (generation) |
|---|---|---|
| `baseB` | **sans `-cmoe`, sans `-ngl`** → tout sur le CPU | **23,76** |
| `cleanB` | `-cmoe -ngl 99 --moe-cache-mib 1024 --moe-cache-static` | **21,32** |
| `cpufbB` | idem + `--moe-cache-cpu-fallback` | **18,64** |
| `allcpuB` | `--moe-cache-mib 32` + static + fallback → `q = 0`, **tous les experts sur le CPU** | **15,1** |

**Outil nouveau `tools/cmp.py`** : retire les lignes de bruit (`build      :`, `t/s`, `Loading model`,
spinner, `>`, lignes vides) **avant** toute comparaison, puis compte les lignes de diff et le
**préfixe commun** sur le texte filtré. Sans ce filtrage, le préfixe commun est de 75 caractères
(il s'arrête à la ligne `build`) et le diff compte 4 lignes de bruit de plus → **fausse mesure**.

**Sortie filtrée, sur prompt B** :

| paire | lignes de diff | préfixe commun (car.) | longueurs filtrées |
|---|---|---|---|
| `cleanB` vs `baseB` | **84** | 699 | 9027 vs 1758 |
| `cpufbB` vs `baseB` | **75** | 699 | 8136 vs 1758 |
| `allcpuB` vs `baseB` | **81** | 699 | 7923 vs 1758 |
| `cleanB` vs `cpufbB` | **121** | **1514** (16,8 % de 9027) | 9027 vs 8136 |
| `cleanB` vs `allcpuB` | **137** | 795 (8,8 % de 9027) | 9027 vs 7923 |
| `cpufbB` vs `allcpuB` | **130** | 795 | 8136 vs 7923 |

**Attention aux longueurs** : `baseB` est lancé avec `-n 256` (1758 caractères filtrés), les trois runs
de cache **sans `-n`** → ils génèrent jusqu'à EOS (9027, 8136, 7923 caractères). Les diffs contre `baseB`
sont donc dominés par la longueur ; **seul le préfixe commun 699 est comparable**. Les chiffres **21, 19,
33, 8, 28, 28** de la session précédente étaient des diffs **en texte brut**, pollués par la ligne
`build      :`, le spinner et la ligne de `t/s` (4 lignes de bruit par run) — **ils sont faux**, les
valeurs ci-dessus sont mesurées par `tools/cmp.py` sur le texte filtré.

**Lecture de ces chiffres — la cause dominante est le chemin dense, pas les experts** : `baseB`
tourne **sans `-cmoe` ni `-ngl`**, donc **les couches denses (attention, layernorm, projections
denses) sont calculées sur le CPU**, alors que les trois runs de cache les calculent **sur CUDA**.
`baseB` **n'est donc pas la baseline numérique des runs de cache.** La comparaison propre est
`cleanB` vs `cpufbB` : mêmes couches denses sur CUDA, mêmes ids d'experts, **seule la destination
de calcul des experts change** (bank GPU vs CPU) → **121 lignes de diff, préfixe commun 1514 sur
9027** → le fallback CPU diverge **après 16,8 % de la génération** (1514 caractères), et la divergence
est attribuée à la **différence de kernel CPU `MMQ` vs CUDA `MMVQ`** mesurée par la sonde
(**0,3 % d'écart relatif**), pas à une accumulation de points flottants.

**Le point de divergence sur prompt B** : à partir du caractère 699, `baseB` écrit
`**Understand User Input:**` et **les trois modes de cache écrivent tous** `**Analyze User
Input:**` — les modes de cache sont **d'accord entre eux** à cet endroit. C'est l'effet d'un
**quasi-égal de logits** en greedy : une différence de 1e-6 sur le logit d'un token de départ
choisit l'autre token, puis la divergence s'amplifie.

**Preuve que le chemin CPU est mathématiquement exact** : `allcpuA`
(`--moe-cache-mib 32 --moe-cache-static --moe-cache-cpu-fallback` → `q = n_slots / n_layers = 0`,
**aucun expert résident, les 8 experts de chaque token calculés sur le CPU**) sur prompt A à
`-n 128` : **diff de contenu = 0** vs `/tmp/base128.txt` (1121 caractères identiques, seules les
lignes `build`/`t/s`/spinner diffèrent). **Le calcul CPU des experts quantifiés reproduit
bit-à-bit la sortie du run sans cache sur prompt A.**

**Stats prompt B** : `cpufbB` hits 31936, misses 52184, hit **37,96 %**, **uploaded 0,00 MiB**,
10515 tokens servis par le cache GPU, **136 304 calculs d'experts laissés au CPU**,
`1115/10496` résidents (1016,88 MiB). `cleanB` hits 47007, misses 34617, hit **57,59 %**,
**uploaded 31 654,70 MiB**, scratch 34617, `1118/10496` résidents (1020,00 MiB), 10203 tokens
servis. Le fallback **ne lit rien** (0,00 MiB uploadés) au prix de 136 304 calculs CPU.

**Ces stats viennent des runs `-n 256`** ; les **masters** de `verify.sh` sont les runs **sans `-n`**
(jusqu'à EOS, ~1900 tokens) : `cleanB` hits **321253**, misses 286771, hit **52,84 %**, uploaded
**261800,45 MiB** ; `cpufbB` hits **153768**, misses 383792, hit **28,60 %**, uploaded **0,00 MiB** ;
`allcpuB` hits **0**, misses **494024**, hit **0,00 %**, uploaded **0,00 MiB**. Les masters sont
copiés dans **`/data/o35/o35/masters/`** (`baseB.txt`, `cleanB.txt`, `cpufbB.txt`, `allcpuB.txt` +
les masters de prompt A) parce que `runB` **écrase** `/tmp/$name.txt` à chaque run — comparer un run
à son propre fichier `/tmp` donnait **0 par construction**, une vérification vide de sens.

**La sonde est construite, fait, et MESURE** (session du 12 octobre, patch sauvegardé dans
`tools/moe_probe.patch`, 224 lignes, 3 fichiers) : pour chaque couche, un `ggml_cpy` du résultat
GPU du slot (`moe_probe_gpu_<il>`, alloué sur le CPU) et un `ggml_mul_mat_id` CPU sur les poids
hôtes d'origine avec les mêmes `ids` (`moe_probe_cpu_<il>`), comparés ligne à ligne pour les
expert **résidents** (`ids_cpu == -1`), sur les `ubatch.n_tokens` réels.

**Résultat sur prompt B, `-n 8`, mode `cpufb`, 300 comparaisons (couche × pas de décode),
1 968 128 éléments comparés** :

| mesure | valeur |
|---|---|
| max abs global | **0,020653** (couche 0) |
| mean abs pondéré sur les 1,97 M d'éléments | **0,003236** |
| moyenne des max-abs par couche | 0,003144 |
| max rel global | **7,455** (couche 0, sur une valeur proche de zéro) |
| moyenne des max-rel par couche | **1,41** |
| max gpu / max cpu observés | 3,630216 / 3,628572 |

**Conclusion — la divergence n'est PAS l'ordre d'accumulation FP32** : un ordre d'accumulation
différent donnerait un écart relatif de ~1e-6. On mesure **0,3 % de l'amplitude typique**
(valeurs 0,1 à 3,6), avec des max-rel de 1 à 7 sur les petites valeurs. C'est la **différence de
noyau** : `MMQ` CPU (`ggml_vec_dot` déquantifiant en fp32) contre `MMVQ` CUDA sur les mêmes octets
`iq2_xxs`/`iq3_xxs`/`iq4_xs`. **Aucun bug dans le chemin CPU** : le chemin CPU est exact vs la
baseline CPU (`allcpuA` = `base128`, diff 0), et la divergence GPU/CPU est la divergence
standard de llama.cpp entre un calcul CPU et le même calcul CUDA (`-ngl 0` vs `-ngl 99`).
**Acceptable** : elle est documentée, bornée (~0,3 %), et le chemin mixte `cpufb` reste le plus
proche du chemin tout-GPU (préfixe commun **1514 sur 9027** caractères, contre 795 pour `allcpuB`).

**Comment faire tourner la sonde** (`LLAMA_MOE_PROBE=1`, `/tmp/runProbe4.sh`, sortie
`/tmp/probeB4.log`) : les tenseurs de la sonde **doivent être consommés** pour entrer dans le
graphe (`res = ggml_add(res, ggml_scale(ggml_add(gpu, cpu), 0.0f))` — ajouter 0 est exact) ;
**`ggml_set_input` est interdit sur un noeud d'op** (`GGML_ASSERT(op == GGML_OP_NONE)` à
`src/llama-context.cpp:654`) → utiliser **`ggml_set_output`**, seul moyen de garder les tampons
vivants : sans lui l'allocateur CPU **réutilise** les tampons et la lecture après le calcul donne
des zéros (`max cpu = 0,000000`) ; et `ne[2]` du graphe est le `n_tokens` **réservé** (2), pas le
`n_tokens` réel → **passer `ubatch.n_tokens`** à la sonde et borner la boucle dessus.

## Premier essai sans MTP (run long, prompt B, `-n 1024`) — session du 12 octobre

**Outil nouveau `tools/runtrial.py`** : python est le parent de `llama-cli` (donc `/proc/PID/io` est
lisible), **pas de strace** (il fausse le t/s), échantillonnage toutes les 0,13 s, sortie compatible
`tools/phase6.py`. Usage : `python3 tools/runtrial.py <name> <n> <mode> [pin]`, `mode = clean | cpufb |
allcpu`. Script **`tools/trial.sh`** : éviction (`tools/evict.py 13`) puis run, pour chaque mode.

**Cold start, prompt B, `-n 1024 -c 2048 --temp 0 -s 42`, `pin 1990`**, page cache évictée avant chaque run :

| mode | t/s préfill | **t/s décode** | hit rate | uploaded | **lectures disque en décode** | RSS |
|---|---|---|---|---|---|---|
| `cpufb` | 24,20 | **16,26** | 30,75 % (101443 hits / 228437 misses) | **0,00 MiB** | **0,05 MiB** (disk 0,13) | 10,57 → 5,10 GiB |
| `clean` | 17,22 | **20,13** | 56,10 % (183663 / 143721) | 131 327,19 MiB | **114,81 MiB** (disk 115,34) | 11,40 → 0,56 GiB |

- **Exigence 5 vérifiée sur un run long** : le fallback CPU ne lit **rien** sur le disque pendant les
  62,9 s de décode (**0,05 MiB** `read_bytes`, **0,13 MiB** `diskstats`) contre **115 MiB** pour le
  chemin tout-GPU, dont les uploads d'experts relisent les pages mmap évictées.
- Coût : **16,26 t/s contre 20,13** (−19 %). Le préfill est **plus rapide** en mode fallback (24,20
  contre 17,22) — aucun upload pendant le préfill.
- Le chargement lit le fichier entier (`read_bytes` **10 576,65 MiB** en mode cpufb) et le fichier de
  11,88 GB **reste résident** après le chargement (blocs 0..10 à 1024/1024 MiB, bloc 11 à 68 MiB) :
  c'est ce qui rend le décode sans lecture disque.
- Sortie **cohérente** dans les deux modes (texte sur le moteur diesel, coupé net à 1024 tokens).
- **`--moe-stats` prend un FILE en argument** : le passer nu avale le flag suivant (`-c`) et le parseur
  échoue sur `error: invalid argument: 2048` (run de 207 octets, exit 1). Les lignes de hit rate viennent
  de `LLAMA_LOG_INFO` et sont dans le log **sans** `--moe-stats`.

## Blocages / risques connus

- **`majflt` N'EST PAS un indicateur de lectures disque sur ce noyau** : une touche froide de
  100 MiB sur le mmap du GGUF donne `minflt += 1604`, **`majflt += 0`**, `read_bytes += 100,25 MiB`
  (mesuré par `/tmp/faulttest2.py` après `evict.py 13`). Les faultes mmap file-backed non résidentes
  sont comptées comme **mineures** → le « majflt = 0 » des runs ne prouvait rien. **Seuls
  `read_bytes` (`/proc/PID/io`) et `/proc/diskstats` sont fiables.**
- **Les lectures du décode sont des pages évictées puis refaultées, pas des lectures d'expert par
  le chemin d'upload** : `strace` sur le processus montre **0 appel `read()` sur le GGUF dans la
  fenêtre de génération** (20,98 MiB de `read()` sur le GGUF, **tous au chargement**). Les ~140 MiB
  lus en début de décode (t 96,6 → 102 s) s'accompagnent d'une **chute de RSS de 11,41 → 10,29 GiB**
  (le noyau vire des pages partagées mappées) puis d'une remontée, et **les lectures s'arrêtent à
  t ≈ 102 s** : les 3,6 dernières secondes de génération ne lisent rien.
- **Exigence 5 remplie à page cache chaude** : la fenêtre de génération lit **0,00 MiB** (pin17).
  En run froid, le coût de convergence est ~140 MiB sur les 5 premières secondes.
- Le pinning complet est impossible : RLIMIT_MEMLOCK = 1,945 GiB < 10 GB d'experts.
- `max_batch_cached = 1` par groupe → cache décode-only, prefill non caché.
- `GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il)` si le
  scratch statique est épuisé, ou si top-k et lookup du slot map tombent dans le même split.
- Seules les couches dont **tous** les tensors d'experts sont en mémoire host sont cachées →
  `-cmoe` est **obligatoire** (sinon 10 GB d'experts vont au GPU → échec de chargement, exit 1).
- Blocage dispatch CPU/GPU : `ggml_mul_mat_id` a un `n_expert_used` fixe, pas de saut d'expert
  par token. MVP (A) = granularité par couche via le choix du budget `-cmoe`/`-ncmoe` piloté par
  le profil. MVP (B) = nouvel op ggml avec `kind[]` par slot (Strata `expert_source.cpp:3522-3760`).
  Rejeté (numériquement faux) : slot rempli de zéros + masquage des poids.
- Kernels MoE de prefill Strata = sm_75+ → inutilisables sur sm_61.

## Mécanismes Strata de référence (file:line)

`expert_cache.hpp:38` `kNotResident=-1` ; `cpp:633-636` `slot_of` ; `cpp:638-656` `admit()`
sans éviction ; `cpp:141-168` `rank_learned_profile` ; `cpp:170-210` `write_expert_profile`
(binaire STRP, `.tmp`+rename atomique) ; **`cpp:170` `set_per_layer_admission`** (mécanisme
porté en plages par couche) ; `cpp:567-568` alignement 256 octets des slots ;
`cpp:527-533` `cudaMemset` pour des slots stale déterministes ;
`expert_source.cpp:3562-3564` décroissance d'usage `adapt_decay` 0,7 ;
**`expert_source.cpp:2526-2586` `pin_cache_complement`** + `expert_source.hpp:104-108`
`make_cache_complement_plan` + `:183` `release()` (garantie de l'exigence 5) ;
`include/strata/core/pinned.hpp` `PinnedArena` ; `hit_hook.hpp:28,39` hook de hit en deux phases ;
`foresight_swap.hpp:37-54`.

**Rejetés** : VMM (Turing+), `mma.sync m16n8k32` (sm_75+), `cp.async` (sm_80+), clusters (sm_90+),
experts peer/remote, `expert_source.hpp:18-21` constantes de géométrie (Strata Qwen3.5-Next
48×512, pas qwen35moe 2048×512). **Reporté** : tier adaptatif (`generate.cpp:8153-8279`).

## Contraintes matérielles

GTX 1070 sm_61 8192 MiB, i7-4790 (AVX2, pas d'AVX-512), 15 GiB RAM, **HDD** (zero disk sur le
chemin de décode). Pas d'Ornith GGUF sur la machine → modèle de test
`/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (même arch `qwen35moe`).
Toolchain : `/snap/bin/cmake` 4.4.4, GNU 11.4.0, nvcc 12.8.93, **pas de ninja** (Makefile),
gdb installé, **pas de numpy**. **Unity build NON activé.** Fusion MMVQ désactivée sur Pascal →
chemin MMQ. Shell = mvdan/sh : `pid=$!` dans un fichier script fonctionne, `&` inline est abîmé ;
**les process substitution `<(...)` exigent bash, pas sh**.

## Pièges rencontrés

- `-cmoe` **obligatoire** ; sans lui le chargement échoue (exit 1).
- Exit 134 = SIGABRT ; le gestionnaire de crash de llama.cpp lance gdb et pollue stdout.
  La faute CUDA est détectée à `ggml_backend_cuda_synchronize` (`ggml-cuda.cu:2633`).
- Locale française : `fprintf("%.6f")` émet une virgule → CSV corrompu (corrigé par formatage
  entier en point fixe).
- `--moe-stats` n'écrit rien si `--moe-cache-mib` n'est pas défini.
- Deux `llama_context` sont créés (main + MTP) ; le premier écrit un profil/stats à 0 paires.
- Le contrôle du header CSV exige le `\n` final : `strcmp(line, "layer,expert,count,weight_sum\n")`.
- `grep -c` renvoie exit 1 quand le compte est 0 (bénin).
- **Champs de `/proc/PID/stat`** : après le token `)`, `f[6]` = flags (constant 4194304),
  **`f[7]` = minflt, `f[8]` = majflt**. L'ancien `tools/majflt.py` lisait `f[6],f[7]` → tous les
  « majflt » des sessions précédentes (293 565 au chargement, +85 en décode, +246 KB) étaient des
  **minflt**. Utiliser `tools/majflt2.py`.
- **`/proc/sys/kernel/yama/ptrace_scope = 1`** → `/proc/PID/io` n'est lisible que pour un enfant
  direct ; un processus frère obtient `PermissionError`. Mesurer les lectures disque système avec
  `/proc/diskstats` (champ index 5 = `sectors_read` de `sda`, × 512).
- **`f[21] = rss`** dans `/proc/PID/stat` après le token `)` (vérifié contre `VmRSS`),
  `f[19] = vsize`.
- Mesurer le **disque ambiant sans aucun processus** avant de conclure : 0,05 MiB en 10 s ici → les
  lectures de la fenêtre de génération sont réelles, pas du bruit.
- `mincore` échantillonné à 0,2 s **ne voit pas** les cycles drop+refill plus courts : une résidence
  « complète » n'exclut pas des lectures disque. `tools/runio6.py` sonde 8 fenêtres de 64 MiB en
  rotation à 20 ms (le `mincore` complet coûte **130 ms** → inutilisable à 20 ms).
- **`majflt` ne compte PAS les lectures disque d'un mmap file-backed sur ce noyau** (voir
  `/tmp/faulttest2.py`) : ne jamais conclure « pas de lecture disque » d'après majflt = 0.
- **`tools/runstrace.sh` + `tools/strace_analyze.py`** attribuent les lectures à un fichier :
  `strace -f -e trace=read,pread64,readv,preadv -yy -tt -s 32 -o f` (le `-yy` imprime le chemin du
  fd). Le format de ligne est `PID HH:MM:SS.mmmmmm read(3<chemin>, ...) = N` — **le pid est avant
  l'horodatage**. Aligner la fenêtre de génération sur l'horloge murale : première ligne strace =
  t=0, fenêtre = `eval time` du log.
- `subprocess.Popen` refuse les **entiers** dans la liste d'arguments (`"-s", 32` → TypeError) :
  passer `"32"`. Le `comm` de `/proc/PID/stat` est **entre parenthèses** →
  `st.split("(",1)[1].split(")",1)[0]`.
- `ctypes` : un `create_string_buffer` indexé renvoie `bytes` → `cast(vec, POINTER(c_ubyte))` pour les
  octets de `mincore` ; taille du fichier via `os.fstat(fd).st_size` (pas `.size`).
- **Horodatage du log = `MM.SS.mmm` (minutes, secondes, millisecondes)** : `1.59.502` = **119,502 s**
  et non 159,502. Convertir avec `int(p[0])*60 + int(p[1]) + int(p[2])/1000`. Une erreur ici place la
  fenêtre de génération **après** la fin des samples et donne une analyse vide.
- `create_string_buffer` n'a **pas** `.contents` : pour compter la résidence, utiliser `vec.raw`
  (`bytes`) puis `raw[a:b].count(1)` — 2 ms pour 11 GiB, contre 131 ms pour l'appel `mincore`.
- `tools/phase.py` détecte mal le début du décode : llama-cli écrit le spinner `Loading model...` sur
  stdout → la taille du fichier devient non nulle **avant** le premier token. Utiliser
  **`tools/phase2.py`** qui aligne la fenêtre sur `eval time = X ms` du log.
- Locale française : `eval time = 6241,37 ms` → `replace(',','')` donne 624137 (faux). Il faut
  `replace(',','.')`.
- **`tools/run_pin.sh` avait la logique STATIC inversée** (`STATIC=0` ajoutait `--moe-cache-static`) :
  pin5/pin4 étaient statiques et pin6 était LRU. Corrigé : défaut = statique, `STATIC=0` = LRU.
- Le binaire `llama-cli` ne contient pas les chaînes : elles sont dans
  `build/bin/libllama.so.0.6.0` et `libllama-common.so.0` (vérifier avec `strings` sur ces `.so`).
- **`GGML_TENSOR_BINARY_OP_LOCALS` (`ggml/include/ggml.h:320`)** : `ne0..ne3` et `nb0..nb3` sont les
  **dimensions et strides de `dst`**, `ne00..ne03`/`nb00..nb03` ceux de `src0`, `ne10..`/`nb10..` ceux
  de `src1`. Écrire `ne[0]` **ne compile pas** (`'ne' undeclared; did you mean 'ne3'?`). Dans
  `mul_mat_id`, `ne0` = `dst->ne[0]` (512 pour gate/up, 2048 pour down) et `nb1`, `nb2` sont les
  strides de `dst`.
- **Le rang de `dst` dans `mul_mat_id` est `i1*nb1 + i2*nb2` avec `i1 = id` (indice dans
  `n_expert_used`) et `i2 = iid1` (ligne de `src1`)** — `ggml-cpu.c:1534`. Tout code annexe qui touche
  `dst` doit utiliser le **même ordre** ; l'inverser écrit hors des limites du tas.
- `hparams.no_alloc` (champ de `llama_hparams`, contexte de mesure mémoire) **n'est pas** le
  `no_alloc` du graphe (champ de `llama_graph_params`) : le garde `!hparams.no_alloc` ne filtre **pas**
  les builds de `graph_reserve`. Les sondes qui lisent les données d'un tenseur du cache doivent être
  gardées par `!hparams.no_alloc` sinon `cpu_map->data` est un pointeur bidon → SIGSEGV.
- Une sonde placée **avant** la boucle de calcul mesure un `dst` **non initialisé** : la placer **après**
  le `for (cur_a)` avec `ggml_barrier(params->threadpool)` et `ith == 0`.
- `grep -o "nt=[0-9]*"` **matche aussi `resident=28`** (sous-chaîne `nt=28`) → les faux `nt=26,27,28`
  venaient des valeurs de `resident=`. Ancre : `grep -o "\bnt=[0-9]*"`.
- `phase4.py` filtre `len(f) == 7` (12 blocs de GiB → 18 champs) et `phase5.py` lit `f[6]` comme
  disque : **les deux sont faux** avec `runio7.py`. Utiliser `phase6.py`.
- `git commit --amend` **ne relit pas** `-c user.name=...` : il faut le repasser à chaque amend
  (`Identité de validateur inconnue` → exit 128).
- **`-p $P` sans guillemets casse tout** : le prompt est coupé en mots, `llama-cli` échoue sur
  `error: invalid argument: Explain` (exit 1), les fichiers de sortie sont **vides**, et **tous les
  diffs valent 0** → **fausse vérification**. Solution : tableau bash
  `base=(... -p "Q: Explain how a diesel engine works. A:" ...)` puis `"${base[@]}"`, et lancer le
  script avec `bash` (les tableaux et `<(...)` ne marchent pas sous mvdan/sh).
- **`--log-probs` n'existe pas** dans ce build (aucune chaîne dans `libllama-cli-impl.so` ni
  `libllama-common.so.0.6.0`, aucun `log_probs` dans `common/arg.cpp`) → **impossible de comparer
  les logits token par token**. La divergence se mesure **uniquement** sur les préfixes de sortie.
- **Comparer des sorties de `llama-cli`** : il faut retirer **avant** le diff la ligne
  `build      : b1155x-<hash>`, la ligne `[ Prompt: … t/s | Generation: … t/s ]`, la ligne
  `Loading model...` avec son spinner, `>` et les lignes vides. Sans ce filtrage, **chaque run
  « diffère » en 4 lignes** et le préfixe commun s'arrête à ~75 caractères. Utiliser
  `tools/cmp.py`.
- **`baseB` (sans `-cmoe` ni `-ngl`) n'est pas la baseline numérique des runs de cache** : ses
  couches denses tournent sur le CPU, celles des runs de cache sur CUDA. Comparer `cleanB` à
  `cpufbB` (même chemin dense CUDA) pour isoler l'effet du chemin CPU des experts.
- **Un tenseur de sonde qui n'est consommé par aucun op n'entre PAS dans le graphe** :
  `ggml_build_forward_expand` n'ajoute que les noeuds **atteignables depuis les sorties**. Un
  `ggml_cpy`/`ggml_mul_mat_id` créé et nommé mais jamais utilisé → le scan du graphe ne le trouve
  jamais (`gpu=0 cpu=0`). Le rendre consommé : `res = ggml_add(res, ggml_scale(ggml_add(a, b), 0.0f))`
  (ajouter 0 est exact).
- **`ggml_set_input` est interdit sur un noeud d'op** : `GGML_ASSERT(tensor->op == GGML_OP_NONE)`
  (`ggml/src/ggml.c:8005` et le contrôle de `src/llama-context.cpp:654`) → SIGABRT (exit 134).
  Pour garder un tampon vivant pendant tout le graphe, utiliser **`ggml_set_output`**.
- **Sans `ggml_set_output`, l'allocateur CPU réutilise les tampons des tenseurs de la sonde** :
  la lecture après le calcul donne des valeurs bidons (tous les `moe_probe_cpu_*` pointaient dans
  le même tampon `0x…818060`, `max cpu = 0,000000`). Le scan par nom ne suffit pas, il faut que les
  tenseurs soient alloués à la préparation et libérés à la fin.
- **`ne[2]` d'un noeud du graphe de décode est le `n_tokens` RÉSERVÉ (2), pas le `n_tokens` réel
  (1)** : le graphe réservé est réutilisé. Une sonde qui boucle sur `ne[2]` lit des lignes **stales**.
  Passer le vrai `ubatch.n_tokens` à la sonde.
- `node->src[1]` (et non `node->src1`) : dans cette version `ggml_tensor` a un tableau `src[GGML_MAX_SRC]`.

## Interdits à respecter

Pas de réécriture de llama.cpp, pas de Qwen Flash Next, pas de toucher le tokenizer, pas de nouveau
format, pas de dépendance au SSD, mmap n'est pas la solution du hot-path, pas d'hypothèse RTX
moderne, pas de sacrifice de correction numérique pour les t/s.
