# CHECKPOINT — état complet du port (sauvegarde contre l'interruption)

Branch : `experimental/ornith-expert-cache` dans `/data/o35/llama.cpp` (base `master` @ `23b0202a1`).
cwd = `/data/o35/o35`. Symlinks : `llama.cpp -> ../llama.cpp/`, `Strata -> ../Strata`.
Docs : `PROGRESS.md` (894 lignes, autorité), `PORT_PLAN.md` (380, en anglais),
`IMPLEMENTATION_NOTES.md` (317, file:line Strata), **ce fichier** (467).
Dernière mise à jour : 2026-10-12, **fallback CPU porté, mesuré et commité (`47a1dbbfc`)** +
**deuxième prompt (prompt B) testé : divergence mesurée** + **premier essai sans MTP sur un run long
(`-n 1024`) : 16,26 t/s et 0,05 MiB lu sur le disque pendant le décode** —
**exigences 1 à 6 toutes FAITES**, `verify.sh` **vert** (7 diffs A = 0 + 6 contrôles prompt B).

## Où j'en suis (une ligne)

**Exigences 1 à 6 : FAITES et vérifiées.** **Le fallback CPU par expert est porté, buildé, mesuré et
COMMITÉ** (`47a1dbbfc`, 11 fichiers, 217 insertions, 57 deletions) : `-n 8`, `-n 128`, `-n 1024` sont
**bit-identiques** aux baselines statiques, `uploaded = 0,00 MiB`, **17,22 t/s** à `-n 1024`
(19,39 t/s sans fallback), **les 7 diffs de `verify.sh` = 0**, lectures disque en décode **inchangées**
(172,42 MiB contre 174,11 MiB pour pin29). `PROGRESS.md` et `IMPLEMENTATION_NOTES.md` **sont à jour**.
**Sonde supprimée, rebuild propre à `47a1dbbfc`, prompt B ajouté à `verify.sh` comme régression —
la vérification complète passe toute verte** (7 diffs A = 0, `cleanB`/`cpufbB`/`allcpuB` = 0 contre
leur master, `cpufbB` vs `cleanB` = **121**, `cpufbB` vs `allcpuB` = **130**, `allcpuB` vs `cleanB` =
**137**). Le patch de la sonde est sauvegardé `tools/moe_probe.patch`. Le `pin` est **mesuré**
(−119 MiB, −41 %) et le batch de 2-4 tokens est **sans objet** (1 token par ubatch en décode glouton).
**Deuxième prompt testé (prompt B)** : la sortie **n'est pas bit-identique** ; la comparaison propre est
`cleanB` vs `cpufbB` = **121 lignes de diff, préfixe commun 1514 sur 9027 (16,8 %)** ; la preuve d'exactitude
du chemin CPU est **`allcpuA` == `base128` (diff de contenu = 0)** avec **tous** les experts sur le CPU.

## Todo list d'origine (les 6 exigences de `PORT_PLAN.md`)

1. **Granularité par expert** (slot map `[il][e] → slot`, un bank par tensor, `nb[2]` conservé) —
   **FAITE** : llama.cpp la possédait déjà (`src/llama-moe-cache.cpp:20-21`, `:364-365`), rien ajouté.
2. **Statistiques (layer, expert)** — **FAITE** : commit `5bfb641e4`, `src/llama-moe-profile.{h,cpp}`,
   option `--moe-profile-output`, CSV `layer,expert,count,weight_sum`, sortie **identique** avec/sans.
3. **Cache statique seedé par le profil, sans éviction, plages de slots disjointes par couche** —
   **FAITE** : commit `1a1811266`, options `--moe-profile`, `--moe-cache-static`, `--moe-stats`.
   Hit 48,77 % à 1024 MiB sur 128 tokens, output **bit-identique** à la baseline sans cache.
4. **Complément RAM : populate / pin / release** — **FAITE** : commit `de0531d8f` (12 fichiers,
   300 insertions), `llama_mmap::populate/pin/release`, `llama_model::mmap_range()`,
   `cache_complement()`, option `--moe-cache-pin-mib`.
5. **Zéro lecture disque sur le chemin de décode** — **FAITE** : à page cache chaude la fenêtre de
   génération lit **0,00 MiB** (pin17) ; à froid le residuel est **112,69 MiB** concentré sur la
   première seconde, et il vient de la **queue du fichier** (peuplement du fichier entier, commit
   `1a6b4a0b0`, 3 fichiers / 45 insertions), **pas** des plages d'experts (leur résidence ne baisse
   jamais).
6. **Experts GPU sur CUDA + experts absents calculés sur le CPU** — **FAITE et COMMITÉE**
   (`47a1dbbfc`) : option `--moe-cache-cpu-fallback`, 11 fichiers, 217 insertions / 57 deletions.
   Voir la section dédiée.

## Todo list complémentaire (les 6 tâches de cette session)

1. **Écrire `CHECKPOINT.md`** — **FAIT** (ce fichier).
2. **Débugger le fallback CPU** — **FAIT** : cause racine trouvée (voir plus bas). L'hypothèse
   `ggml_backend_dev_supports_op` est **falsifiée** ; l'anomalie des « 2 tokens » est une **piste
   fausse** (graphes réservés).
3. **Run avec `--moe-cache-cpu-fallback`** — **FAIT** : 128 tokens **et** 1024 tokens, output
   bit-identique, pas de crash, `uploaded = 0,00 MiB`.
4. **Mesurer t/s et lectures disque en décode** — **FAIT** : froid (`runio7.py cpufb 1024 1990 fb`) →
   **`read_bytes + 170,02 MiB`, `disk + 172,42 MiB`, `majflt = 0`** contre **173,30 / 174,11 MiB** pour
   pin29 → **le fallback n'ajoute aucune lecture disque** ; chaud → **17,22 t/s** contre **19,39 t/s**
   statique (`clean512`). Analyse par **`tools/phase6.py`** (nouvel outil, index corrects).
5. **Vérifier que les diffs de `verify.sh` restent à 0** — **FAIT** : **7 runs / 7 diffs = 0**
   (clean8==base8, clean128==base128, clean512==base512, lru8==base8, lru128==lru_master,
   pin128==clean128, **cpufb==clean128**).
6. **Commit et mise à jour de `PROGRESS.md`** — **FAIT** : commit `47a1dbbfc`, `PROGRESS.md` et
   `IMPLEMENTATION_NOTES.md` mis à jour, `CHECKPOINT.md` mis à jour.

## Le diff du fallback (11 fichiers, +217 / -57) — **COMMITÉ : `47a1dbbfc`**

| fichier | lignes | contenu |
|---|---|---|
| `common/arg.cpp` | +8 | option `--moe-cache-cpu-fallback` |
| `common/common.h` | +2 | `bool moe_cache_cpu_fallback = false` |
| `common/common.cpp` | +2 | câblage dans `cparams` |
| `include/llama.h` | +2 | champ dans `llama_context_params` |
| `src/llama-cparams.h` | +2 | même champ |
| `src/llama-context.cpp` | +4/-1 | copie du cparam (:303), passage au ctor (:496), défaut (:3937) |
| `src/llama-graph.h` | +10/-4 | `build_lora_mm_id(..., ggml_tensor * ids_cpu)`, `build_moe_cache_slots(..., int il, ggml_tensor ** ids_cpu)` |
| `src/llama-graph.cpp` | +33/-6 | `ids_cpu_rows = ggml_get_rows(ctx0, cpu_map, ids)` forcé sur `backend_cpu`, `res_cpu = ggml_mul_mat_id(w, cur, ids_cpu)` forcé CPU puis `ggml_add` ; câblé aux 4 sites d'appel |
| `src/llama-moe-cache.h` | +6/-1 | ctor avec `bool cpu_fallback`, `get_cpu_map(int32_t il)` |
| `src/llama-moe-cache.cpp` | +183/-57 | slots zéro, carte CPU par couche, `copy()` court-circuitée, `get_slot_map` sans limite |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +22/-1 | `if (i02 < 0) continue;` dans le groupage des lignes + **zeroing des lignes non calculées** |

## Conception du fallback CPU (schéma du « slot zéro »)

`ggml_mul_mat_id` a un `n_expert_used` **fixe** par token : il est impossible de sauter des experts
par token. Le schéma retenu :

- **Bank GPU** : `n_slots + n_extra` slots, `n_extra = cpu_fallback ? 2 : 1`
  (`llama-moe-cache.cpp:671`). Le slot `n_slots` est **le slot zéro** que le GPU lit pour les experts
  calculés sur le CPU ; `n_slots+1` est le padding que CUDA lit au-delà de la fin.
- **Le slot zéro est vraiment zéro** : `ggml_backend_buffer_clear(d.buf.get(), 0)`
  (`llama-moe-cache.cpp:724`). **Prouvé** : des octets tous à zéro se déquantifient en poids tous à
  zéro pour **tous** les types du modèle (`iq2_xxs`, `iq3_xxs`, `iq4_xs`, `iq2_s`, `q2_K`) — test
  compilé `/tmp/zeroq.c` (recette de compilation plus bas). Donc la contribution GPU d'un expert
  non résident est **exactement 0**, et le CPU fournit la valeur réelle.
- **Carte CPU par couche** : `l.cpu_map` (I32 `[1, n_expert]`, `llama-moe-cache.cpp:691-695`),
  `cpu[e] = -1` si l'expert est résident, `= e` sinon. Le slot GPU d'un expert non résident est mis à
  `g.n_slots` (`:779-792`). Buffer host cleared à `0xff` (`:733`).
- **Mode statique + fallback** : **pas de LRU, pas de scratch** — tous les slots sont résidents pour
  tout le run, `q = g.n_slots / n_layers` (27 par couche dans le groupe 0),
  `g.max_batch_cached = max_batch_mmvq = 4` (constante `:325`, bloc `:760-795`).
- **`copy()` court-circuitée** (`llama-moe-cache.cpp:992-1005`) : compte hits/misses, **n'upload
  rien**, copie la carte des slots. `uploaded = 0,00 MiB` est structurel.
- **`get_slot_map`** (`:895-905`) : la limite `n_tokens*n_expert_used > hi-lo` est sautée quand
  `g.cpu_fallback`.
- **Graph** : `build_moe_cache_slots` (`llama-graph.cpp:2484-2534`) construit
  `ids_cpu_rows = ggml_get_rows(ctx0, cpu_map, ids)` forcé sur `backend_cpu` (`:2522`), reshapé en
  `[8, n_tokens]` ; sortie anticipée (ids_cpu reste nullptr) si `slot_map == nullptr` (`:2494`) →
  **le prefill n'est pas touché**.
- **`build_lora_mm_id`** (`llama-graph.cpp:1615-1640`) : si `ids_cpu != nullptr`, ajoute
  `res_cpu = ggml_mul_mat_id(ctx0, w, cur, ids_cpu)` forcé sur le CPU (`:1630`), puis
  `res = ggml_add(ctx0, res, res_cpu)`. Câblé aux 4 sites (`:2288, :2307, :2320, :2421`).
- **CPU** : `ggml_compute_forward_mul_mat_id` saute `i02 < 0` dans le groupage (`ggml-cpu.c:1655`)
  et **zero** les lignes correspondantes (`:1667-1681`) pour que `ggml_add` ne leur ajoute rien.

**Rejeté** : masquer les poids à zéro et laisser le GPU calculer (numériquement faux, coût identique) ;
ajouter un garde `-1` dans `mm_ids_helper` de CUDA (`ggml/src/ggml-cuda/mmid.cu:33-108`) — le schéma
du slot zéro évite `-1` côté GPU, donc aucun kernel CUDA n'est modifié.

## Le bug du fallback — cause racine TROUVÉE

**Symptôme** : `--moe-cache-cpu-fallback` produisait **1 token**, `stop_type = "eos"`, contenu vide,
contre 128 tokens à 19,2 t/s en baseline.

**Cause racine** : dans le bloc de zeroing (`ggml-cpu.c:1667-1681`), la ligne de `dst` était indexée
`dst->data + iid1*nb1 + id*nb2`. Le compute (`ggml_compute_forward_mul_mat_id_one_chunk`,
`ggml-cpu.c:1534`) écrit `dst->data + i1*nb1 + i2*nb2` avec `i1 = id` (l'index dans `n_expert_used`)
et `i2 = iid1` (la ligne de `src1`). **Les deux strides sont inversés.**
Avec `dst = [512, 8, 2]` : `nb1 = 2048` octets (une ligne de 512 floats), `nb2 = 16384` octets.
Pour `iid1 = 1, id = 7` l'offset écrit est `1*2048 + 7*16384 = 116 320` octets, alors que `dst` fait
`512*8*2*4 = 32 768` octets → **écriture hors des limites du buffer** → corruption du tas, logits
faux, EOS immédiat.
**Correctif** : `dst->data + id*nb1 + iid1*nb2` (et la même correction dans la sonde).
**Longueur de ligne** : `ne0` dans `GGML_TENSOR_BINARY_OP_LOCALS` est **`dst->ne[0]`** (512), **pas**
`src0->ne[0]` (2048) — les strides de `src0` sont nommés `nb00..nb03` / `ne00..ne03`. Utiliser `ne[0]`
ne **compile pas** (`ne` n'est pas un tableau).

**Hypothèse falsifiée** (la piste principale de la session précédente) : `ggml_backend_dev_supports_op`
refuserait un `mul_mat_id` CPU dont `src0` est dans un buffer `CUDA_Host`. **FAUX** :
- `CUDA_Host` n'est **pas** dans `ggml_backend_cpu_get_extra_buffer_types()`
  (`ggml-cpu/ggml-cpu.cpp:42` : AMX, repack, spacemit seulement) →
  `ggml_backend_cpu_is_extra_buffer_type(CUDA_Host) = false` (`:88`) ;
- `ggml_backend_cpu_device_supports_op` (`:424`) n'a **aucun cas `GGML_OP_MUL_MAT_ID`** →
  `default: return true` ;
- `CUDA_Host` a `is_host = ggml_backend_cpu_buffer_type()->iface.is_host`
  (`ggml-cuda/ggml-cuda.cu:1317`) → `ggml_backend_buft_is_host(CUDA_Host) = true` →
  `ggml_backend_cpu_device_supports_buft` (`:490`) = true → **aucune copie de `w`** ;
- `ggml_backend_sched_set_tensor_backend` (`ggml-backend.cpp:2183`) pose
  `tensor_backend_id(node)`, la passe 5 (`:1440-1530`) l'honore ;
- **confirmé empiriquement** : la sonde tire depuis `ggml-cpu.c`, l'op CPU s'exécute bien sur le CPU.

**Piste fausse des « 2 tokens »** : la sonde affichait `dst=512,8,2` alors que
`build_moe_cache_slots` affichait `nt=1`. Explication : `graph_reserve` réserve des graphes pour
`n_tokens = 1, 2, 4, 8, …` et le CPU y est calculé. Le garde `!hparams.no_alloc` **ne filtre pas**
les graphes réservés : `no_alloc` est un champ de `llama_graph_params`, `hparams.no_alloc` est un champ
de `llama_hparams` (contexte de mesure mémoire) — les deux sont **indépendants**.
`graph_reserve` ne réserve que 1 et 512, mais les graphes construits à 2 et 4 tokens sont réels.
**Aucun 2-token anormal** : le schéma est correct à 1, 2 et 4 tokens.

**Autre piège de lecture** : `grep -o "nt=[0-9]*"` sur les sondes compte aussi `resident=28`
(la sous-chaîne `nt=28`) → les valeurs `nt=26,27,28` étaient **des `resident=`**. Le parsing, pas les
données.

## Résultats mesurés du fallback (build sans sondes)

```
impl: CPU fallback: 1035 resident experts, 8437 experts left to the CPU, quota 27 per layer, up to 4 tokens per batch
impl: CPU fallback: 54 resident experts, 458 experts left to the CPU, quota 27 per layer
impl: CPU fallback: 26 resident experts, 230 experts left to the CPU, quota 26 per layer
impl: uploaded 929,88 MiB of seeded experts
cache_complement: MoE complement: 8329,12 MiB faulted into RAM, 1986,48 MiB of the rest faulted,
                  0,00 MiB pinned in 0 experts, 1016,88 MiB of resident pages dropped
llama_moe_cache: ubatch <= 8: hits = 12953, misses = 30207, hit rate = 30,01%, uploaded = 0,00 MiB, scratch = 0
llama_moe_cache: 40/41 layers cached, 1115/10496 experts resident (1016,88 MiB),
                 5395 tokens served by the GPU cache, 73367 expert computations left to the CPU
prompt eval time = 790,72 ms / 21 tokens (26,56 t/s)
eval time = 7142,85 ms / 128 tokens (56,24 ms par token, 17,78 t/s)
```

- `-n 8` : **diff = 0** contre `/tmp/clean8.txt` ; `-n 128` : **diff = 0** contre
  `/tmp/clean128.txt` (filtrés comme `verify.sh`).
- `uploaded = 0,00 MiB` → **l'exigence « zéro upload en décode » est structurellement remplie**.
- Hit rate **30,01 %** contre 48,77 % en statique : en mode fallback il n'y a **ni LRU ni scratch**,
  seules les 27 places seedées par couche comptent comme hits. Le calcul CPU est le prix payé.
- 17,78 t/s à `-n 128` contre 19,18 t/s (`clean128`) ; **17,22 t/s à `-n 1024`** (176 tokens, EOS) contre
  **19,39 t/s** (`clean512`) : le CPU (i7-4790 AVX2) absorbe **99 341** calculs d'experts sur 157 861
  (63 % du total), contre 73 367 sur 116 527 à `-n 128`. Coût : **11 %** de t/s.
- **Froid** (`runio7.py cpufb 1024 1990 fb`) : fenêtre de décode 138,76 → 151,61 s,
  **`read_bytes + 170,02 MiB`, `disk + 172,42 MiB`, `majflt = 0`** ; baseline pin29 : **173,30 / 174,11 MiB**
  → **aucune lecture ajoutée**. Résidence : seul le **bloc de GiB 10** bouge (`906 → 1024`, +118 MiB) =
  la **queue du fichier**, pas les plages d'experts. Lectures **concentrées sur les 2 premières secondes**
  (27 MiB par sample jusqu'à t = 140,66), puis **0,00 MiB** de t = 143,24 à la fin. `strace` dans la
  fenêtre : **0,15 MiB** de `read()` seulement. t/s froid = **13,62**.
- **Le `pin` sert en mode fallback** : le même run froid **sans** `--moe-cache-pin-mib`
  (`runio7.py cpufb0 1024 0 fb`) lit **290,99 MiB (`read_bytes`) / 291,66 MiB (disk)** dans la fenêtre de
  décode (97,36 → 110,42 s) contre **170,02 / 172,42 MiB** avec `pin 1990` → **−119 MiB (−41 %)** ;
  t/s froid 13,40 contre 13,62 (sans effet).

**Deuxième prompt (prompt B), `-n 256 -s 42 --temp 0`** (`/tmp/runB.sh`, `/tmp/runAllCPU.sh`,
comparaisons via `tools/cmp.py`) :

| run | flags | t/s |
|---|---|---|
| `baseB` | **sans `-cmoe`, sans `-ngl`** (tout sur le CPU) | **23,76** |
| `cleanB` | `-cmoe -ngl 99 --moe-cache-mib 1024 --moe-cache-static` | **21,32** |
| `cpufbB` | idem + `--moe-cache-cpu-fallback` | **18,64** |
| `allcpuB` | `--moe-cache-mib 32` + static + fallback (`q = 0` → **tous les experts sur le CPU**) | **15,1** |

| paire | lignes de diff (contenu filtré) | préfixe commun (car.) | longueurs filtrées |
|---|---|---|---|
| `cleanB` vs `baseB` | **84** | 699 | 9027 vs 1758 |
| `cpufbB` vs `baseB` | **75** | 699 | 8136 vs 1758 |
| `allcpuB` vs `baseB` | **81** | 699 | 7923 vs 1758 |
| `cleanB` vs `cpufbB` | **121** | **1514** sur 9027 (16,8 %) | 9027 vs 8136 |
| `cleanB` vs `allcpuB` | **137** | 795 (8,8 %) | 9027 vs 7923 |
| `cpufbB` vs `allcpuB` | **130** | 795 | 8136 vs 7923 |

**Longueurs** : `baseB` est lancé avec `-n 256` (1758 caractères filtrés), les runs de cache **sans
`-n`** → génération jusqu'à EOS (9027, 8136, 7923 caractères). Les diffs contre `baseB` sont dominés
par la longueur, seul le préfixe 699 est comparable. Les chiffres **21, 19, 33, 8, 28, 28** de la session
précédente étaient des diffs **en texte brut** pollués par `build      :`, le spinner et la ligne de `t/s`
— **faux** ; les valeurs ci-dessus sont mesurées par `tools/cmp.py` sur texte filtré.

- **`baseB` n'est pas la baseline numérique des runs de cache** : ses couches denses tournent sur le CPU,
  celles des runs de cache sur CUDA. La comparaison propre est `cleanB` vs `cpufbB` (même chemin dense
  CUDA, mêmes ids) → **121 lignes de diff, préfixe commun 1514 sur 9027 = 16,8 % de la génération
  identique** (le « 84 % » précédent divisait 1514 par la longueur de `baseB`, 1753 caractères).
- **Preuve d'exactitude du chemin CPU** : `allcpuA` (prompt A, `-n 128`, **tous** les experts sur le CPU)
  a un **diff de contenu = 0** vs `/tmp/base128.txt` (1121 caractères identiques).
- Point de divergence sur B : au caractère 699, `baseB` écrit `**Understand User Input:**` et **les trois
  modes de cache écrivent** `**Analyze User Input:**` → quasi-égal de logits en greedy.
- `cpufbB` : hits 31936, misses 52184, hit **37,96 %**, **uploaded 0,00 MiB**, 10515 tokens servis,
  **136 304 calculs d'experts sur le CPU**, `1115/10496` résidents (1016,88 MiB). `cleanB` : hit
  **57,59 %**, **uploaded 31 654,70 MiB**, scratch 34617, `1118/10496` résidents.
- Fichiers de référence : **`/data/o35/o35/masters/`** (`baseB.txt`, `cleanB.txt`, `cpufbB.txt`,
  `allcpuB.txt`, plus les masters de prompt A `base8/base128/base512/lru_master/clean128`) — copiés
  depuis `/tmp` parce que `verify.sh` **écrase** `/tmp/$name.txt` à chaque run. `/tmp/allcpuA.txt` reste
  la référence du run tout-CPU sur prompt A. Les logs `allcpu*.log` sont **vides** (runs sans `--moe-stats`).
- **Deux configurations de runs prompt B** : les stats ci-dessous (hits 31936, 10515 tokens servis,
  136 304 calculs CPU) viennent des runs **`-n 256`** ; les **masters** de `verify.sh` sont les runs
  **sans `-n`** (jusqu'à EOS, ~1900 tokens), dont les stats sont `cleanB` hits **321253**, misses 286771,
  hit **52,84 %**, uploaded **261800,45 MiB** ; `cpufbB` hits **153768**, misses 383792, hit **28,60 %**,
  uploaded **0,00 MiB** ; `allcpuB` hits **0**, misses **494024**, hit **0,00 %**, uploaded **0,00 MiB**.

**Sonde CPU vs GPU construite et fait** (patch `tools/moe_probe.patch`, `LLAMA_MOE_PROBE=1`,
`/tmp/runProbe4.sh`, sortie `/tmp/probeB4.log`) — prompt B `-n 8` mode cpufb, **300 comparaisons
(couche × pas de décode), 1 968 128 éléments** :

| mesure | valeur |
|---|---|
| max abs global | **0,020653** (couche 0) |
| mean abs pondéré | **0,003236** |
| moyenne des max-abs par couche | 0,003144 |
| max rel global | **7,455** (couche 0, valeur proche de zéro) |
| moyenne des max-rel par couche | **1,41** |
| max gpu / max cpu | 3,630216 / 3,628572 |

**Verdict : la divergence `cleanB`/`cpufbB` n'est PAS l'ordre d'accumulation FP** (1e-6 attendu) :
**0,3 % de l'amplitude typique** = **différence de noyau** (`MMQ` CPU déquantifiant en fp32 contre
`MMVQ` CUDA sur les mêmes octets `iq2_xxs`/`iq3_xxs`/`iq4_xs`). Le chemin CPU est **exact** contre la
baseline CPU (`allcpuA` = `base128`, diff 0) ; la divergence CPU-vs-GPU est le comportement standard de
llama.cpp (`-ngl 0` vs `-ngl 99`). **Acceptable** : bornée, documentée, et `cpufb` reste le plus proche
du chemin tout-GPU.

## Premier essai sans MTP (run long, prompt B, `-n 1024`)

Outil nouveau **`tools/runtrial.py`** (python parent de `llama-cli` → `/proc/PID/io` lisible, **pas de
strace**, sortie compatible `tools/phase6.py`) ; script `tools/trial.sh` = éviction + run par mode.
Cold start, `-n 1024 -c 2048 --temp 0 -s 42`, `pin 1990`, `tools/evict.py 13` avant chaque run :

| mode | t/s préfill | **t/s décode** | hit rate | uploaded | **lectures disque en décode** | RSS |
|---|---|---|---|---|---|---|
| `cpufb` | 24,20 | **16,26** | 30,75 % (101443/228437) | **0,00 MiB** | **0,05 MiB** (disk 0,13) | 10,57 → 5,10 GiB |
| `clean` | 17,22 | **20,13** | 56,10 % (183663/143721) | 131 327,19 MiB | **114,81 MiB** (disk 115,34) | 11,40 → 0,56 GiB |

- **Exigence 5 vérifiée sur un run long** : le fallback CPU **ne lit rien** pendant les 62,9 s de décode
  contre **115 MiB** pour le chemin tout-GPU (les uploads d'experts relisent les pages mmap évictées).
- Coût **−19 %** de t/s (16,26 contre 20,13) ; le **préfill est plus rapide** en fallback (24,20 contre
  17,22 — aucun upload pendant le préfill).
- Le chargement lit le fichier entier (10 576,65 MiB) et le fichier de 11,88 GB **reste résident**
  (blocs 0..10 à 1024/1024 MiB, bloc 11 à 68 MiB) → décode sans lecture disque.
- Sortie **cohérente** dans les deux modes.

## Commandes

```
# build (Makefile, arch 61, pas de ninja, pas de ccache) — incrémental rapide, complet = 453 étapes
cd /data/o35/llama.cpp && /snap/bin/cmake --build build --config Release -j8 2>&1 | grep -Ei "error|warning" | head -20; echo "BUILD_DONE"

# vérification complète : 7 runs prompt A + 3 runs prompt B — bash, PAS sh ; enlever les marqueurs avant
# masters = /data/o35/o35/masters/ (ne pas les laisser dans /tmp, verify.sh écrase /tmp/$name.txt)
rm -f /tmp/*.done && bash /data/o35/o35/tools/verify.sh

# premier essai (run long, froid) : éviction puis run, mode = clean | cpufb | allcpu
python3 tools/evict.py 13 && python3 tools/runtrial.py trial 1024 cpufb 1990 > /tmp/trial.samples 2>&1
python3 tools/phase6.py /tmp/trial.samples /tmp/trial.log

# baseline statique (-v OBLIGATOIRE pour LLAMA_LOG_INFO ; --no-display-prompt, pas --no-display)
cd /data/o35/llama.cpp && timeout 1200 ./build/bin/llama-cli -v \
  -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 --moe-cache-mib 1024 \
  --moe-profile /tmp/prof2.csv --moe-cache-static -c 2048 -n 128 --temp 0 -s 42 \
  -p "Q: What is the capital of France? A:" --no-display-prompt > /tmp/out.txt 2>/tmp/out.log

# fallback CPU
cd /data/o35/llama.cpp && timeout 1200 ./build/bin/llama-cli -v \
  -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 --moe-cache-mib 1024 \
  --moe-profile /tmp/prof2.csv --moe-cache-static --moe-cache-cpu-fallback -c 2048 -n 128 \
  --temp 0 -s 42 -p "Q: What is the capital of France? A:" --no-display-prompt \
  > /tmp/cpufb128.txt 2>/tmp/cpufb128.log

# comparaison numérique
diff <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/clean128.txt) \
     <(grep -v "t/s\|Loading model\|Exiting\|^>$\|^$" /tmp/cpufb128.txt)

# fautes / corruption
grep -Ei "ABORT|CUDA error|stale|clobber|mismatch|hit rate" /tmp/cpufb128.log

# gdb batch pour les crashs (ptrace_scope=1 autorise l'enfant direct)
cat > /tmp/gdgrun.sh <<'EOF'
#!/bin/bash
cd /data/o35/llama.cpp
gdb -batch -q -ex "run -v -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 --moe-cache-mib 1024 --moe-profile /tmp/prof2.csv --moe-cache-static --moe-cache-cpu-fallback -c 2048 -n 8 --temp 0 -s 42 -p Q: --no-display-prompt" \
  -ex "bt 15" -ex "info registers rip" ./build/bin/llama-cli > /tmp/gdb.out 2>&1
EOF
bash /tmp/gdgrun.sh; tail -30 /tmp/gdb.out

# page cache froid + lectures disque (python doit être le parent pour /proc/PID/io)
# 4ᵉ argument "fb" de runio7.py ajoute --moe-cache-cpu-fallback
python3 /data/o35/o35/tools/evict.py 13 && python3 /data/o35/o35/tools/runio7.py cpufb 1024 1990 fb > /tmp/cpufb.samples 2>&1
python3 /data/o35/o35/tools/phase6.py /tmp/cpufb.samples /tmp/cpufb.log
python3 /data/o35/o35/tools/resident.py

# attribution des read() par fichier
python3 /data/o35/o35/tools/evict.py 13 && sh /data/o35/o35/tools/runstrace.sh pin23 1024 1990
python3 /data/o35/o35/tools/strace_analyze.py /tmp/pin23.strace /tmp/pin23.log

# analyse du profil de routage
cd /data/o35/o35 && python3 tools/analyze_moe_profile.py /tmp/prof2.csv --expert-used 8 --sizes 10,20,30,40,50

# recette de compilation du test zéro-déquantification (valide)
gcc -DGGML_COMMON_DECL_C -I /data/o35/llama.cpp/ggml/src -o /tmp/zeroq /tmp/zeroq.c \
    /data/o35/llama.cpp/build/bin/libggml-base.so.0.26.0 -lm
LD_LIBRARY_PATH=/data/o35/llama.cpp/build/bin /tmp/zeroq
```

Références texte (prompt `Q: What is the capital of France? A:`, `-s 42 --temp 0`) :
`/tmp/base8.txt`, `/tmp/base128.txt`, `/tmp/base512.txt` (sans cache), `/tmp/clean8.txt`,
`/tmp/clean128.txt`, `/tmp/clean512.txt` (statique, **bit-identiques** aux baselines),
`/tmp/lru_master.txt` (référence du chemin LRU), `/tmp/cpufb128.txt` (fallback, == clean128),
`/tmp/prof2.csv` (1026 tokens, 82 % des paires vues).

## Géométrie (ground truth)

`/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`, arch `qwen35moe` : 256 experts, 8 utilisés,
41 blocs (`blk.0..39` + `blk.40` MTP), `n_embd` 2048, `n_ff_exp` 512, fichier 11,88 GB, experts
10,06 GB (95,6 %), **0,913 MiB par expert**.
**Règle : expert `e` du tensor `w` = `(uint8_t*)w->data + e*w->nb[2]`, taille `w->nb[2]`.**
Types des tensors d'experts : `iq2_xxs` (78, gate/up), `iq3_xxs` (37, down), `iq2_s` (2),
`iq4_xs` (3) — tous chargés en buffers **CUDA_Host**.
Groupes à 1024 MiB : **g0 = 37 couches `[0..33,35,36,37]` iq3_xxs, 1035 slots** ;
**g1 = 2 couches `[34,38]` iq4_xs, 54 slots** ; **g2 = 1 couche `[39]` iq4_xs, 26 slots** ;
MTP `blk.40` non caché. `impl: CUDA0 MoE cache size = 1023,12 MiB for 9346,00 MiB of host experts`.

## Pièges (tous mesurés)

**Nouveaux (cette session)**
- **`--moe-stats` prend un FILE en argument** : le passer nu avale le flag suivant (`-c`) et le parseur
  échoue sur `error: invalid argument: 2048` (run de 207 octets, exit 1). Les lignes de hit rate viennent
  de `LLAMA_LOG_INFO` et sont dans le log **sans** `--moe-stats`.
- **Strides de `dst` dans `mul_mat_id`** : le compute écrit `id*nb1 + iid1*nb2` (`ggml-cpu.c:1534`).
  Tout code annexe qui indexe `dst` doit utiliser **le même ordre**. Inverser → écriture hors buffer.
- `ne0` dans `GGML_TENSOR_BINARY_OP_LOCALS` = **`dst->ne[0]`**. Les strides/formes de `src0` sont
  `ne00..ne03` / `nb00..nb03`. `ne[0]` **n'existe pas** comme tableau → ne compile pas.
- Une sonde qui lit les données d'un tensor host du cache doit être gardée par `!hparams.no_alloc`
  **et** savoir que ce drapeau **ne filtre pas** `graph_reserve` (le `no_alloc` du graphe est un autre
  champ) : le contexte de mesure mémoire (`common_get_device_memory_data_impl` → `sched_reserve`)
  alloue des buffers de taille 0 → `cpu_map->data` est un pointeur bidon → SIGSEGV (exit 139).
  `ggml_get_data()` asserte ; `cpu_map->data` segfaulte.
- Une sonde placée **avant** la boucle de calcul mesure un `dst` non initialisé (ordures). La placer
  **après** la boucle `for (cur_a)` avec `ggml_barrier(params->threadpool)` et `ith == 0`.
- `grep -o "nt=[0-9]*"` compte aussi `resident=28` (sous-chaîne). `sed 's/.*nt=\([0-9]*\).*/\1/'` est
  glouton et matche `resident=28` parce que « resident » finit par `nt`.
- `hparams.n_expert` est `uint32_t` → cast en `int` dans les boucles (`-Wsign-compare`).
- **Un tenseur de sonde non consommé n'entre pas dans le graphe** (`ggml_build_forward_expand` n'ajoute
  que les noeuds atteignables depuis les sorties) → le scan par nom ne le trouve jamais. Le rendre
  consommé : `res = ggml_add(res, ggml_scale(ggml_add(a, b), 0.0f))` (ajouter 0 est exact).
- **`ggml_set_input` est interdit sur un noeud d'op** : `GGML_ASSERT(tensor->op == GGML_OP_NONE)`
  (`ggml.c:8005`, et le contrôle `src/llama-context.cpp:654`) → SIGABRT (exit 134). Pour garder un
  tampon vivant pendant tout le graphe : **`ggml_set_output`**.
- **Sans `ggml_set_output` l'allocateur CPU réutilise les tampons** : après le calcul, tous les
  `moe_probe_cpu_*` pointaient dans le même tampon et la lecture donnait `max cpu = 0,000000`.
- **`ne[2]` du graphe de décode = le `n_tokens` RÉSERVÉ (2), pas le réel (1)** (graphe réservé réutilisé) :
  borner la boucle sur le vrai `ubatch.n_tokens`, sinon on compare des lignes stales.
- `node->src[1]` et non `node->src1` : `ggml_tensor` a un tableau `src[GGML_MAX_SRC]`.

**Hérités (de `PROGRESS.md`)**
- `-cmoe` **obligatoire** ; sans lui le chargement échoue (exit 1). `--moe-cache-mib` est obligatoire.
  `--moe-stats` n'écrit rien sans `--moe-cache-mib`.
- Exit 134 = SIGABRT, 139 = SIGSEGV ; le gestionnaire de crash de llama.cpp lance gdb et pollue
  stdout. La faute CUDA est détectée à `ggml_backend_cuda_synchronize` (`ggml-cuda.cu:2633`).
- `init_chains()` doit recevoir `g.layers`, **pas** la liste des couches du modèle
  (`llama-moe-cache.cpp:604`) → sinon les chaînes par couche sont détruites → CUDA illegal access.
- `GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il)` si le scratch
  statique est épuisé, ou si top-k et lookup du slot map tombent dans le même split.
- `ggml_backend.cpp:1835-1838` : le callback de copie n'est appelé **que** pour les poids host ;
  `ggml_backend_sched_compute_splits` copie **toutes** les entrées d'un split avant de le calculer.
- `unmap_fragment` est **DANGEREUX** (le prefill lit les tensors d'experts directement dans le mmap →
  SIGSEGV) ; `MADV_DONTNEED` garde le mapping valide.
- **`majflt` N'EST PAS un indicateur de lecture disque** sur ce noyau : 100 MiB froids du mmap donnent
  `minflt += 1604`, `majflt += 0`, `read_bytes += 100,25 MiB`. Seuls `/proc/PID/io` (`read_bytes`) et
  `/proc/diskstats` (champ index 5 de `sda`, × 512) sont fiables.
- `/proc/PID/stat` après le token `)` : `f[7] = minflt`, `f[8] = majflt`, `f[19] = vsize`,
  `f[21] = rss`.
- `strace` montre **0 `read()` sur le GGUF pendant la génération**.
- Horodatage du log = `MM.SS.mmm` → `1.59.502` = **119,502 s** (`p[0]*60 + p[1] + p[2]/1000`).
- `mincore` d'une passe complète coûte 130 ms ; compter via `create_string_buffer.raw`
  (`raw[a:b].count(1)` = 2 ms pour 11 GiB) avec `cast(vec, POINTER(c_ubyte))` ; taille du fichier via
  `os.fstat(fd).st_size`.
- `subprocess.Popen` refuse les entiers dans argv (`"-s", 32` → TypeError).
- `ptrace_scope = 1` → `/proc/PID/io` lisible seulement pour un enfant direct.
- `RLIMIT_MEMLOCK` = 1,945 GiB ; noyau 6.8 (≥ 5.11) ne le compte plus (cgroup) →
  `--moe-cache-pin-mib 1990` passe.
- Locale française : `fprintf("%.6f")` émet une virgule ; `eval time = 6241,37 ms` →
  `replace(',','.')`.
- Deux `llama_context` sont créés (main + MTP) ; le premier écrit un profil/stats à 0 paires.
- Les chaînes vivent dans `build/bin/libllama.so.0.6.0` et `libllama-common.so.0`, pas dans le
  binaire `llama-cli`.
- `grep -c` renvoie exit 1 quand le compte est 0 (bénin).
- Layout GGUF : magic(4) + **version uint32(4)** + n_tensors uint64(8) + n_kv uint64(8) ; clé KV =
  **uint64 longueur + octets, PAS d'octet de type** (le type suit la clé) ; section KV = 10,94 MB →
  infos de tensors à l'offset 10943825.
- Shell = **mvdan/sh** : `&` inline abîmé, `<(...)` exige bash, `pid=$!` marche dans un fichier
  script. Pas de ninja, pas de ccache (`-D GGML_CCACHE=OFF` coupe le warning), pas de numpy, gdb
  présent, unity build NON activé, CUDA graphs désactivés sur sm_61, `USE_GRAPHS = 1`.
- Git sans identité → `git -c user.name="yves" -c user.email="yves@caspar.local" commit`.
- **`-p $P` sans guillemets** coupe le prompt en mots → `error: invalid argument: Explain`, exit 1,
  sorties **vides**, **tous les diffs = 0** (fausse vérification). Utiliser un tableau bash
  `base=(... -p "Q: Explain how a diesel engine works. A:" ...)` et `"${base[@]}"`, lancé avec `bash`.
- **`--log-probs` n'existe pas** dans ce build (aucune chaîne dans `libllama-cli-impl.so` /
  `libllama-common.so.0.6.0`, pas de `log_probs` dans `common/arg.cpp`) → **pas de comparaison de
  logits token par token**, uniquement des préfixes de sortie.
- **Comparer des sorties** : retirer **avant** le diff `build      : b1155x-<hash>`, la ligne
  `[ Prompt: … t/s | Generation: … t/s ]`, `Loading model...` + spinner, `>`, lignes vides. Sans ça,
  **chaque run « diffère » en 4 lignes** et le préfixe commun s'arrête à ~75 caractères. Utiliser
  **`tools/cmp.py`**.
- **`baseB` (sans `-cmoe`, sans `-ngl`) n'est pas la baseline numérique des runs de cache** : ses
  couches denses sont sur le CPU, celles des runs de cache sur CUDA.

## Interdits à respecter

Pas de réécriture de llama.cpp, pas de Qwen Flash Next, pas de toucher le tokenizer, pas de nouveau
format, pas de dépendance au SSD, mmap n'est pas la solution du hot-path, pas d'hypothèse RTX moderne,
pas de sacrifice de correction numérique pour les t/s.

## Prochaine étape exacte

1. ~~Laisser `verify.sh` finir~~ **FAIT** : **les 7 diffs = 0** (dont `cpufb == clean128`).
2. ~~Mesurer les lectures disque du fallback~~ **FAIT** : 172,42 MiB contre 174,11 MiB (pin29),
   `majflt = 0`, t/s chaud 17,22 contre 19,39. Analyse par **`tools/phase6.py`** (nouvel outil).
3. ~~Commit~~ **FAIT** : `47a1dbbfc` (11 fichiers / 217 insertions / 57 deletions), arbre propre.
   `git commit --amend` **exige de repasser** `-c user.name=... -c user.email=...`.
4. ~~Mettre à jour `PROGRESS.md` et `IMPLEMENTATION_NOTES.md`~~ **FAIT** (section « Fallback CPU porté »
   dans `PROGRESS.md`, slot zéro sans analogue Strata dans `IMPLEMENTATION_NOTES.md`).
5. ~~Batch de 2-4 tokens~~ **SANS OBJET, vérifié** : le décode glouton traite **1 token par ubatch**
   (les `n_tokens = 1, 2, 4, 8, …` du log sont les réservations de graphes `graph_reserve`), donc la
   limite `max_batch_cached = 4` ne s'applique jamais. **La priorité est d'activer MTP**
   (`model has unused tensor blk.40.nextn.eh_proj.weight ... ignoring`), qui créerait des batches
   multi-tokens.
6. ~~Réduire le coût des page faults du début de décode~~ **MESURÉ** : le `pin` **sert** en mode
   fallback — le run froid **sans** `--moe-cache-pin-mib` (`runio7.py cpufb0 1024 0 fb`) lit
   **290,99 MiB (`read_bytes`) / 291,66 MiB (disk)** dans la fenêtre de décode contre
   **170,02 / 172,42 MiB** avec `pin 1990` → **−119 MiB (−41 %)** ; t/s froid 13,40 contre 13,62
   (le `pin` ne change pas le t/s, il protège 1,945 GiB du complément contre les évictions).
7. **Divergence du prompt B mesurée — pas un bug** : la sonde est construite et fait (patch
   `tools/moe_probe.patch`, `LLAMA_MOE_PROBE=1`, `/tmp/runProbe4.sh`). Sur prompt B `-n 8` mode cpufb,
   **300 comparaisons (couche × pas), 1 968 128 éléments** : **max abs global 0,020653** (couche 0),
   **mean abs pondéré 0,003236**, **max rel global 7,455**, **moyenne des max-rel par couche 1,41**,
   max gpu 3,630216 / max cpu 3,628572. **Ce n'est pas l'ordre d'accumulation FP** (1e-6 attendu) :
   **0,3 % de l'amplitude typique** = **différence de noyau MMQ CPU vs MMVQ CUDA**. Le chemin CPU est
   exact vs la baseline CPU (`allcpuA` = `base128`, diff 0). **FAIT** : sonde supprimée
   (`git checkout -- src/llama-graph.h src/llama-graph.cpp src/llama-context.cpp`), rebuild propre à
   `47a1dbbfc`, `verify.sh` **vert** (7 diffs A = 0 + les 6 contrôles prompt B = 0/0/121/130/0/137).
   Le patch est gardé dans `tools/moe_probe.patch` (`git apply` pour le rejouer).
   Pièges : les tenseurs de sonde doivent être **consommés** pour entrer dans
   le graphe ; `ggml_set_input` est **interdit sur un noeud d'op** (`llama-context.cpp:654`) → utiliser
   `ggml_set_output` ; borner la boucle sur le vrai `ubatch.n_tokens` (`ne[2]` = réservé = 2).
8. **Prompt B ajouté à `tools/verify.sh` comme régression — FAIT et VERT cette session** : `runB` /
   `checkB` (comparaison via `tools/cmp.py`) ; masters dans **`/data/o35/o35/masters/`** (pas `/tmp`,
   que `runB` écrase — comparer un run à son propre fichier `/tmp` donnait **0 par construction**).
   Attendu et vérifié : `cleanB` = **0**, `cpufbB` = **0**, `allcpuB` = **0** vs leur master,
   `cpufbB` vs `cleanB` = **121**, `cpufbB` vs `allcpuB` = **130**, `allcpuB` vs `cleanB` = **137**.
   Les constantes **8 et 28** attendues auparavant étaient **fausses** (mesure en texte brut) et
   produisaient deux `REGRESSION FAIL`. Les changements futurs sont vérifiés contre cette référence,
   pas contre `baseB`.
9. **Premier essai sans MTP fait** (run long prompt B `-n 1024`, froid, `pin 1990`) : **16,26 t/s** et
   **0,05 MiB** lus sur le disque pendant le décode (chemin tout-GPU : 20,13 t/s et **115 MiB**).
   Prochaine étape réelle : **activer MTP** (`blk.40.nextn.eh_proj.weight` ignoré) pour créer des
   batches de 2-4 tokens et rendre `max_batch_cached = 4` utile ; mesurer le t/s à `-n 1024` avec MTP.
