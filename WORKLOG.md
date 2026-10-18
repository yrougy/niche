# WORKLOG — appliquer les idées de Strata pour monter le tg du port (session du 2026-10-12 après-midi)

Objectif : le port (branche `experimental/ornith-expert-cache`, HEAD `47a1dbbfc`) donne pp +60 % mais tg au niveau du bruit vs baseline amont. Analyser les idées de Strata (sans réimplémenter Strata) et les appliquer une par une. L'utilisateur dira quand faire une pause.

## État des outils (à jour)

- `tools/bench.sh` : bench llama-server via timings `/completion`, modes `fb|static|lru|off` × `mtp`, env `PORT=8002 CACHE_MIB=4096 CTX=8192 PP_N=2048 TG_N=128 DEPTH=0 COLD=1`. Flags alignés sur le billet rougy.net (fa on, KV q4_0, ub 512, t 8, parallel 1). **Fixés** : attente de la fin du serveur avant résumé (les lignes hit rate sont émises par `~llama_moe_cache` → `log_stats()` `src/llama-moe-cache.cpp:887`, et le script fermait le fd du log avant) ; bug moniteur `$10` → `${10}` (bash lit `$10` comme `${1}0`) ; `printf "%d"` dans l'awk du moniteur (locale fr → `3,91554e+11`).
- `tools/chat.sh` : même config, même fix moniteur.
- `tools/analyze_moe_profile.py` sur `tools/prof2.csv` : courbe couverture q↔hit (q=25→51,7 %, 51→71,5 %, 76→83,3 %, 102→90,8 %, 128→95,4 %). 40 couches × 256 experts, ~0,95 MiB/expert.

## Mesures de référence

- **Baseline billet (upstream mai, build DIFFÉRENT du nôtre)** : pp 367-398, tg128 25,78 (peak 28-30) **avec MTP**.
- **Run utilisateur `fb mtp` CACHE_MIB=4096 CTX=8192, 14:28** (`/tmp/bench/server.log`) : **pp 604-640, tg128 23,17** (tg_3s 22,30), draft acceptance 0,596 mean len 2,76. → +60 % pp, **−10 % tg** vs billet.
- Avertissements de ce run : `MADV_POPULATE_READ failed: Invalid argument` puis `Cannot allocate memory` ; `mlock failed on a 274432-byte range: Cannot allocate memory` → **le complément ne tient plus en RAM 15 GiB quand le bank passe à 4096** (avant : bank 1024, complément ~10,5 GiB tenant). À traiter (point 3b).
- Pas de lignes hit rate dans ce log : cause identifiée (course fd/kill), corrigée dans bench.sh.

## Point 1 — draft MTP duplique-t-il le cache ? **RÉSOLU : non**

Hypothèse initiale (le contexte draft hérite `moe_cache_size` et construit un 2ᵉ bank) **fausse** : `common/speculative.cpp:2569-2574` met déjà `moe_cache_size = 0` pour tout contexte draft ("the MoE cache is only used by the target context"). Le log du run confirme : un seul bank, MTP actif (acceptance 0,596). Rien à corriger.

## Mesures propres sur LE BUILD COURANT (page cache chaud, 15:0x, bench.sh corrigé)

| mode | pp 2048 | tg 128 (MTP) | acceptance |
|---|---|---|---|
| `off mtp` (baseline amont, auto-offload) | **421,5** | **24,18** | 0,596 / len 2,76 |
| `fb mtp` (bank 4096) | **612,4 (+45 %)** | **22,00 (−9 %)** | 0,596 / len 2,76 |

→ Le port **coûte ~9 % de tg malgré 71 % de hit** : le surplomb structurel (sous-graphe CPU + `ggml_add` + sync par couche × 40) mange le gain du hit rate. Conclusion de travail : les idées qui **suppriment du travail** (tail-skip) ou **évitent la branche CPU** (monter le hit) restent bonnes, mais il faudra aussi mesurer le surplomb du chemin bank lui-même.

**Hit rate mesuré à bank 4096** (llama-cli -v, prompt A, `-n 8`, fallback) : **71,30 %** (3394 hits / 1366 misses), bank 4095,16 MiB = 4481 slots ≈ **112 experts/couche (43,8 % des experts)**, `uploaded = 0,00`. La courbe prédite par `prof2.csv` (90,8 % à q=102) est **trop optimiste** : le CSV ne compte que ~1 sélection/token/couche (`expert_used=1` inféré, 335184 sélections pour 8208 tokens) — il ne voit qu'une fraction des 8 picks réels. **La source d'autorité pour la couverture, c'est le hit rate mesuré, pas la courbe du profil.**

**Logs serveur** : les lignes `impl:` (construction) et `llama_moe_cache:` (stats, émises par `~impl`) sont **supprimées par le logger du serveur** (composant inconnu filtré ; les W passent). Workaround : `--moe-stats <file>` (CSV) — à câbler dans bench.sh. Les runs llama-cli avec `-v` les montrent normalement.

**RAM** : à bank 4096, `MADV_POPULATE_READ` échoue (ENOMEM) et `mlock` bute toujours sur RLIMIT 1,945 GiB (échec à 405504 octets, comme à bank 1024). Le complément (9346 MiB d'host experts) ne tient plus : point 3b confirmé.

## Point 2 — tail-skip sur le zéro-slot (idée Strata la plus rentable, +13-16 % mesuré sur P100)

**IMPLÉMENTÉ et mesuré (option `--moe-cache-tail-skip N`, 0 = désactivé).** Avec le fallback CPU, les picks non résidents sélectionnés au rang ≥ N (0-indexé ; `argsort_top_k` trie bien par probabilité décroissante) sont sautés : contribution 0, ni CPU ni GPU. Mécanisme réel (différent du plan initial) : le graphe pose `ids_cpu->extra = N` (extra est NULL-initialisé, inutilisé sur src[2] — seuls les poids src[0] l'utilisent, vérifié) ; `mm_ids_helper` (ggml-cpu.c) saute `id >= tail_skip` dans le grouping et **zérote** leurs lignes dst via la boucle de zéroage existante. Côté GPU : rien à changer, le zéro-slot est déjà lu pour les non-résidents. Compteur `skipped` ajouté aux stats ; validité : fallback-only, rang dans [1, n_expert_used-1] sinon warn + désactivé. Touchés : llama.h, llama-cparams.h, llama-context.cpp ×3, common.h/.cpp, arg.cpp, llama-moe-cache.{h,cpp}, llama-graph.cpp, ggml-cpu.c ×3. Build OK. Test fonctionnel : `skipped = 1769`/10675 misses à N=7 (16,6 %), pas de crash, sortie cohérente.

**Mesures (TG_N=256, page cache chaud, même build) :** off no-mtp **24,37** ; fb tail0 no-mtp 19,58 ; fb tail6 no-mtp **20,55 (+4,9 %)** ; off mtp 24,18 ; fb tail0 mtp 22,00/22,88 (bruit ±4 %) ; fb tail7 mtp 21,33 (acceptance 0,545) ; fb tail6 mtp **25,29** (acceptance 0,634).

**Analyses :** (1) signal propre sans MTP : tail 6 = +4,9 % de décode brut ; tail 7 (16,6 % du travail CPU) ≈ neutre. Gain réel mais modeste — loin du +13-16 % de Strata (leur baseline P100 était du streaming PCIe lent, ici le CPU est déjà compétitif). (2) **DÉCOUVERTE clé : sans MTP, `off` (24,37) bat `fb` (19,58) de 20 %** — l'auto-offload amont laisse ~7 GiB d'experts résidents en VRAM nativement (ordre de couche, zéro surplomb) et streame le reste ; c'est plus rapide que notre chemin CPU. (3) Le port ne regagne qu'avec MTP (verify ≤4 tokens : fb 19,58 → 22-23) car son surplomb est **par passe** (sous-graphe CPU + ggml_add + syncs × 40), pas par token — c'est LA cible tg. Avec tail 6 + MTP le port atteint 25,29 > 24,18 (off mtp), mais l'acceptation (0,634 vs 0,596) dépend du contenu généré : partie du gain non imputable proprement. (4) Caveat qualité : les picks de rang ≥ N sont réellement droppés (poids softmax petits mais non nuls) ; le bit-à-bit avec masters/ ne vaut que tail_skip=0. bench.sh accepte `TAIL_SKIP=N` (mode fb).

## Point 3 — uploads vs CPU (réorienté par les mesures : DÉCOUVERTE PRINCIPALE de la session)

**Statut : mesures faites, conclusion claire.** Le split upload/CPU calibré n'est pas nécessaire : les mesures montrent quel chemin est gagnant selon le mode de batch.

**Mesures (TG_N=256 sauf mention, page cache chaud) :**

| config | pp | tg |
|---|---|---|
| static bank 4096 no-mtp | 428 | 24,14 |
| static bank 2048 no-mtp | — | 19,83 |
| static bank 4096 tail6 no-mtp | 426 | 24,06 (skipped 2228 picks, uploads inchangés 5310 vs 5298 MiB) |
| **static bank 6144 no-mtp** | **489,5** | **30,00** |
| static bank 7168 | OOM (bank + KV + poids > 8 GiB) | |
| static bank 5120 mtp | 423 | 18,49 (verify 4 tokens NON caché → chemin normal) |
| fb bank 5120 mtp | 436 | 25,69 (fit_params warning : VRAM saturée, pp dégradé) |
| fb bank 4096 mtp tail6 | 612 | 25,29 |

**Analyses :**
- **Test statique avec MTP, scratch 4 — CONFIRMÉ** : `static` bank 5120 mtp = **tg 38,21 / 36,65** (2 runs, acceptance identique 0,552 ; vs 18,49 avec scratch 1, vs 24,18 off mtp) — cacher les verify MTP en mode static est LE gagnant de la session. Code : `max_batch_scratch` 1→4 (llama-moe-cache.cpp:322, commenté "size of a verified MTP draft batch"). Scratch 4 aide aussi sans MTP : static 6144 no-mtp = **33,37** (vs 30,00 avec scratch 1) — un pool scratch plus grand améliore la réutilisation LRU entre tokens, compense largement la baisse de q. Bank 6144 + MTP = OOM même à CTX 4096 (le draft ~262 MiB ne rentre pas) ; 5120 est le max avec MTP à CTX 8192.
- **Régression complète passée** (`tools/verify.sh`, bank 1024) : toutes les sorties bit-identiques aux masters (diff = 0) en static, LRU, pin, fb — le rewrite de `plan()`, le code tail-skip (à 0) et scratch 4 sont sans effet comportemental quand les options sont désactivées. Les stats affichent désormais `skipped = 0` partout (nouveau format OK).
- **Recommandation pratique en l'état :** avec MTP → `static` bank 5632 avec `-c 4096` (tg ~39,8, pp ~446) — +65 % vs off mtp ; bank 5120 avec `-c 8192` si contexte long nécessaire (tg ~38,4). Génération pure sans MTP → `static` bank 6144 (tg 33,4, pp 483). Le mode fb (CPU) ne sert plus que pour les gros pp purs (612).
- **static + MTP est mauvais** : `max_batch_cached = min(32, sci/n_expert_used)` avec `max_batch_scratch=1` (llama-moe-cache.cpp:322) limite le mode static au cache 1 token ; les verify MTP (4 tokens) retombent sur le chemin normal → 18,49. **Piste de code : remonter `max_batch_scratch` à 4** (scratch 4× plus grand, q réduit) pour cacher les verify — LE point qui pourrait donner static+mtp > 30. À tester.
- **tail-skip en mode static : inefficace** (mesuré) : un expert skipé dans un ubatch est head-pické dans un autre → uploadé quand même ; les picks skipés ne suppriment des uploads que si l'expert est tail-only dans TOUS les ubatchs. Garder tail-skip pour fb seulement (où il supprime vraiment du CPU : +4,9 %).
- **fb + MTP** : seul mode qui cache les verify (slots statiques, pas de limite de tokens). fb 4096 tail6 = 25,29 ; fb 5120 = 25,69 mais bank 5120 + draft + KV 8192 ne tient pas en VRAM (warning `common_fit_params` → pp effondré 436). Bank max sûr avec MTP+CTX8192 : 4096.
- **3b FAITE** (voir section Point 3b) : budget RAM du complément + pin borné par RLIMIT_MEMLOCK + skip des trous de mmap dans populate/pin/release. static 5120 mtp désormais propre (zéro warning), tg 38,42 confirmé après fix.
- Code ajouté pour ce point : tail-skip étendu aux modes static/LRU dans `plan()` (min-rank par expert → zéro-slot, pas d'upload ; `n_extra=2` quand tail_skip>0 ; `copy_experts` ne copie plus le zéro-slot) — implémenté mais jugé inefficace en static, conservé (utile en LRU ? à voir).

## Point 3b — budget RAM du complément (FAIT, 12/10/2026)

**Diagnostic affiné** : les warnings observés à bank ≥ 5120 avaient DEUX causes distinctes :
1. **Trous de mmap** : après offload GPU, `llama_mmap::unmap` munmappe les fragments inutilisés (poids attention des couches GPU, padding). Le bloc « rest » de `cache_complement()` peuplait ces trous → `MADV_POPULATE_READ: Invalid argument` (EINVAL = adresses non mappées) et `mlock: Cannot allocate memory` (ENOMEM = idem). Ce n'était PAS un manque de RAM.
2. **RLIMIT_MEMLOCK = 1991,97 MiB** (`ulimit -l` = 2039780 kB) : le pin par défaut 1990 MiB + l'arrondi aux pages de mlock (6604 ranges, ~13 MiB de dépassement) dépassaient la limite → mlock ENOMEM (Linux renvoie ENOMEM quand RLIMIT est dépassé, pas EPERM).

**Correctifs (3) :**
- `llama-mmap.cpp` : `populate`/`pin`/`release` itèrent maintenant sur `mapped_subranges(first,last)` — sous-intervalles intersectant `mapped_fragments`, trous sautés. Supprime tous les warnings populate/mlock liés aux trous.
- `llama-moe-cache.cpp` : helper `llama_moe_cache_ram_available()` (lit MemAvailable dans /proc/meminfo, 0 si inconnu → pas de plafond). Budget = MemAvailable − 1 GiB de marge ; le « rest » (non-experts, toujours nécessaire au decode) est peuplé en premier puis déduit du budget ; les ranges du complément sont **sélectionnés du plus chaud au plus froid** (ordre `ranked` du profil, puis ordre fichier pour les experts sans données de routage) jusqu'à épuisement du budget ; les ranges sélectionnés sont fusionnés avant populate (une syscall par bloc contigu comme avant). WARN « X MiB left out of RAM » si le complément ne tient pas — leurs misses relisent le GGUF (correct mais lent).
- Pin : ne pinne que les ranges **sélectionnés**, et `pin_limit = min(pin_bytes, RLIMIT_MEMLOCK − 4 MiB)` avec **accounting page-aligné exact** (longueurs arrondies comme les aligne mlock : `first & ~(page-1)`, `min(align_up(last), mm->size())`) — le kernel compte exactement ces octets, la marge de 16 MiB du premier jet était trop juste (warning mlock réapparu à bank 6144) et est tombée à 4 MiB.

**Tests :**
- static 5120 : plus aucun warning ; complement 5398 MiB + rest 1986 MiB peuplés (tiennent dans le budget ~14 GiB → pas de plafond, correct) ; pinned 1975,92 MiB (plafond RLIMIT−16 appliqué) ; zéro lecture disque au decode (moniteur majflt=0).
- Chemin plafond validé de bout en bout (marge temporairement montée à 10 GiB) : budget ≈ 5 GiB → 1996 MiB sélectionnés en ordre chaud-d'abord + 5120 MiB left-out + WARN émis ; pin ⊆ sélection. Marge recréée à 1 GiB.
- `bench.sh static mtp` bank 5120 après fix : **tg 38,42** (cohérent avec 38,21/36,65), pp 434, acceptance 0,552.
- **bank 5632 + MTP à CTX 4096 : ça tient — NOUVEAU RECORD tg 39,81-39,86, pp 446-448**, zéro warning, zéro lecture disque. bank 6144 + MTP à CTX 4096 : OOM GPU (cudaMalloc 262 MiB du draft échoue) — 5632 est le bank max avec MTP.
- Après accounting exact du pin : pinned 1987,78 MiB (cap RLIMIT−4), zéro warning à bank 5120 et 5632 ; `verify.sh` re-passé 100 % bit-identique.
- `verify.sh` : **100 % bit-identique aux masters** (diff = 0 partout) après les trois correctifs.

## Point 4 — chaleur vivante + swaps au lieu du seed statique (FAIT, 12/10/2026)

**Implémenté** : `--moe-cache-adapt N` (llama.cpp, `src/llama-moe-cache.cpp` `adapt_tier`). Re-rank de la tier épinglée d'un cache static+cpu_fallback tous les N tokens cachés (compteur **par groupe**, incrémenté sur la copie du slot map du premier layer du groupe). Classement par **moyenne mobile des compteurs live** : `heat = heat×3/4 + delta_fenêtre`, **prior initialisé aux counts du profil** (`load_profile` renvoie maintenant les counts bruts). Swap : l'expert promu est écrit dans le slot de l'expert démote (copie brute mmap→bank, `slots`/`cpu_map`/`key_of` mis à jour en place avant la copie du slot map → le batch voit un tier cohérent, pas de double-computation). Cap : `q/4` swaps/round/couche. Nécessite `--moe-cache-static` + `--moe-cache-cpu-fallback` (sinon WARN + désactivé). Les poids ne changent pas : seulement la résidence.

**Pièges trouvés en mesurant** :
1. delta brut de fenêtre (64 tokens, sans prior) → **churn** : 1253 swaps en un round, tg 32,7-34,5 vs baseline 39,3-39,7. Rejeté.
2. Prior + EMA + fenêtre 512 → **zéro trigger** sur le bench apparié au profil, tg 38,97 ≈ baseline : **pas de régression** sur workload apparié. C'est le comportement voulu (le seed n'est jamais cassé par une fenêtre courte).
3. Session longue hors profil (Kyoto, 1000 tokens, bank 1024, ADAPT=128) : hit **9,75 % → 26,43 %** (×2,7), tg **15,93 → 17,3-17,5 (+9 %)**, swaps convergents (94 → 2 → 1 par round). Le prior décroît (×0,75/round) et se fait override par la chaleur persistante en ~4-5 fenêtres.

**Déterminisme** : texte identique entre deux runs adapt-on (seuls le spinner et les t/s diffèrent). La différence adapt-on vs adapt-off = différence numérique CPU-vs-GPU sur les experts non résidents — inhérente au cpu_fallback, pas à l'adapt (verify.sh n'a toujours couvert que les chemins tout-GPU).

**Comparaison Strata** : Strata decay ×0,7 / gain>1,5 / ≤96 swaps/round ; ici ×0,75 / cap q/4 par couche / pas de seuil de gain. Même esprit, paramètres différents.

**Outils** : `bench.sh` accepte `ADAPT=` et `VERB=1` (les logs INFO de la lib llama sont masqués sans `-v` — c'est un piège de lecture des mesures : « swaps = 0 » était un artefact de log level).

## Point 5 — gratuit : taskset (Strata +10 % decode), route prior (opt-in qualité)

Statut : **TESTÉ — sans effet sur la config recommandée**. `taskset -c 0-3` (cœurs physiques seuls, i7-4790 : siblings (0,4)(1,5)(2,6)(3,7)) + `THREADS=4` (nouvel env dans bench.sh) sur static 5632 mtp CTX 4096 : **tg 39,87 vs 39,81-39,86 sans taskset** (bruit). Explication : le scheduler place déjà les threads sur des cœurs physiques distincts, et le decode de cette config est GPU-bound (le CPU ne fait que les misses ~15 % et les uploads). Le gain taskset de Strata concernait un decode CPU-bound (mode fb) — hors config cible. À ne pas reprendre.

## Point 6 — complétude du profil de routage (FAIT, 12/10/2026, nouvelle frontière de performance)

**Diagnostic** : le hook `--moe-profile-output` marchait mais `prof2.csv` ne comptait que ~1 pick/token/couche (335184 sélections pour 8208 tokens, soit ~1047 tokens capturés) → le seed statique classait mal ~54 % des experts au bank 4096 (recouvrement top-112 ancien/nouveau : 45,8 %). De plus `--moe-profile-output` + cache actif = **GGML_ABORT** (`llama-moe-cache.cpp:1122`, « selected in the same split as their MoE cache lookup ») : les copies hôte du profil forcent un split CPU là où le lookup du cache a besoin des ids calculés dans un split antérieur. Tentative de déplacer le hook en fin de `build_moe_ffn` : échoue (le décalage casse la couche suivante). **Retenu** : le profil est ignoré quand le cache est actif (WARN dans `llama-context.cpp`), on profile en config off — le routage ne dépend pas de l'endroit où vivent les poids.

**Outil** : `tools/profile.sh` — serveur en config off + `--moe-profile-output`, rejoue le trafic de bench.sh (warmup, pp 4096, tg 256×2), vérifie la complétude. `prof3.csv` : 4637 tokens × 8 picks × 40 couches, complet (couche 39 partielle = l'optimisation out_ids de la dernière couche, attendue).

**Mesures (static + MTP, CTX 4096, même trafic)** :
- bank 4096 : ancien seed hit 79,1 % tg 36,51 pp 405,7 → nouveau seed tg 36,33 pp 474,3 (tg inchangé : les misses en statique sont des lectures hôte sans fault, peu coûteuses en decode ; le gain est sur le pp)
- **bank 5632 : nouveau record tg 41,48 / pp 511,51, hit 89,0 %** (vs ancien seed 39,8 / 446) — +4,2 % tg, +14,6 % pp vs l'ancien record
- `verify.sh` : 100 % bit-identique après le correctif (le seed ne change pas les sorties, seulement la résidence)

**Honnête sur la méthode** : `prof3.csv` est généré sur le trafic du bench → le seed est parfaitement apparié au benchmark. Pour un déploiement réel, `profile.sh` doit rejouer le trafic réel (c'est exactement ce que fait Strata : profiler l'usage, pas un prompt synthétique).

**`tools/{bench,serve,chat}.sh` basculés sur prof3.csv** (défaut ; `PROFILE=` surchargeable).

## Point 7 — montée en quant mesurée (FAIT, 12/10/2026)

Contexte : le port déplace la limite de la VRAM (8 GiB) vers la RAM (15 GiB, MemAvailable ~11) + PCIe + disque rotatif. Test sur les quants locaux du même modèle (profils régénérés par quant — le routage dépend des poids ; `profile.sh` accepte maintenant `MODEL=` et un chemin absolu, `bench.sh` accepte `MODEL=`).

| quant | taille | bank max testé | hit | tg | pp | acceptance MTP | majflt |
|---|---|---|---|---|---|---|---|
| UD-IQ2_M | 11,1 GiB | 5632 | 89,0 % | **41,48** | 511,5 | 0,42-0,59 | 0 |
| UD-Q2_K_XL | 11,7 GiB | 5376 | 81,7 % | **40,27** | 486,2 | 0,65-0,68 | 5 |
| UD-Q3_K_M | 15,9 GiB | 4608 | 68,0 % | **6,74** | 47,6 | 0,65-0,77 | 16 601 |

- **Q3_K_M = mur confirmé** : le hot set par expert ×1,44 ne tient plus dans la RAM ; le tail chaud part en fautes disque (16 601 majflt) → decode à 6,7 t/s. Q3_K_S (16,4 GB, −4 %) bute au même mur — téléchargement interrompu à 10,3/16,4 GB (`curl -C - .../Qwen3.6-35B-A3B-UD-Q3_K_S.gguf` reprenable), inutile pour le test.
- **Q2_K_XL = palier viable** : −2,9 % tg / −5 % pp vs IQ2_M pour un palier de qualité supérieur (bpw effectif plus élevé, classement unsloth). La compensation vient de l'**acceptance MTP 0,65-0,68 vs 0,42-0,59** (mean len 3,0 vs 2,1-2,8) : la tête draft est moins bruitée, donc moins de passes du modèle cible par token généré. Preuve mesurée de qualité supérieure, pas un argument bpw.
- L'analyse a priori (Q3 ≈ 20-28 t/s) était **trop optimiste** : elle ignorait que le tail chaud hors RAM tombe sur le disque rotatif (~100-150 MB/s), pas sur la page cache. Le cliff est l'occupation RAM du hot set, pas la taille du fichier.
- IQ3_M/Q3_K_S non testés (non locaux) : même taille que Q3_K_M ±5 %, même verdict.

## Point 8 — pcie-frac auto (FAIT, 12/10/2026, verdict : share = 1,0 — déjà implémenté par static)

**Idée Strata** : partager les experts manquants entre le CPU et l'upload PCIe (`--pcie-frac`), auto-calibrée depuis le temps CPU mesuré : `share = L/(L+1,85R)` avec R = octets d'expert / temps CPU (GB/s) et L = vitesse de lien sondée (docs/DETAILS.md:1642-1665).

**Constat structurel qui clos l'axe** : la formule de Strata price le **recouvrement** CPU∥GPU (leur engine publie la part GPU en premier : « GPU starts while CPU works », src/core/expert_source.cpp:3664-3665). Le scheduler de llama.cpp ne le fournit pas : `ggml_backend.cpp:1864-1871` synchronise le split précédent avant chaque split et la copie des entrées synchronise le backend source (`:1843`) — le temps CPU est **exposé**. Dans un scheduler sérialisé, l'optimum du partage est toujours à un bord : tout CPU ou tout upload. Les deux modes existants sont exactement ces deux bords (fb = share 0, static = share 1) ; un knob hybride n'a pas de raison d'exister dans ce port.

**A/B propre (12/10, même session, même trafic, sans -v, MTP, CTX 4096, seed prof3)** — tiers épinglés quasi identiques, seul le traitement des misses change :

| config | tier épinglé | traitement des misses | tg |
|---|---|---|---|
| static CACHE_MIB=5632 | 4676 (+1312 scratch) | upload 0,905 MiB → GPU | **39,85** |
| fb CACHE_MIB=4264 | 4529 | CPU | 32,58 |

**+22 % en faveur de l'upload.** Complété par la session précédente : fb@5632 (tier 5988, +28 % plus épinglé que static@5632) = 39,3-39,7 — ne bat pas static@5632 (39,85-41,5). Si le CPU était moins cher par miss, fb avec un tier strictement plus grand gagnerait. Il ne gagne jamais. Caveat honnête : les deux modes génèrent des textes différents (arrondis CPU-vs-GPU sur les non-résidents) → acceptance MTP et hit rates divergent (fb@4264 hit tier 75,7 % mesuré sur stats.csv, acceptance 0,69 ; static hit 90,5 %, acceptance 0,55) — la direction est cohérente entre sessions mais les coûts absolus par miss sont encadrés, pas mesurés proprement.

**Chiffres dérivés** : expert = 0,905-0,9125 MiB (5464,55 MiB / 5988 slots) ; upload ≈ 80 µs/miss à PCIe 3.0 x16 (L ≈ 12 GB/s — lien très loin d'être saturé : 28,3 MiB/token soutenus = 1,2 GB/s) ; CPU ≈ 0,1-0,25 ms/miss (encadré par les Δ de tg). R ≈ 3,6-4,7 GB/s. La formule de Strata donnerait share ≈ 0,6, mais elle est calibrée pour leur régime recouvrant ; ici la comparaison directe (80 µs recouvrable sur le stream GPU vs ≥100 µs CPU exposé) donne **share = 1,0** — ce que static fait déjà. Le côté CPU ne gagnerait que sur une machine avec CPU plus rapide que le lien (R > L/1,85 ≈ 6,5 GB/s) ET un engine qui recouvre CPU∥GPU — pas ce port. fb reste utile pour le pp pur (pin).

**Non-reproductibilité du record** : re-mesuré aujourd'hui, static 5632 = **39,85** (pp 531, plus haut que le record 511). Le tg de cette config varie 39,9-41,5 selon les runs ; le chiffre honnête est ~40. Le `-v` n'est pas en cause (39,91 avec vs 39,85 sans, ±0,2 %) — la session adapt n'était pas faussée par VERB.

## Frontière VRAM confirmée (bank max avec MTP)

- 5632 + MTP à CTX 4096 : OK (record tg 41,5 avec le seed prof3.csv). 5888 + MTP : OOM CUDA en cours d'exécution (compute buffer, 0.05 s). 6144 + MTP : OOM au démarrage même à CTX 2048 — le KV n'est pas le facteur limitant (poids non-experts ~1,7 GiB + draft 262 MiB + compute buffers). **5632 = bank max avec MTP** ; sans MTP, 6144 tient (tg 33,4).

## Rejetés (mesuré perdant chez Strata ou hors matériel)

Pipeline CPU∥GPU (−7 %), foresight slots, VMM (Turing+), batched DMA (CUDA 13), tensor cores pour le decode, CUDA graphs (absents de l'API de ce build).

## Prochaine action immédiate

Record actuel : **static 5632 mtp CTX 4096 seed prof3.csv = tg 39,9-41,5 (41,5 = pic non reproductible, re-mesuré 39,85 le 12/10) / pp 511-531, hit 89-90 %** (+71 % vs off mtp 24,18). Points 1-3b, 4, 5, 6, 8 terminés ; Point 7 (quant) en attente de décision. Axes restants, par valeur attendue : (1) **profil réaliste pour le déploiement** — `profile.sh` rejoue le trafic du bench, hit 89 % = plafond apparié au bench ; pour servir autre chose, re-générer le seed sur le trafic réel (ce que fait Strata : profiler l'usage) — et `--moe-cache-adapt 512+` comme filet live (×2,7 hit / +9 % tg mesurés sur session longue hors profil, sans régression sur workload apparié). (2) ~~pcie-frac auto~~ **clos au Point 8** : share optimal = 1,0 sur cette machine (upload 80 µs/miss vs CPU ≥100 µs exposé — le scheduler llama.cpp sérialise CPU et GPU, pas de recouvrement possible) ; static l'implémente déjà, fb = l'autre bord (share 0). (3) tail-skip re-mesuré avec le nouveau seed (faible espoir en statique). (4) **quant par expert pilotée par profil** : mise en attente — verdict complet dans la mémoire `o35-per-expert-quant-feasibility` (requant vers le haut depuis le GGUF = zéro gain de qualité ; la vraie version exige un pack offline depuis une source Q8_0 — projet qualité, pas tg).

**Commités (12/10/2026)** : llama.cpp `experimental/ornith-expert-cache` → `f3088e5f6` (trous mmap dans populate/pin/release) + `ac38254f8` (tail-skip + scratch 4 + budget RAM/RLIMIT du complément) ; workspace o35 `master` → `d4589af` (WORKLOG, bench.sh, serve.sh, prof2.csv, chat.sh, try.sh). `tools/serve.sh` : serveur avant-plan 0.0.0.0:8002 config optimale, CTX en variable d'entrée, bank dérivé du CTX (5632 ≤4096, 5120 ≤8192, décru au-delà) — validé (health 200).

---

# Session 2026-10-14 : port sur LFM2.5-24B-A2B (Liquid AI, lfm2moe)

Modèle : `/data/models/unsloth/LFM2.5-24B-A2B-UD-IQ3_XXS.gguf` (8,96 GB disque = 8544 MiB, IQ3_S/IQ4_XS mélangés par couche + imatrix unsloth). Architecture `llama_model_lfm2moe` → graphe de `lfm2.cpp`, FFN via `build_moe_ffn` dès la couche 2 — **le cache MoE et le hook de profil s'appliquent sans changement de code** (vérifié : pas d'allowlist d'archi ; `llama-moe-cache.cpp:220` itère gate/gate_up/up/down indifféremment).

Géométrie : 40 couches dont 38 MoE × 64 experts (4 utilisés), 2 denses en tête ; `n_ff_exp=1536`. Slot = 3 projections ≈ **3,17 MiB** (couches iq2_s/iq3_s) ou **4,15 MiB** (iq3_s/iq4_xs) — 3,6× le slot Qwen. Experts totaux ≈ 7,34 GiB ; non-experts ≈ 0,5 GiB. Hybrid : ~11 couches attention (head_count_kv périodique 8/0), ~29 couches shortconv. **Pas de tête MTP** (pas de clé nextn) — le levier n°1 du Qwen (MTP+scratch4) n'existe pas ici.

Baseline off (auto-offload amont, CTX 8192) : **tg 32,3 / pp 836** — le modèle tient presque entier sur la 1070 (41/41 offloaded, 205 MiB débordent en Host : model 8536 vs 8098 dispo). C'est une baseline bien plus haute que celle du Qwen (23-26 avec MTP) : la marge du port y est structurellement plus faible.

Profil : `MODEL=... tools/profile.sh tools/prof-lfm.csv` → 38 couches, 688 844 picks (~4640 tokens × 4), 2261/2560 paires vues (98 % des experts au moins une fois, misses obligatoires 0,33 %). Courbe couverture : 25 % des experts → 83 % hit, 32 % → 91 %, **48 % → 96,9 %** (routing très concentré : top-64 experts globaux = 24 % des picks).

Benchs (CTX 8192, PP_N 4096, TG_N 256, seed prof-lfm.csv) :

| mode | pp | tg | hit decode |
|------|----|----|-----------|
| off | 836 | 32,3 | — |
| static 4096 | 810 | 31,6 | 89,3 % |
| static 6144 | 924 | 46,3 | 98,2 % |
| **static 6656** | **934** | **47,2** | **99,3 %** |
| static 7168 | — | OOM VRAM (« resource allocation failed ») au-delà de ~6900 | |
| fb 4096 | 875 | 24,9 | 75,4 % |
| LRU 6144 (sans seed) | 726 | 44,4 | 96,8 % |

- **Record LFM : static 6656 CTX 8192 = tg 47,2 (+46 % vs off), pp 934 (+12 %)** — meilleur tg jamais mesuré sur cette machine (devant Qwen 41,5 avec MTP).
- CTX 32768 : off tg 30,7 (les experts débordés coûtent), static 4096 tg 31,3 — le KV hybride est minuscule (45 MiB à 8192), la pression CTX est faible.
- `fb` est perdant sur ce modèle (24,9 < 32,3) : les experts tiennent presque tous en VRAM, le chemin CPU exposé ne sert à rien. Hiérarchie conservée : static > off > fb.
- Vérification texte : 2 runs static 6656 **bit-identiques** (déterminisme OK, équivalent du gate verify.sh). Off↔static↔fb divergent sur le prompt de bench (arrondis CPU des experts non-résidents, divergence dès le 10e caractère) — même comportement connu côté Qwen, sans rapport avec la justesse.

## Bugs/gotchas LFM découverts

1. **`fb` + LRU sans profil = crash CUDA** (illegal memory access au warmup, dans `ggml_backend_sched_copy_input`) : LFM seul. Avec seed profil, fb fonctionne. À investiguer : plan LRU (quota/scratch) avec tailles de couches hétérogènes (194 vs 266 MiB/couche) ; homogène chez Qwen, donc jamais rencontré.
2. **`bench.sh`/`profile.sh` passent `PROFILE` au serveur APRÈS `cd $LLAMA`** : un `PROFILE` relatif est introuvable côté serveur → le static tombe silencieusement en LRU (WARN « needs a routing profile »). Toujours passer un chemin absolu. (Les premiers benchs « static » de cette session étaient en fait des LRU, refaits avec seed.)
3. `--moe-profile-output` ignore le cache (inchangé) ; prof-lfm.csv complet du premier coup (pas le problème de prof2).

## Outils

- Profil persisté : `tools/prof-lfm.csv`.
- `tools/serve.sh` et `tools/chat.sh` : `MODEL`/`PROFILE` surchargables par env (défauts Qwen inchangés) ; `MTP=0` dans serve.sh pour modèle sans tête MTP (ex. LFM). Exemple LFM : `MODEL=.../LFM2.5-24B-A2B-UD-IQ3_XXS.gguf PROFILE=/data/o35/o35/tools/prof-lfm.csv MTP=0 BANK_OVERRIDE=6656 CTX=8192 tools/serve.sh`.
- Pas de masters de vérification LFM créés (les masters Qwen de verify.sh ne s'appliquent pas) ; déterminisme vérifié par double run.

## Suite d'outils reproductibilité (14/10/2026, non commités)

- **`tools/serve-lfm.sh`** : serveur LFM2.5 dédié — CTX **65536**, bank 6656 (plate de 8k à 64k, −128 MiB/8k au-delà), API OpenAI-compatible `0.0.0.0:8002`, mêmes flags d'API que serve.sh. Sourçe `tools/machine-lfm.env` (PIN/BANK_MAX prioritaires). Pas de MTP. **Validé en service (14/10)** : health 200, chargement ~2,7 s (mmap + page cache), tg 42,8 à 256 tok décode (off ~32), hit 95,9 % (39 956 utilisations) relevé via `--moe-stats /tmp/moe-stats-lfm.csv` (compteurs écrits au chargement, flush finaux à l'arrêt propre SIGINT/Ctrl-C). ⚠️ le template chat LFM2.5 active le mode raisonnement par défaut → la sortie arrive dans `reasoning_content`, `content` peut être vide ; `chat_template_kwargs thinking:false` ignoré.
- **`tools/calibrate.sh TAG GGUF_absolu`** : mesure sur machine cible (VRAM, RAM, `ulimit -l`→PIN, threads, nextn) → baseline off → profil (sauf `SKIP_PROFILE=1`) → **plafond de bank par recherche binaire** (OOM détecté dans le log serveur, sonde pp1024/tg128) → paliers tg/pp/hit → écrit `tools/machine-<TAG>.env` (lignes `VAR=${VAR:-mesure}` : l'env déjà exporté a la priorité). KV réel extrait de la breakdown VRAM (context − bank du dernier run). **Run LFM complet** : plafond 6912 (7168 OOM à CTX 8192), meilleur 6912 → tg 45,9 pp 889 hit 99,6 ; **KV ≈ 40 MiB/8k tokens réels** (hybride). ⚠️ le plafond est sondé sans MTP : pour un modèle à MTP il est optimiste — serve.sh garde sa propre dérivation bank/CTX, BANK_MAX ne sert que de coupe.
- **`tools/setup.sh [qwen|lfm|both]`** : build llama.cpp (warning si branche ≠ experimental/ornith-expert-cache — le port n'est hébergé nulle part, une nouvelle machine doit apporter le checkout) → téléchargement curl avec reprise `.part` (catalogue : `unsloth/Qwen3.6-35B-A3B-GGUF/…UD-IQ2_M.gguf` 11,5 GB ; `unsloth/LFM2.5-24B-A2B-GGUF/…UD-IQ3_XXS.gguf`) → check taille/RAM (cliff) → calibrate.sh. `SKIP_DOWNLOAD=1` dispo.
- `tools/machine-lfm.env` (machine de dev, 14/10 22:29) : PIN 1975, BANK_MAX 6912, BANK_BEST 6912, KV 40 MiB/8k, off pp 878 tg 31,5. `machine-qwen.env` non généré ici (les défauts codés en dur de serve.sh sont les mesures de cette machine ; calibrate le régénérera sur une machine cible via setup.sh both).
- bench/profile/chat/serve : chemins portables `LLAMA`/`HERE` surchargables par env.

## CTX 131072 mesuré (16/10)

- LFM2.5-24B-A2B tient ses **128k natifs** (context_length GGUF = 131072) : plafond bank 6144-6400 à CTX 131072 (6656 = OOM, 6144 = OK **tg 46,5 / pp 949 / hit 98,2 %**) — soit ~1,5 t/s (-3 %) sous la config 65536/6656. Pente réelle de KV ≈ 48 MiB/8k tokens (la plateau 8k→64k était réel, pas un artefact). `serve-lfm.sh` dérive maintenant le bank à **64 MiB/8k au-delà de 65536** (131072 → 6144) ; relancé en service à CTX 131072 bank 6144 le 16/10 (health 200).
- Donnée service intéressante : le serveur utilisateur (65536/6656, arrêté proprement le 16/10) affichait **hit 82,9 % sur 3,06 M utilisations** en trafic réel, contre 99 % en bench (le bench rejoue le trafic du bench). Piste : re-générer le seed profil sur le trafic réel du serveur (`--moe-profile-output` ?) — le hit réel est le vrai plafond en service.
- Méthodo : méfiance des benchs quand un serveur tourne déjà sur 8002 — bench.sh n'a pas pu bind, ses requêtes ont interrogé **l'autre** serveur et donné de faux OK. Vérifier `n_ctx_slot` dans server.log et `ss -ltnp | grep PORT` avant de conclure.

### Sampling model card (16/10)
- `serve-lfm.sh` applique la card LFM2.5 : `--temp 0.1 --top-k 50 --repeat-penalty 1.05` (env TEMP/TOPK/RPEN surchargeables) ; vérifié live via `/props` (temp 0.1, top_k 50, repeat_penalty 1.05 ; defaults llama.cpp conservés par ailleurs : top_p 0.95, min_p 0.05). Les tools de mesure (bench/profile/verify) gardent `--temp 0 -s 42` pour le déterminisme. Serveur relancé à CTX 131072 avec ces valeurs.

## TUI d'installation « Niche » (ex-Ornith, renommé : Ornith est un modèle bien trop gros pour nous) (18/10, à la maya.sh de project-maya)

- **`tools/niche.sh`** : le point d'entrée écran — machine → modèles → build → calibrage → serveur. Bootstrap `tools/.venv` + Textual **depuis les roues locales** `tools/wheels/` (textual 8.2.8, 9 roues 2,5 Mo ; les re-générer : `python3 -m pip download textual==8.2.8 -d tools/wheels`). `--plain` (ou stdin non-tty) = la même suite en texte brut. Journal complet : `tools/niche-setup.log`.
- Architecture reprise de maya : moteur sans écran `tools/niche_setup.py` (parle à un objet ui), pont à événements `tools/niche_bridge.py` (`Bridge` file d'événements + questions bloquantes ; `PlainUI` texte), écran Textual `tools/niche_tui.py` (onglets d'étapes, carte « en cours » + barre de progression, journal, menus de question ; le moteur tourne en thread daemon, l'écran pompe la file à 12 Hz).
- Catalogue = celui de setup.sh (qwen UD-IQ2_M 11,5 GiB, lfm UD-IQ3_XXS). **Réutilisation des poids présents** : absent / `.part` (reprise `curl -C -` auto) / **présent ailleurs** (arborescences `GGUF_DIRS` + `/data/models` + `~/Downloads`/`~/models` en plus de `MODEL_DIR`) → menu lier (symlink, défaut) / copier / re-télécharger. Garde-fou RAM (cliff hot set) repris de setup.sh.
- Testé (18/10) : reprise `.part` sur vrais fichiers HF (**« Resuming transfer from byte position 51200 »**, sha identique après reprise, `.part` nettoyé au renommage) ; téléchargement complet 8,5 GiB réel ; détection réutilisation + symlink OK ; TUI en headless (`App.run_test` + frappes clavier réelles sur les menus) ; `niche.sh --plain` sur la machine de dev : branche/binaire/RAM/GPU/GGUF tous detectés.
- Pièges Textual 8.2.8 relevés : `Static/Container(border_title=…)` n'existe pas au constructeur (mettre `widget.border_title` après mount) ; `RichLog.write()` n'accepte pas `style=` (passer un `rich.Text`) ; dans les tests, ne JAMAIS appeler `Bridge.ask()` depuis le thread asyncio principal (bloque la boucle d'événements) — pousser l'événement `ask` et répondre, ou piloter au clavier via `pilot.press`.
- ⚠️ en test pipe : la question « descendre <modèle> ? » a **télécharger** par défaut → un stdin vide lance un 11 GiB. Répondre « 2 » (passer).

### Correctif TUI (18/10, crash signalé à l'usage)
- **`Bridge._ask` était une tuple** → `answer()` faisait `_ask[4] = …` → `TypeError: 'tuple' object does not support item assignment` au premier menu (le TUI ne répondait à AUCUNE question). Passé en liste. Le test headless initial ne couvrait pas `Bridge.ask()` (il poussait l'événement à la main) — d'où l'oubli.
- UX menus améliorée : le choix **[défaut] est pré-surligné** (`OptionList.highlighted`, propriété nommée ainsi en 8.2.8 — `.highlight` n'existe pas, l'affectation passée inaperçue était un no-op silencieux en test) ; Entrée = défaut, ↑↓ pour changer.
- Non-régression pérenne : **`tools/test_niche_tui.py`** (headless via `App.run_test`) — rendu des événements + vrai `Bridge.ask()` depuis un thread avec réponses clavier (`pilot.press enter` / `down,enter`). Lancer : `tools/.venv/bin/python tools/test_niche_tui.py`.

## Choix du quant adapté à la machine (18/10, Niche)

- Étape Modèles de `niche_setup.py` : la liste des quants vient de l'**API HF en direct** (`/api/models/<repo>/tree/main`, tailles exactes en octets, mmproj exclus) ; repli sur le quant d'office si l'API est injoignable. Catalogue enrichi : `prefix` (glob famille), `mesure` (paliers mesurés maison), `evite` (murs mesurés).
- **Budget = hot set RAM** (contrainte mesurée 12/10, pas la VRAM) : `MemTotal − 3 GiB`. Machine de dev 15 GiB → 12,0 GiB. Reco = plus gros quant tenant au budget, en privilégiant un palier MESURÉ ≥ 75 % de la taille max → qwen **UD-Q2_K_XL** (11,45 GiB, marquée « limite »), lfm **UD-IQ3_XXS** (8,34, config de référence). Menu : 2 candidats + « liste complète » (⚠ par entrée : dépasse budget ou `evite`) + « passer » ; tout dépassement budget redemandé (confirmer/renoncer, défaut renoncer).
- Quanta multiples présents : le « meilleur » est réutilisé (MODEL_DIR d'abord, puis mesurés-bons, puis plus gros) — l'ancien tri alphabétique pick-ait IQ1_M sur la machine de dev qui contient 6 quants 35B (IQ1_M/IQ2_M/Q2_K_XL/Q3_K_M/Q4_K_M + .part Q3_K_S). Avertissement si le quant réutilisé est dans `evite`.
- Menu Prêt : `MODEL=<chemin installé>` exporté vers serve.sh/serve-lfm.sh (gagne sur machine-*.env) ; warning « refaire le calibrage » si le quant diffère du seed (profil de routage dépend du quant).
- Vérifié : menus plausibles sur machine vide (qwen : Q2_K_XL/IQ2_M en candidats ; lfm : Q3_K_XL/IQ3_XXS), liste complète annotée, réutilisation sur machine réelle = Q2_K_XL + IQ3_XXS.

###CTX demandé au menu Prêt (18/10)
- `niche_setup.ctx_choix` : question « quelle taille de contexte ? » avant chaque démarrage serveur, options annotées des pentes bank mesurées (qwen : 4k défaut bank ~5632, −512 MiB/8k, plancher 4096 ; lfm : plateau 6656 de 8k→64k, 131072 max natif → bank 6144 tg −3 %). Le choix passe en `CTX=` exporté (les serve*.sh dérivent le bank du CTX). Défaut = celui du script.
