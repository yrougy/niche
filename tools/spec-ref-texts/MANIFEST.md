# Références de texte spec (LFM2.5-24B-A2B IQ3_XXS)

Sonde : prompt « Translate to French: The cat sleeps… », temp=0, seed=42, 200 tok
(voir tools/probe_traffic.py, bloc determinism ; sessions pilotées par tools/sweep_session.sh,
TRAFFIC=full|text ; sorties de session dans /tmp, à copier ici si elles deviennent référence).

- **Famille A** (md5 4d6f138d…) — spec 64 × trafic complet (cache préalable) = TEXTE PROD
  actuel, déterministe : base1, ctx65, smoke, bs1, on/off/on2, probe*, clone*, minacc025, p10, fast1.
- **Famille B** (md5 adc97e17…) — decode plain (SPEC=none) et spec SANS cache préalable
  (sessions text-only 32k/65k) : specnone, n65, n6561, q4none, q4spec, s65, s6561.
- A ≠ B = le spec × cache-reuse est lossy (décision 26/10 : ACCEPTÉ, prix du ×3,5 rep).
- **Famille C** — n_max courts, uniques : nmax16, nmax32 (knobs fermés, archivés).
- rp=1.0 : rp1spec / rp1none — deux familles propres (rp innocenté comme déclencheur).

Usage non-régression : rejouer probe_traffic sur la prod ; le texte temp-0 doit rester
md5 4d6f138d… (famille A). Tout changement de config qui déplace le md5 = changement de
distribution de sortie (à valider explicitement).
