# START — Dr. Zero auf unserem Cluster

Stand: 26.11.2026. Dieses Dokument ist der Startpunkt, damit ein Lauf auf *unserer*
Hardware reproduzierbar wird, statt die Referenzzahlen von Meta zu übernehmen.

## 1. Was hier geforkt wurde

- Upstream: `github.com/facebookresearch/drzero` (Meta Superintelligence Labs, Code zum
  Paper *Dr. Zero: Self-Evolving Search Agents without Training Data*, arXiv **2601.07055**)
- Unser Fork: `github.com/mariuszkreft/drzero`, `origin` = Fork, `upstream` = Meta
- Lizenz: **non-commercial** (LICENSE.md). Kommerzielle Nutzung braucht eine eigene
  Klärung, bevor das in ein Produkt wandert.

Der Code ist Forschungssoftware: Skripte mit festen Pfaden, keine Tests, kein CI,
`verl` als Vendor-Kopie im Baum. Er läuft, aber er läuft nur so, wie Meta ihn gebaut hat.

## 2. Was das Verfahren macht

Zwei Kopien desselben Basismodells (Referenz: `Qwen/Qwen2.5-3B-Instruct`) trainieren
sich gegenseitig, ohne einen einzigen menschlich geschriebenen Trainingsdatensatz:

1. **Proposer** bekommt ein Seed-Dokument plus Suchwerkzeug und schreibt eine Frage,
   die Antwort und einen Suchpfad. Die Frage hat eine Hop-Struktur (1–4 Sprünge).
2. **Solver** versucht dieselbe Frage mehrfach (`reward_rollout_n`, Standard 5) mit dem
   Suchwerkzeug zu beantworten.
3. Aus der Trefferquote wird ein **Schwierigkeits-Score**: alles richtig = zu leicht,
   alles falsch = zu schwer, beides 0. Dazwischen liegt der Ertrag
   (`compute_difficulty_score`: `(n - richtig) / (n - 1)`).
4. **Proposer** wird mit HRPO trainiert: statt pro Seed mehrere Kandidaten zu ziehen
   (Standard-GRPO), wird *eine* Frage pro Seed erzeugt und die Baseline über Fragen
   **gleicher Hop-Zahl** gebildet. Das ist der eigentliche Sparhebel des Papers — er
   ersetzt das verschachtelte Sampling.
5. **Solver** wird auf den erzeugten Fragen mit GRPO trainiert (Gruppengrösse 5,
   Format- und Antwort-Rewards, Mehrfach-Tool-Nutzung bis 5 Assistenten-Turns).
6. Ist der Solver besser, werden leichte Fragen wertlos, der Proposer muss härter
   werden. Das ist der automatische Lehrplan. Danach Checkpoint konvertieren
   (FSDP → HF), damit die nächste Iteration den neuen Solver als Reward-Geber lädt.

Formatrewards (aus `verl/custom_reward/reward_function.py`, hart im Code, nicht konfigurierbar):
Denk-Block vorhanden, Tool-Aufrufe deckungsgleich mit Tool-Antworten, Antwort kurz und
im Dokument enthalten, Frage nicht identisch mit der Antwort. Alles zu einem Viertel gewichtet,
`integrity_reward` als Tor (0 = der ganze Score fällt auf 0).

## 3. Was der Lauf braucht

### Referenzaufbau von Meta (gemessen aus den Skripten, nicht geschätzt)

- `iter1_challenger.sh`: **8 GPUs**, FSDP, `tp=2`, `dp=4`, sglang-Rollout mit
  `mem_fraction_static=0.25`, Batch 256, Offload von Parametern *und* Optimizer auf die CPU
- `iter1_gen_data.sh`: 8 GPUs, `tp=1`, `dp=8`, 512 Prompts pro Partition, Generierung
  mit `n=5`
- Prüf- und Trainingsfenster: Prompt 1536, Antwort 2560 Token
- `faiss-gpu`-Retriever über den Wikipedia-Dump `wiki-18` (`e5-base-v2`, topk 3)

### Was wir tatsächlich haben (heute gemessen)

| Host | GPU | Speicher | Anmerkung |
| --- | --- | --- | --- |
| 8× Spark (Cluster, 192.168.31.212) | GB10 | 128 GB *unified*, arm64 | trägt aktuell DeepSeek-V4.1-Flash, TP8/EP4, 1M Kontext — **belegt** |
| 192.168.31.99 | 2× RTX 3090 | 24 GB | trägt Ornith-35B, TP2 — belegt |
| m2.2 | 1× RTX 3090 | 24 GB | frei genug für kleine Sachen |

Damit gilt: **die Sparks sind unified-memory-arm64-Kisten, keine 80-GB-Hopper.** 3B
mit FSDP-Offload und TP2 passt dort knapp, `faiss-gpu` ist auf arm64/CUDA-13 der
wahrscheinlichste Bruchpunkt. Auf den 3090ern passt der Lauf nur mit 2 Karten und
deutlich kleinerem Batch, dafür ist der Stack x86 und erprobt.

### Speicher (kritisch)

- FAISS-Index `e5_Flat`: **64,6 GB** (HF-Metadaten, `usedStorage`)
- Korpus `wiki-18.jsonl.gz`: 5,1 GB gepackt, ~21 GB entpackt
- m2.2 Wurzel: **36 GB frei** — zu knapp. Ziel: der 3,6-TB-Mount
  (`/srv/nix-m2o/mounts/sdjs/…`, 1,5 TB frei) oder eine eigene Platte.

## 4. Einrichten

```bash
git clone git@github.com:mariuszkreft/drzero.git && cd drzero
git remote add upstream https://github.com/facebookresearch/drzero.git

python3.10 -m venv .venv && source .venv/bin/activate     # 3.10 oder neuer
pip install -e .            # falls kein Paket: requirements aus verl v0.5.0 ziehen
pip install "verl==0.5.0" sglang[all] faiss-gpu transformers torch
```

Ablage (nicht in den Fork committen):

```bash
mkdir -p /srv/nix-m2o/mounts/sdjs/drzero/corpus
cd drzero
save_path=/srv/nix-m2o/mounts/sdjs/drzero/corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

Daten vorbereiten (erzeugt `data/zero_ratio4321.parquet` und `data/test.parquet`):

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
```

## 5. Lauf auf unserem Cluster

Die Referenzskripte bleiben unangetastet; `run_sparks_iter1.sh` ist unsere Variante mit
den Parameter-Deltas für die Sparks. Vor dem ersten Start:

```bash
export DRZERO_CORPUS=/srv/nix-m2o/mounts/sdjs/drzero/corpus
export DRZERO_MODEL=Qwen/Qwen2.5-3B-Instruct   # 3B zuerst, nicht grösser anfangen
bash run_sparks_iter1.sh                # Phase 1: Proposer
bash run_sparks_gen_data.sh             # Phase 2: Synthese
bash run_sparks_iter1_solver.sh         # Phase 3: Solver
bash convert.sh                         # Checkpoint nach HF
```

Reihenfolge und Abbruchkriterien:

- **Gate 1** nach 25 Schritten (`save_freq`): Checkpoint existiert, Wandb zeigt
  `format_score > 0.2` — sonst Prompt-Format prüfen, nicht weiterlaufen lassen.
- **Gate 2** nach der Synthese: `wc -l data/zero_*.parquet`, Ziel ~50k Fragen.
  Unter 10k ist der Proposer noch zu schwach.
- **Gate 3** nach dem Solver: `data/test.parquet` auswerten (EM), Vergleich gegen das
  Basismodell ohne Training. Ohne messbaren Zuwachs ist die Iteration wertlos.

Erwartung: Referenz sind 8×80 GB. Auf 8 Sparks mit TP2 ist der Durchsatz pro Schritt
deutlich niedriger; die erste Iteration ist ein **Machbarkeitslauf**, kein
Ergebnislauf. Plane ihn auf 3–5 Tage und brich ab, wenn Gate 1 reisst.

## 6. Was am ehesten bricht (vorher entscheiden)

1. **arm64 + GB10**: `faiss-gpu`, `flash-attn`, vorkompilierte sglang-Wheels. Plan B:
   Retriever auf x86 (3090-Box) und per HTTP anbinden (`retrieval_service_url` in
   `config/search_tool_config.yaml`), Training auf den Sparks.
2. **Belegung**: die Sparks tragen den Produktions-Endpoint. Entweder Lauf in ein
   Wartungsfenster legen oder 4 Sparks abzweigen und TP2/DP2 fahren.
3. **`verl==0.5.0`** ist alt und pinnt Torch/CUDA. Nicht upgraden, ohne die
   Vendor-Kopie zu prüfen.
4. **Lizenz**: non-commercial. Für Kundenarbeit früh klären.

## 7. Was das kosten darf

Referenz: 3B, 8 GPUs, 1 Epoche, 50 Schritte pro Speicherpunkt. Auf unserem Cluster ist
Strom der Preis, nicht Miete — aber die Kisten sind tagsüber der Endpoint. Vor dem Start
ein Zeitfenster setzen und im Skript `trainer.total_epochs` klein halten (Standard 1,
nicht erhöhen, bevor Gate 3 sitzt).

## 8. Nächste Schritte

1. `pip`-Umgebung auf **einer** Spark aufsetzen und `import faiss, sglang, verl` beweisen.
2. Retriever mit dem echten Index starten, eine Suche von Hand abfragen (`/retrieve`).
3. `process_train.py` einmal komplett laufen lassen und die Parquet-Zeilen zählen.
4. Erst dann `run_sparks_iter1.sh` mit 20 Schritten als Trockenlauf.

## 9. Quellen

- Paper (HTML): https://arxiv.org/html/2601.07055 · Abstract: https://arxiv.org/abs/2601.07055
- Upstream-Code: https://github.com/facebookresearch/drzero
- Übersicht des Verfahrens (Drittquelle): https://alphaxiv.org/abs/2601.07055
- Video, das den Anlass gab (Better Stack, 90 s): https://youtube.com/shorts/iKlBCW2Orwc
- Search-R1, Basis des Such-/Rollout-Stacks: https://github.com/PeterGriffinJin/Search-R1
- VeRL v0.5.0 (Trainer): https://github.com/volcengine/verl/tree/v0.5.0
- Index-Dataset: https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index
- Korpus-Dataset: https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus
- Verwandt, gleiche Idee ohne Suchwerkzeug: R-Zero (Huang et al., 2025)

## 10. Einordnung, ehrlich

Das Paper zeigt einen sauberen Trick: Schwierigkeit als Reward, Gruppenbaseline statt
verschachteltem Sampling, und ein automatischer Lehrplan. Es ist aber auf **kurze,
verifizierbare Questions** begrenzt (EM-Match). Für lange Rechercheaufgaben, bei denen
die Antwort nicht wortgleich prüfbar ist, braucht es einen anderen Reward — genau die
Lücke, an der unsere Gutachten-Pipeline sitzen würde. Der Code ist ein Startpunkt, kein
Produkt.