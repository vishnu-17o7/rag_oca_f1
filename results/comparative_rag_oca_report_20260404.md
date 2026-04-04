# Meta Heuristic Optimization Techniques - 19MAM83

## Assignment III: Comparative Meta-Optimization of RAG Pipelines

- Assignment Date: 17.03.2026
- Due Date: 04.04.2026
- Total Marks: 20
- Selected Meta-heuristic: Overclocking Algorithm (OCA), a population-based optimizer used as the search engine for RAG hyperparameters.

## 1. Objective and Task Mapping
This report evaluates a meta-heuristic optimizer on a motorsport RAG system by searching for the best parameter vector [chunk_size, chunk_overlap, temperature, top_k] under a strict 50-call budget with a local Ollama model.

Parameter mapping used throughout the report:
- G1 = chunk_size
- G2 = chunk_overlap
- G3 = temperature
- G4 = top_k

## 2. Implementation Summary (Rubric: Pipeline + Algorithm + Fitness)
- RAG integration: src/rag_pipeline.py (PDF loading, chunking, Chroma retrieval, Ollama query chain).
- Meta-heuristic logic: research/oca.py (population update, velocity, exploration terms).
- Fitness evaluator: src/fitness.py (semantic similarity via SentenceTransformers cosine similarity against gold-standard answers).
- Benchmark orchestration and trace capture: api/routes_benchmark.py (iteration history, state snapshots, budgeted run execution).
- Gold-standard validation set: src/gold_standard.json (8 Q&A pairs, within required 5-10 range).
- Penalty rule implemented: if chunk_overlap >= chunk_size then penalized fitness (constraint handling).
- Caching implemented: repeated parameter configurations reuse stored evaluations (cache_hits tracked).

## 3. Evidence Artifacts Used
- Primary optimization trace: benchmark_20260404_154603.json (50 evaluations).
- Final recomputed metrics from same trace: benchmark_20260404_154603_metrics_final.json.
- Subdomain controlled sweep for domain impact: subdomain_sweep_20260404_154603.json.
- Visual artifacts in this report are generated from these exact files.

## 4. UI Screenshots
### 4.1 Benchmark Dashboard During Run
![Benchmark dashboard in progress](ui_bench_inprogress.png)

### 4.2 Benchmark Dashboard After Completion
![Benchmark dashboard complete](ui_bench_complete.png)

### 4.3 Chat/RAG Interface
![Chat interface](ui_chat.png)

## 5. New Plots (Derived from Raw JSON)
### 5.1 Convergence (Existing Canonical Plot)
![Convergence plot](convergence_plot_20260404_154603.png)

### 5.2 Parameter Evolution (G1/G2/G3/G4 vs Iteration)
![Parameter evolution](plot_parameter_evolution_20260404_154603.png)

### 5.3 Plateau and Breakthrough Timeline
![Plateau and breakthroughs](plot_plateau_breakthroughs_20260404_154603.png)

### 5.4 Subdomain Chunk Sensitivity (Controlled Sweep)
![Subdomain sensitivity](plot_subdomain_chunk_sensitivity_20260404_154603.png)

### 5.5 Cache-Hit and Penalty Timeline
![Cache and penalty timeline](plot_cache_penalty_timeline_20260404_154603.png)

## 6. Required Comparison Table
| Iteration | Best Fitness | Parameters [G1, G2, G3, G4] |
|---:|---:|---|
| 1 | 0.5863 | [615, 200, 1.00, 10] |
| 2 | 0.6388 | [848, 200, 1.00, 10] |
| 3 | 0.6741 | [770, 200, 1.00, 10] |
| 4 | 0.6741 | [892, 200, 1.00, 10] |
| 5 | 0.6741 | [673, 200, 1.00, 10] |
| 6 | 0.6741 | [394, 200, 1.00, 10] |
| 7 | 0.6741 | [780, 200, 1.00, 10] |
| 8 | 0.6741 | [357, 102, 1.00, 10] |
| 9 | 0.6804 | [960, 194, 1.00, 8] |
| 10 | 0.6804 | [108, 200, 1.00, 10] |
| 11 | 0.6804 | [1000, 200, 1.00, 10] |
| 12 | 0.6804 | [869, 200, 1.00, 10] |
| 13 | 0.7404 | [954, 200, 0.00, 1] |
| 14 | 0.7404 | [964, 200, 1.00, 10] |
| 15 | 0.7404 | [688, 147, 1.00, 1] |
| 16 | 0.7404 | [1000, 200, 1.00, 10] |
| 17 | 0.7404 | [780, 200, 1.00, 10] |
| 18 | 0.7404 | [516, 100, 1.00, 10] |
| 19 | 0.7404 | [997, 190, 1.00, 10] |
| 20 | 0.7404 | [1000, 200, 1.00, 10] |
| 21 | 0.7404 | [1000, 200, 0.00, 1] |
| 22 | 0.7404 | [954, 200, 0.00, 1] |
| 23 | 0.7404 | [1000, 200, 0.00, 1] |
| 24 | 0.7404 | [1000, 200, 0.00, 1] |
| 25 | 0.7404 | [706, 200, 1.00, 1] |
| 26 | 0.7404 | [1000, 200, 0.00, 10] |
| 27 | 0.7404 | [889, 200, 0.00, 10] |
| 28 | 0.7404 | [1000, 200, 1.00, 1] |
| 29 | 0.7404 | [936, 200, 1.00, 1] |
| 30 | 0.7404 | [1000, 200, 0.00, 1] |
| 31 | 0.7404 | [1000, 200, 0.00, 1] |
| 32 | 0.7404 | [988, 200, 0.00, 1] |
| 33 | 0.7452 | [978, 200, 0.00, 1] |
| 34 | 0.8526 | [976, 200, 1.00, 1] |
| 35 | 0.8526 | [991, 200, 1.00, 1] |
| 36 | 0.8526 | [1000, 200, 0.00, 10] |
| 37 | 0.8526 | [1000, 200, 0.00, 10] |
| 38 | 0.8526 | [888, 200, 1.00, 1] |
| 39 | 0.8526 | [928, 200, 0.00, 1] |
| 40 | 0.8526 | [1000, 200, 0.00, 1] |
| 41 | 0.8526 | [1000, 200, 1.00, 1] |
| 42 | 0.8526 | [994, 200, 0.00, 1] |
| 43 | 0.8526 | [964, 200, 0.00, 1] |
| 44 | 0.8526 | [968, 200, 1.00, 1] |
| 45 | 0.8526 | [931, 200, 1.00, 1] |
| 46 | 0.8526 | [1000, 200, 1.00, 10] |
| 47 | 0.8526 | [894, 200, 1.00, 1] |
| 48 | 0.8526 | [885, 185, 1.00, 1] |
| 49 | 0.8526 | [975, 200, 1.00, 1] |
| 50 | 0.8526 | [1000, 200, 0.00, 10] |

## 7. Comparative Analysis (Required Point 5)
### 5A. Landscape Analysis
- Total evaluations: 50
- Invalid-config frequency: 1/50 = 2.00%
- Unique-configuration rate: 36/50 = 72.00%
- Cache-hit rate: 14/50 = 28.00%
- Mean absolute fitness delta: 0.2908
- Maximum absolute fitness delta: 0.7404
- Breakthrough iterations (best-so-far improvements): [1, 2, 3, 9, 13, 33, 34]
- Longest plateau length: 20 iterations
Direct answer for 5A: The search landscape is rugged. The run shows long plateaus with sparse breakthroughs, which is consistent with local-optima behavior before later jumps in best fitness.

### 5B. Domain Impact (Sporting, Technical, Financial/General)
- Controlled experiment setup for fair comparison: G3 fixed at 1.0 and G4 fixed at 1; only G1/G2 were swept.

| Subdomain | Best G1 | Best G2 | Best Avg Fitness |
|---|---:|---:|---:|
| Sporting | 900 | 200 | 0.6355 |
| Technical | 300 | 50 | 0.6338 |
| Financial/General | 700 | 150 | 0.6000 |

| Pairwise Comparison | Delta G1 (%) | Delta G2 (%) | Delta Best Fitness (%) |
|---|---:|---:|---:|
| Sporting vs Technical | 200.00% | 300.00% | 0.26% |
| Sporting vs Financial/General | 28.57% | 33.33% | 5.90% |
| Technical vs Financial/General | -57.14% | -66.67% | 5.63% |

Scope limit for 5B: gold-standard questions are mostly sporting/technical, so financial/general interpretation is transfer-oriented under this evaluation set.
Direct answer for 5B: Dataset domain materially changes optimal chunking. Sporting favored larger chunks (900/200), technical favored smaller chunks (300/50), and financial/general favored mid-range chunks (700/150). Relative overlap shift reached 300% between sporting and technical settings.

### 5C. Efficiency
- First iteration where best fitness exceeded 0.8: iteration 34.
- Best achieved: 0.8526 at iteration 34.
- Budget efficiency to threshold: 34/50 evaluations (68.00% of budget).
Direct answer for 5C: The algorithm reached the >0.8 target at iteration 34.

## 8. Inference on Algorithm Performance
- OCA successfully optimized the RAG system to a high score (0.8526) within the 50-call budget.
- The optimization path demonstrates rugged behavior, so stopping too early can miss late breakthroughs.
- Domain-specific chunk tuning is necessary; one global chunking policy does not maximize performance across all motorsport document subsets.
- Caching improved compute efficiency by avoiding redundant evaluations while preserving result consistency.

## 9. Reproducibility Notes
- Run benchmark with budget 50 through the benchmark API/UI and ensure results are persisted as JSON.
- Regenerate analysis plots from benchmark_20260404_154603.json and subdomain_sweep_20260404_154603.json.
- Convert this markdown report to PDF using pandoc after ensuring a PDF engine is installed.
