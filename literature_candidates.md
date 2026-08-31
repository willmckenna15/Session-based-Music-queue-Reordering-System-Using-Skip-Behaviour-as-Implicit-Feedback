# Candidate sources — pool to select from

Not a reading list to cite wholesale. Roughly 45 candidates across 7 themes; a
lit review of this size typically lands on **30–35** of them. Selection, reading
and critical analysis are yours.

**Verification key**
- `[V]` — I confirmed the title/venue/identifier against a publisher, arXiv or author page during this session.
- `[C]` — I am confident from background knowledge but did **not** verify it here. **Check every one of these on the publisher site before it goes in your bibliography.** The Birkbeck brief warns specifically about fabricated references, and `[C]` entries are exactly where a fabrication would hide.

---

## A. Skip behaviour and music streaming sessions (target: 6–8)
The theme closest to your research question. Weakest coverage in the literature,
which is what makes the project defensible.

| # | Source | Covers | Report use |
|---|---|---|---|
| A1 | `[V]` Brost, Booth & Lamere (2019). *The Music Streaming Sessions Dataset*. WWW '19. arXiv:1901.09851 | MSSD construction, skip_1/2/3 flags, session semantics, context columns | **Must cite** — Data chapter, and the skip-label definition |
| A2 | `[V]` Meggetto, Revie, Levine & Moshfeghi (2023). *Why People Skip Music? On Predicting Music Skips using Deep Information Retrieval*. CHIIR '23. doi:10.1145/3576840.3578312 | Skips are polysemous — dislike, context change, familiarity; deep IR skip prediction | Motivating the implicit-feedback framing; limitations |
| A3 | `[C]` Meggetto et al. *On Skipping Behaviour Types in Music Streaming Sessions*. CIKM 2021 | Taxonomy of skip types; not all skips mean the same thing | Directly supports your "skip ≠ dislike" caveat |
| A4 | `[V]` Adapa (2019). *Sequential Modeling of Sessions using Recurrent Neural Networks for Skip Prediction*. arXiv:1904.10273 | WSDM Cup 7th place; per-session-length RNNs | Closest methodological precedent to SkipLSTM |
| A5 | `[V]` Chang, Lee & Lee (2019). *Sequential Skip Prediction with Few-shot in Streamed Music Contents*. arXiv:1901.08203 | Few-shot framing of the same task | Alternative formulation to contrast with yours |
| A6 | `[C]` Jeunen & Goethals (2019). Adrem Data Lab WSDM Cup solution — single RNN, **custom weighted loss**, minimal feature engineering. Code: github.com/olivierjeunen/sequential-skip-prediction | Weighted-loss precedent | **Precedent for PWTS** — worth reading before you defend the loss |
| A7 | `[C]` Béres et al. (2019). *Sequential skip prediction using deep learning and ensembles*. WSDM Cup 2019 | GBT + LSTM ensembles | Justifies keeping tree baselines alongside neural models |
| A8 | `[C]` AIcrowd (2019). *Spotify Sequential Skip Prediction Challenge* | Task definition, leaderboard, metric | Context for "what the field already tried" |
| A9 | `[C]` Hansen et al. (2020). *Contextual and Sequential User Embeddings for Large-Scale Music Recommendation*. RecSys '20 | Spotify production sequential modelling | Deployment realism |

**Gap this theme leaves open (yours to argue):** these predict *whether* a track
is skipped. None of them reorder the remaining queue using that prediction.

## B. Sequential and session-based architectures (target: 6–7)

| # | Source | Covers | Report use |
|---|---|---|---|
| B1 | `[V]` Kang & McAuley (2018). *Self-Attentive Sequential Recommendation*. ICDM '18. cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf | The architecture you adapted; self-attention over action history | **Must cite** — Methodology. State plainly what you changed (no item embeddings; `Linear(15→H)` over continuous features) |
| B2 | `[C]` Hidasi, Karatzoglou, Baltrunas & Tikk (2016). *Session-based Recommendations with RNNs*. ICLR '16. arXiv:1511.06939 | GRU4Rec — founding session-based RNN paper | **Must cite** — justifies the LSTM baseline |
| B3 | `[C]` Sun et al. (2019). *BERT4Rec*. CIKM '19. arXiv:1904.06690 | Bidirectional masked-item training | Relevant: your context/query mask is a related idea |
| B4 | `[C]` Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS | Transformer mechanics | One sentence; don't over-cite |
| B5 | `[C]` Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation 9(8) | LSTM | One sentence |
| B6 | `[C]` Quadrana, Cremonesi & Jannach (2018). *Sequence-Aware Recommender Systems*. ACM CSUR 51(4) | The standard survey; taxonomy of sequence-aware tasks | Framing the whole review |
| B7 | `[C]` Ludewig & Jannach (2018). *Evaluation of session-based recommendation algorithms*. UMUAI 28 | Simple kNN baselines beat neural models | **Pairs with F1** — supports your LogReg/ExtraTrees baselines |
| B8 | `[V]` *A Survey on Sequential Recommendation* (2024). arXiv:2412.12770 | Recent survey | Optional, for currency |

## C. Implicit feedback (target: 5–6)

| # | Source | Covers | Report use |
|---|---|---|---|
| C1 | `[C]` Hu, Koren & Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM '08 | Confidence-weighted implicit signals; absence ≠ negative | **Must cite** — foundational |
| C2 | `[C]` Rendle et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI '09 | Pairwise ranking from implicit data | Justifies ranking metrics over accuracy |
| C3 | `[V]` Jannach, Lerche & Zanker (2018). *Recommending based on Implicit Feedback*. In *Social Information Access*, LNCS 10100 | Survey of implicit signal types and their reliability | Best single framing citation for your Chapter 2 |
| C4 | `[V]` Wang et al. (2021). *Denoising Implicit Feedback for Recommendation*. WSDM '21. arXiv:2006.04153 | Implicit positives are noisy | Supports treating skips as noisy labels |
| C5 | `[C]` Yi et al. (2014). *Beyond clicks: dwell time for personalization*. RecSys '14 | Dwell time as graded feedback | **Direct precedent for PWTS's time weight** |
| C6 | `[V]` *Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation* (2022). arXiv:2205.06058 | Three-way implicit signal in sessions | Close analogue of your completed/skipped/unknown channel |
| C7 | `[C]` *Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders*. RecSys '23 (Google) | Negative feedback in production sequential recommenders | Deployment framing |
| C8 | `[V]` Wu et al. (2020). *Neural News Recommendation with Negative Feedback*. arXiv:2101.04328 | Short dwell time as negative signal | Cross-domain support |

## D. Music recommendation and audio content (target: 4–5)
This theme carries your **audio-only ablation result** (AUC 0.506 LogReg / 0.550
ExtraTrees — essentially the random floor of 0.502).

| # | Source | Covers | Report use |
|---|---|---|---|
| D1 | `[V]` van den Oord, Dieleman & Schrauwen (2013). *Deep content-based music recommendation*. NIPS '13 | CNN on spectrograms → latent factors; the case *for* audio content | **Must cite** — the position your ablation complicates |
| D2 | `[V]` Schedl, Zamani, Chen, Deldjoo & Elahi (2018). *Current challenges and visions in music recommender systems research*. IJMIR 7(2). doi:10.1007/s13735-018-0154-2 | Cold start, playlist continuation, evaluation; why music is different (short items, repeat plays, emotion) | **Must cite** — Chapter 2 opener |
| D3 | `[V]` Oramas et al. (2017). *A Deep Multimodal Approach for Cold-start Music Recommendation*. arXiv:1706.09739 | Audio alone is insufficient; multimodal | Directly supports your ablation finding |
| D4 | `[V]` Zamani, Schedl, Lamere & Chen (2019). *An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Playlist Continuation*. ACM TIST 10(5). arXiv:1810.01520 | What actually works for playlist continuation at scale | Positions queue reordering against APC |
| D5 | `[V]` Bendada et al. (2023). *A Scalable Framework for Automatic Playlist Continuation on Music Streaming Services*. SIGIR '23. doi:10.1145/3539618.3591628 | Production APC (Deezer) | Recent, deployment-facing |
| D6 | `[C]` Spotify Developer Blog (27 Nov 2024) — deprecation of the audio-features endpoint | The 10 audio features are no longer obtainable for new apps | Worth a footnote in Limitations/Future work — a real constraint on reproducing your feature set |

## E. Loss functions and ranking objectives (target: 3–5)

| # | Source | Covers | Report use |
|---|---|---|---|
| E1 | `[V]` *Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution*. AAAI 2025. arXiv:2506.15120 · ojs.aaai.org/index.php/AAAI/article/view/33450 | **DrRL itself** — Rényi-divergence DRO generalising softmax and cosine-contrastive loss | **Must cite** — this is the source of one of your six loss arms |
| E2 | `[C]` Duchi & Namkoong (2021). *Learning models with uniform performance via distributionally robust optimization*. Annals of Statistics 49(3) | The DRO theory under E1 | One paragraph, if you explain why DrRL is robust |
| E3 | `[C]` Lin et al. (2017). *Focal Loss for Dense Object Detection*. ICCV '17 | Per-example reweighting for imbalance | **Closest published ancestor of PWTS** — cite it and say what PWTS does differently |
| E4 | `[C]` Burges et al. (2005). *Learning to Rank using Gradient Descent*. ICML '05 (RankNet) | Pointwise vs pairwise vs listwise | Optional; only if you discuss why you stayed pointwise |
| E5 | `[C]` Cao et al. (2007). *Learning to Rank: From Pairwise Approach to Listwise Approach*. ICML '07 | Listwise objectives | Optional — good Future Work hook |

## F. Offline evaluation methodology (target: 5–7)
Your strongest methodological chapter. These sources let you defend the
per-session multi-point protocol rather than just describe it.

| # | Source | Covers | Report use |
|---|---|---|---|
| F1 | `[V]` Ferrari Dacrema, Cremonesi & Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*. RecSys '19 (Best Paper). arXiv:1907.06902 | 7/18 reproducible; 6 beaten by simple heuristics | **Must cite** — justifies your baselines *and* your honest reporting |
| F2 | `[V]` Ferrari Dacrema et al. (2021). *A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research*. ACM TOIS 39(2). arXiv:1911.07698 | Extended version | Pick F1 **or** F2, not both |
| F3 | `[V]` Castells & Moffat (2022). *Offline recommender system evaluation: Challenges and new directions*. AI Magazine 43(2). doi:10.1002/aaai.12051 | What offline metrics can and cannot establish | Framing the Evaluation chapter |
| F4 | `[C]` Järvelin & Kekäläinen (2002). *Cumulated gain-based evaluation of IR techniques*. ACM TOIS 20(4) | NDCG | **Must cite** — your headline metric |
| F5 | `[C]` Krichene & Rendle (2020). *On Sampled Metrics for Item Recommendation*. KDD '20 | Sampled metrics are inconsistent with full ranking | Justifies scoring the **full** remaining queue |
| F6 | `[V]` *Improving Methodological Standards in Recommender Systems Offline Evaluation*. ACM TORS. doi:10.1145/3800587 | Protocol standards | Recent; supports design choices |
| F7 | `[V]` *On the Reliability of Sampling Strategies in Offline Recommender Evaluation*. RecSys '25. doi:10.1145/3705328.3748086 | Sampling reliability | Optional, currency |
| F8 | `[C]` Smucker, Allan & Carterette (2007). *A comparison of statistical significance tests for IR evaluation*. CIKM '07 | Bootstrap and randomisation tests beat t-tests for IR | **Cite this for your paired bootstrap** on `eval_per_session.npz` |
| F9 | `[C]` Cañamares & Castells (2020). *On Target Item Sampling in Offline Recommender System Evaluation*. RecSys '20 | Candidate-set construction changes conclusions | Optional |

## G. Variance, tuning and honest reporting (target: 2–3)
These are what let you write "the differences are inside seed noise" as a
*finding* rather than an apology — which is the right way to report the PWTS
ablation.

| # | Source | Covers | Report use |
|---|---|---|---|
| G1 | `[C]` Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys '21 | Seed variance often exceeds method differences | **Directly supports your 5-seed protocol and 2×sd rule** |
| G2 | `[C]` Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. AAAI '18 | Seeds, reporting practice | Alternative to G1; famous and quotable |
| G3 | `[C]` Shehzad & Jannach (2023). *Everyone's a Winner! On Hyperparameter Tuning of Recommendation Models*. RecSys '23 | Unequal tuning budgets manufacture wins | Justifies your equal 1,080-config grid across all arms |

---

## Suggested allocation

| Theme | Take | Why |
|---|---|---|
| A Skip behaviour | 6–8 | Your specific gap lives here |
| B Architectures | 6–7 | Methodology needs grounding |
| C Implicit feedback | 5–6 | The conceptual core |
| D Music/audio | 4–5 | Carries the ablation result |
| E Losses | 3–5 | You ran six loss arms; two are from papers |
| F Evaluation | 5–7 | Your strongest chapter |
| G Variance | 2–3 | Makes the null results reportable |
| **Total** | **31–41** | Aim ~33 |

## Before you cite anything

1. Every `[C]` entry: open the DOI or publisher page and confirm authors, year, venue, page range.
2. Every `[V]` entry: I confirmed it exists with those details — still open it, because I verified the *record*, not that it says what the "Covers" column claims.
3. Where I wrote "must cite", that is because the claim in your report depends on it, not because the source is famous.
4. A2/A3, F1/F2 and G1/G2 are near-substitutes. Take one of each pair unless you are contrasting them.
