# Report outline — Session-Based Music Queue Reordering

Structure follows the briefing (slide 9). Word budget targets ~10,000 of the
12,000 maximum. Cover page, contents, references and appendices are excluded
from the count.

**Marks are weighted 20 / 30 / 30 / 20** across Specification & Design,
Implementation, Testing & Evaluation & Reflection, and Presentation. Chapters
5–8 therefore carry 80% of the marks — budget words accordingly.

---

## Cover page + Abstract (~250 words)

- Title, name, programme, School of Computing and Mathematical Sciences, date,
  **word count** (required — omitting it risks the Presentation penalty)
- Abstract 200–300 words: problem, approach, main findings. No technical detail,
  no citations.
- The findings worth naming: audio features at chance alone; behavioural features
  carrying the signal; PWTS best on both architectures; LSTM and SASRec
  statistically indistinguishable.

## 1. Introduction, Aims and Objectives (~1,000 words)

**Chapters 1 and 2 are merged.** The aim of this project follows directly from the
problem statement, and separating them across a chapter boundary forces the
problem to be restated twice. Merging also removes the weakest section of a
typical dissertation — a 400-word chapter that says little the introduction has
not already implied.

**Numbering consequence, decide before writing.** Your drafted chapters are
numbered 4 to 7. Merging 1 and 2 leaves chapter 3 for the literature review and
nothing at 2 — so either renumber everything down by one (touching four written
chapters and every cross-reference), or keep the literature review at 3 and let
chapter 2 be something else. The cheapest resolution is to make **2 the
literature review** and **3 the methodology**, which the current chapter 4
partially is.

**Write this chapter last.** Every figure in it must match the final results
table, and the framing only settles once the results are known.

### 1.1 The problem (~180 words)

- Open on the concrete situation, not the field. A listener puts an album on
  shuffle and skips the first track within thirty seconds, then the second.
  **Do not open with "with the rise of streaming services" — the marker has read
  that opening forty times.**
- **Get the contrast right, because the obvious version of it is wrong.** Do not
  write that the field asks *what to play next* while this project asks *in what
  order* — deciding what plays next is exactly what this system does. The real
  distinction is the **candidate set and what changes as a result**:
  - recommendation and autoplay select from a catalogue of millions, and the
    *contents* of the session change;
  - queue reordering selects from a bounded set the listener already holds —
    nothing is added, nothing removed, and **only the sequence changes**.
- That second property is the claim. The same tracks are heard either way.
- It follows that this is a **ranking** problem over a bounded, given candidate
  set, not a retrieval problem over an open one — which justifies the architecture
  in ch. 5: no catalogue-scale cold start, and **no item embedding table**.
- **Say "bounded and given", not "chosen by the user".** The listener chose a
  source — a playlist, an album, an artist — not the individual tracks, and under
  shuffle not the order either. The model needs a queue, not its provenance.
- **Shuffle is the common case and the one with most to gain**, which is a
  stronger argument than user choice: 62.7% of plays occur in shuffle mode, and
  those are skipped at 0.438 against 0.287 unshuffled. A deliberately sequenced
  queue may have an order worth preserving; a shuffled one has none.
- Keep the *second* contrast — that the skip-prediction literature predicts
  without acting on the prediction — for 1.6. Conflating the two contrasts into
  one sentence is what produces the wrong version above.

### 1.2 Skip behaviour as the signal (~140 words)

- Explicit feedback — likes, saves, ratings — is sparse. Most listening produces
  no explicit signal at all.
- A skip is abundant, automatic and costless to collect: every session generates
  them, and the user does nothing extra.
- But it is **ambiguous**. A skip can mean dislike, wrong mood, heard-it-already,
  or an interruption with nothing to do with the track. One sentence here; it
  returns as a limitation in ch. 8. Do not bury it — a marker who spots the
  ambiguity before you name it reads the whole report as naive.
- State your operational definition of `skipped` in plain words (one sentence);
  defer the exact rule to ch. 4.
- **Name the random floor here.** Because most tracks are not skipped, random
  ordering already scores NDCG@5 ≈ 0.688. Giving the floor up front reframes
  every later number and stops a marker reading a headline 0.79 as impressive.
  The single most valuable sentence in the chapter for the Evaluation marks.

### 1.3 The deployability constraint (~140 words)

*The project's distinctive move. Give it its own paragraph and claim it.*

- The rule in one sentence: **a feature may be used only if its value is knowable
  for a track that has not yet been played.**
- `actively_selected` was excluded by it, and `reason_start` — from which it is
  derived — was never used as an input. Both describe how a track's playback
  *began*, so both exist only after the fact.
- Why it matters: a model using them scores well offline and cannot be deployed.
  Much of the skip-prediction literature, the WSDM Cup 2019 framing included,
  predicts skips for tracks *already playing* — a different task with a different
  information set.
- **State the cost honestly.** The constraint lowers the achievable ceiling. That
  is the point: you traded offline score for a system that could run, and that
  trade is a design contribution rather than something to apologise for.
- One sentence on what it does *not* forbid: skip outcomes for tracks already
  played earlier in the same session are legitimately available, and both neural
  models use them.

### 1.4 Aim and objectives (~220 words)

*This is the merged material, and it belongs here because it follows from 1.1–1.3
rather than restating them.*

- **The aim in one sentence:** to rank the unplayed tracks in a listening queue by
  predicted skip probability, using only information available at the moment of
  reordering, and to establish whether sequential models improve that ranking over
  simpler alternatives.
- **Objectives as work packages**, each with an outcome that can be checked:
  1. Collect and process volunteer listening histories into session-structured
     data with a leak-free chronological split — *ch. 4*
  2. Define an evaluation protocol appropriate to within-session ranking, with a
     stated random floor — *ch. 7*
  3. Implement four model families and three loss functions under a common
     interface and an equal tuning protocol — *ch. 5, 6*
  4. Establish whether sequence modelling improves ranking over flat baselines —
     *ch. 8.2–8.4*
  5. Establish whether audio content features contribute beyond behavioural ones —
     *ch. 8.5*
- **State how success is measured**, explicitly: per-session NDCG@5 as the primary
  metric, AUC secondary, both against the random-ordering floor, with differences
  below the seed-noise threshold reported as unresolved.
- **Phrase each objective so it maps to a results subsection.** Markers check
  objectives against outcomes, and an objective with no corresponding result reads
  as abandoned. Note that objectives 4 and 5 are questions, not targets — they are
  satisfied by a credible answer, including a negative one.

### 1.5 What was built and what was found (~250 words)

- Four model families in one sentence: logistic regression, extremely randomised
  trees, an LSTM, and a SASRec adaptation without item embeddings. Three losses
  across the two neural architectures — BCE, PWTS, DrRL — giving six trained arms
  plus two baselines and a random floor.
- Scale in one clause, because it evidences the Implementation marks: 1,080 tuning
  configurations, five seeds per arm, run as SLURM job arrays.
- Then the three findings, one sentence each, with numbers:
  - **Audio features alone are at chance.** Session AUC 0.507 and 0.504 against a
    floor of 0.502 — and ExtraTrees on audio alone scores NDCG@5 0.694,
    *fractionally below* the 0.696 floor. A cleaner result than "slightly above
    chance".
  - **Behavioural features carry the signal**, reaching 0.592 / 0.605 AUC, and
    adding the ten audio features on top changes nothing (0.594 / 0.603) — inside
    seed noise, and the two models disagree in *sign*, which is the cleanest
    evidence that the contribution is zero rather than small.
  - **Architecture matters less than expected.** PWTS is the best loss on both
    architectures; the gap between them is small relative to seed variance.
- One sentence on why the null results are the report's strongest material: they
  contradict a standing assumption, and they come from an ablation designed in
  advance rather than a failure explained afterwards.
- **Do not oversell.**

### 1.6 Comparable systems (~90 words — detail in ch. 2/3)

- **Spotify smart shuffle / autoplay** — the commercial comparator. Reorders and
  extends a queue, but is proprietary and its objective unpublished.
- **WSDM Cup 2019 Spotify Sequential Skip Prediction** — the academic comparator.
  Same signal, different task: predict skips within a session as it unfolds, with
  no reordering and no deployability constraint.
- **Automatic playlist continuation** (RecSys Challenge 2018) — the adjacent task.
  *Extends* a playlist; this project *reorders* what the user already chose.
- The gap in one sentence: none convert a skip prediction into a queue order and
  then measure the ranking quality of that order.

### 1.7 Roadmap (~80 words)

- One sentence per chapter, written as prose. **Not a bulleted list** — a bulleted
  roadmap reads as filler and costs Presentation marks.
- Factual and specific: "Chapter 4 describes the data pipeline and the
  chronological per-user split; Chapter 5 sets out the design decisions that
  follow from the deployability constraint; …"

### Before you write this chapter

- Every number must match ch. 8 exactly, including rounding. If `evaluate.py` is
  re-run, this chapter changes with it — another reason to write it last.
- Keep citations to two or three. The introduction is not the literature review.
- Do not state a hypothesis you do not test. If you frame the project as testing
  whether audio features help, the ablation must be presented as the test of it,
  not as a side experiment.
- Decide the chapter numbering before writing any cross-reference.

## 3. Background / Literature review (~1,600 words)

*Content and reference selection are yours. Structural notes only:*

- The brief asks for **critical analysis**, not summary — identify gaps and use
  them to justify your design decisions
- Address any marker feedback on the proposal
- Each design choice in ch. 5 should be traceable to something argued here
- Verify every reference against the publisher before it goes in

**Verification key.** `[V]` — title, venue and identifier confirmed against a
publisher, arXiv or author page. `[C]` — plausible from background knowledge but
**not** confirmed here; open the DOI and check authors, year, venue and pages
before it enters your bibliography. The brief warns specifically about fabricated
references, and `[C]` entries are exactly where a fabrication would hide.

### 3.1 Traceability — every design decision to a source

*Read each row as: the decision is the claim, the source is what makes it more
than a preference. A row you cannot defend from its source is a row an examiner
can ask you to justify from first principles. Anything you cite here must be
**analysed** in this chapter — a citation that appears for the first time in
ch. 5 has not been reviewed, it has been name-dropped.*

| § | Design decision | Sources to analyse |
|---|---|---|
| 4.3 | 30 minutes of inactivity ends a session | Catledge & Pitkow (1995) `[V]`; Brost et al. (2019) `[V]` |
| 4.4 | Ten audio features joined from a bulk corpus | Spotify (2024) endpoint deprecation `[C]` |
| 4.5 | `fwdbtn` / `clickrow` define the skip label | Brost et al. (2019) `[V]`; Meggetto et al. (2021) `[C]` |
| 4.6 | Historical skip rates as confidence-style features | Hu, Koren & Volinsky (2008) `[C]` |
| 4.7 | Validity filtering discards 61.5% of sessions | Ludewig & Jannach (2018) `[C]`; Cañamares & Castells (2020) `[C]` |
| 4.8 | Per-session z-scoring; chronological per-user split | Meng et al. (2020) `[C]` |
| 5.1 | Only features knowable for an *unplayed* track | Zinkevich (2017) `[C]` |
| 5.2–5.3 | `actively_selected` excluded despite predictive value | Xu et al. (2020) `[V]`; Vapnik & Vashist (2009) `[C]` |
| 5.4 | Strong non-neural baselines are mandatory | Ferrari Dacrema et al. (2019) `[V]`; Ludewig & Jannach (2018) `[C]` |
| 5.4 | Equal tuning budget across every arm | Shehzad & Jannach (2023) `[C]` |
| 5.5 | Baselines score each track independently | Ludewig & Jannach (2018) `[C]` |
| 5.6 | An LSTM to isolate recurrence | Hochreiter & Schmidhuber (1997) `[C]`; Hidasi et al. (2016) `[C]`; Adapa (2019) `[V]` |
| 5.7 | SASRec adapted from a public implementation | Kang & McAuley (2018) `[V]`; Huang (2020) `[V]` *(code)*; Vaswani et al. (2017) `[C]` |
| 5.8 | Three-state status channel, summed, *unknown* zero-initialised | Kim et al. (2022, arXiv:2205.06058) `[V]`; Wu et al. (2020) `[V]`; Sun et al. (2019) `[C]` |
| 5.9 | Context/query mask stops queue tracks attending each other | Kang & McAuley (2018) `[V]`; Sun et al. (2019) `[C]` |
| 5.10 | PWTS **position** weight | Lin et al. (2017) `[C]`; Jeunen & Goethals (2019) `[C]` |
| 5.10 | PWTS **time** weight | Yi et al. (2014) `[C]`; Wu et al. (2020) `[V]` |
| 5.10 | PWTS **historical** weight | Hu, Koren & Volinsky (2008) `[C]` |
| 5.10 | DrRL as a third objective | Zhang et al. (2025) `[V]`; Duchi & Namkoong (2021) `[C]` |
| 5.10 | Pointwise rather than pairwise or listwise | Burges et al. (2005) `[C]`; Cao et al. (2007) `[C]` |
| 7.1 | Per-session rather than pooled scoring | Castells & Moffat (2022) `[V]`; Krichene & Rendle (2020) `[C]` |
| 7.2 | Multiple split points per session | Meng et al. (2020) `[C]` |
| 7.3 | NDCG@5 as the selection metric | Järvelin & Kekäläinen (2002) `[C]` |
| 7.3 | AUC via the Mann–Whitney identity (`fast_auc`) | Hanley & McNeil (1982) `[V]` |
| 7.3 | P@5 reported alongside | Manning, Raghavan & Schütze (2008) `[V]` |
| 7.3 | The full remaining queue is scored, not a sample | Krichene & Rendle (2020) `[C]` |
| 7.4 | Five seeds; differences judged against 2 × sd | Bouthillier et al. (2021) `[C]`; Henderson et al. (2018) `[C]` |
| 7.5 | Paired bootstrap over per-session scores | Smucker, Allan & Carterette (2007) `[C]` |
| 8.5 | Audio-only ablation sits at the random floor | van den Oord et al. (2013) `[V]`; Oramas et al. (2017) `[V]`; Schedl et al. (2018) `[V]` |
| 8.6 | The PWTS null result reported as a finding | Bouthillier et al. (2021) `[C]`; Ferrari Dacrema et al. (2019) `[V]` |
| 8.7 | A skip is not a dislike | Meggetto et al. (2023) `[V]`; Seshadri et al. (2024) `[V]` |
| 1.6 / 8.7 | Reordering is already deployed at scale | Moor et al. (2023) `[V]`; Hansen et al. (2020) `[C]`; Bendada et al. (2023) `[V]` |
| 4.10 / 6.1 | Tooling | Paszke et al. (2019) `[C]`; Pedregosa et al. (2011) `[C]` |

**Three rows carry unusual weight.** *5.7* is where you declare what you
inherited — an examiner who finds `PointWiseFeedForward` unattributed will
discount everything else you claim to have built. *5.1* is the constraint the
whole design turns on, and it currently rests on one grey-literature source, so
Xu et al. gives it a peer-reviewed anchor. *8.7* is where the polysemy of skips
turns a limitation into an argued position rather than an admission.

### 3.2 The candidate pool

*Roughly 50 candidates across seven themes; a review of this length lands on
**30–35**. Take one of each near-substitute pair: Meggetto 2023/2021, Ferrari
Dacrema 2019/2021, Bouthillier/Henderson.*

| Theme | Take | Core sources |
|---|---|---|
| **A. Skip behaviour & streaming sessions** | 6–8 | Brost, Booth & Lamere (2019) `[V]`; Meggetto et al. (2023) `[V]`, (2021) `[C]`; Adapa (2019) `[V]`; Chang, Lee & Lee (2019) `[V]`; Jeunen & Goethals (2019) `[C]`; Béres et al. (2019) `[C]`; Hansen et al. (2020) `[C]`; Seshadri, Shashaani & Knees (2024) `[V]`; Moor et al. (2023) `[V]` |
| **B. Sequential architectures** | 6–7 | Kang & McAuley (2018) `[V]`; Hidasi et al. (2016) `[C]`; Sun et al. (2019) `[C]`; Vaswani et al. (2017) `[C]`; Hochreiter & Schmidhuber (1997) `[C]`; Quadrana et al. (2018) `[C]`; Ludewig & Jannach (2018) `[C]` |
| **C. Implicit feedback** | 5–6 | Hu, Koren & Volinsky (2008) `[C]`; Rendle et al. (2009) `[C]`; Jannach, Lerche & Zanker (2018) `[V]`; Wang et al. (2021) `[V]`; Yi et al. (2014) `[C]`; Kim et al. (2022) `[V]`; Wu et al. (2020) `[V]` |
| **D. Music & audio content** | 4–5 | van den Oord et al. (2013) `[V]`; Schedl et al. (2018) `[V]`; Oramas et al. (2017) `[V]`; Zamani et al. (2019) `[V]`; Bendada et al. (2023) `[V]`; Spotify (2024) `[C]` |
| **E. Losses & ranking objectives** | 3–5 | Zhang et al. (2025) `[V]`; Duchi & Namkoong (2021) `[C]`; Lin et al. (2017) `[C]`; Burges et al. (2005) `[C]`; Cao et al. (2007) `[C]` |
| **F. Offline evaluation** | 5–7 | Ferrari Dacrema et al. (2019) `[V]`; Castells & Moffat (2022) `[V]`; Järvelin & Kekäläinen (2002) `[C]`; Krichene & Rendle (2020) `[C]`; Smucker, Allan & Carterette (2007) `[C]`; Meng et al. (2020) `[C]`; Cañamares & Castells (2020) `[C]`; Hanley & McNeil (1982) `[V]`; Manning et al. (2008) `[V]` |
| **G. Variance & honest reporting** | 2–3 | Bouthillier et al. (2021) `[C]`; Henderson et al. (2018) `[C]`; Shehzad & Jannach (2023) `[C]` |
| **H. Deployability & train/serve skew** | 2–3 | Zinkevich (2017) `[C]`; Xu et al. (2020) `[V]`; Vapnik & Vashist (2009) `[C]` |

**The gap you are claiming.** Theme A predicts *whether* a track is skipped;
none of it reorders the remaining queue using that prediction. State the gap in
those terms and every design decision above becomes a response to it. Be careful
with Moor et al. (2023) — a 7M-user randomised trial at Spotify that increased
completion and reduced skips. It does not close your gap, but it does refute any
claim that no service attempts this, so cite it before a marker raises it.

### 3.3 Full details for the sources not already in your bibliography

- Catledge, C.L. and Pitkow, J.E. (1995) 'Characterizing browsing strategies in the World-Wide Web', *Computer Networks and ISDN Systems*, 27(6), pp. 1065–1073. `[V]` — the 30-minute threshold's origin; note it is a 1995 web-browsing study, not a music one.
- Kang, W.-C. and McAuley, J. (2018) 'Self-Attentive Sequential Recommendation', *ICDM '18*. IEEE, pp. 197–206. doi:10.1109/ICDM.2018.00035 `[V]`
- Huang, Z. (2020) *SASRec.pytorch*. GitHub: `pmixer/SASRec.pytorch`, Apache-2.0. `[V]` — the implementation you adapted; §4 of the licence requires a copy of the licence and a notice of changes.
- Hanley, J.A. and McNeil, B.J. (1982) 'The meaning and use of the area under a receiver operating characteristic (ROC) curve', *Radiology*, 143(1), pp. 29–36. `[V]`
- Xu, C. et al. (2020) 'Privileged Features Distillation at Taobao Recommendations', *KDD '20*, pp. 2590–2598. doi:10.1145/3394486.3403309 `[V]`
- Zinkevich, M. (2017) *Rules of Machine Learning: Best Practices for ML Engineering*. Google. `[C]` — grey literature; pair it with Xu et al.
- Vapnik, V. and Vashist, A. (2009) 'A new learning paradigm: Learning using privileged information', *Neural Networks*, 22(5–6), pp. 544–557. `[C]`
- Meng, Z. et al. (2020) 'Exploring Data Splitting Strategies for the Evaluation of Recommendation Models', *RecSys '20*. `[C]`
- Zhang, S., Chen, J., Li, C., Zhou, S., Shi, Q., Feng, Y., Chen, C. and Wang, C. (2025) 'Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution', *AAAI*, 39(12), pp. 13286–13294. `[V]`
- Moor, D., Yuan, Y., Mehrotra, R., Dai, Z. and Lalmas, M. (2023) 'Exploiting Sequential Music Preferences via Optimisation-Based Sequencing', *CIKM '23*. `[V]`
- Seshadri, V., Shashaani, S. and Knees, P. (2024) *RecSys '24*. doi:10.1145/3640457.3688188 `[V]`
- Manning, C.D., Raghavan, P. and Schütze, H. (2008) *Introduction to Information Retrieval*. Cambridge University Press. `[V]`

**Do not cite Campos et al. (2018), *Skip RNN* (arXiv:1708.06834)** unless you
mean to. It is about skipping RNN *state updates* for efficiency, nothing to do
with music. The name collides with your `SkipLSTM` and a reader may assume a
relationship that does not exist — one sentence distinguishing them is cheap
insurance.

## 4. Methodology and Methods (~1,000 words)

The brief is emphatic that every choice must be *justified*, not just described.
The test for each paragraph below: does it say why, not only what? Chapter 5
covers model design — keep this chapter to data and protocol.

**Pipeline attrition — state this before the split table.** It is the honest
account of how 5.2M raw plays became 3.5M modelled rows, and a marker will look
for it:

| stage | rows | sessions | rows kept | sessions kept |
|---|---|---|---|---|
| raw exports | 5,242,623 | — | 100% | — |
| session construction | 5,242,623 | 297,141 | 100% | 100% |
| audio-feature join | 4,714,208 | 281,365 | 89.9% | 94.7% |
| validity filter | 3,546,781 | 108,417 | **67.7%** | **36.5%** |

Source: `Models/pipeline_attrition.csv`.

**Headline figures to state once, early, and then reuse:**

| | users | sessions | rows | skip rate | median length | date range |
|---|---|---|---|---|---|---|
| training | 33 | 75,875 | 2,537,905 | 0.381 | 22 | 2011-04-01 → 2026-03-12 |
| validation | 33 | 16,262 | 510,661 | 0.346 | 21 | 2020-11-04 → 2026-04-11 |
| testing | 33 | 16,280 | 498,215 | 0.356 | 20 | 2022-09-09 → 2026-07-19 |
| **total** | **33** | **108,417** | **3,546,781** | | 7 min / 1,148 max | |

### 4.1 Data provenance and ethics (~170 words)

- 33 volunteers supplied Spotify extended streaming history exports under GDPR
  data-portability requests. State the recruitment route and the consent process.
- Ethics approval — reference number and approving body. **Do not omit this**; it
  is the one item in the chapter that cannot be fixed later.
- Pseudonymisation: `user_id` is a hash, `session_id` is `<user_id>_<n>`. Say what
  identifying fields were dropped and at which pipeline stage.
- Storage and retention: where the raw exports live, who can access them, what
  happens to them after submission. Note that `RAW Data/` is git-ignored and the
  published repository contains no listening data.
- One sentence on the sample's limits, stated now rather than in ch. 8: 33
  self-selecting volunteers, skewed toward the researcher's own network. No claim
  to population representativeness follows from this data.

### 4.2 Pipeline overview (~110 words)

- `data_processing_main.py` orchestrates five stages in order:
  `Json2csv` → `session_compiler` → `audio_features_clean` →
  `feature_and_filter` → `dataset_splitter`.
- **A dataflow diagram belongs here** and is cheap marks — boxes for the five
  stages, arrows annotated with what each one drops. Markers reward a reader who
  can see the shape of the pipeline without reading code.
- One sentence on why the ordering matters, taken from the code comment: sessions
  are *constructed* before the audio join but *filtered* after it, because the
  join drops rows and filtering first would leave sessions that no longer satisfy
  the validity criteria and features computed over tracks no longer present.
  That is a real design decision with a reason — say so.

### 4.3 Session construction (~110 words)

- Definition: a new session starts after a **30-minute gap** in a user's listening,
  or at a user boundary. Justify the threshold — it is the conventional choice in
  session-based recommendation, and cite rather than assert.
- `session_id = "<user_id>_<n>"`, unique per user and readable downstream.
- Note what this definition cannot capture: a session boundary is inferred from
  inactivity, so a user who pauses for 35 minutes mid-album is split into two
  sessions, and one who leaves audio running is not split at all.

### 4.4 Audio feature join (~90 words)

- Ten audio descriptors joined on `spotify_track_uri` (with the `spotify:track:`
  prefix stripped) against an external audio-features corpus.
- **`key` was deliberately excluded** — it is categorical with 12 unordered levels
  and would need one-hot encoding for no expected benefit. State this; an omission
  you explain reads differently from one you did not notice.
- **This stage drops 528,415 rows (10.1%) and 15,776 sessions, but not all of it
  is the join.** Separate the two, because they are different kinds of loss:
  a full-row `drop_duplicates()` removes 22,201 exact duplicate plays (4.2% of
  the loss), and the inner join removes the remaining 506,214 (95.8%). Those are
  tracks with no audio-feature match in the corpus. Say what kind of track goes missing —
  local files, podcasts, regional or very recent releases — because a
  non-random 10% loss is a sampling decision, not a technicality. It is *not* the
  pipeline's largest cut; the validity filter is (→ 4.7).
- **State the corpus's provenance and cite it.** The features come from a 11 GB
  third-party parquet dump in `RAW Data/audio_features/`, not from the Spotify API.
  Where it was obtained, when it was compiled, and what coverage it claims all
  need saying — a marker will ask, and "audio features were joined" without a
  source is the weakest sentence you could write here.
- **The join is inner, and that is a design choice.** A play whose track has no
  match is dropped entirely; the alternative was a left join with imputed or
  masked features. Say which you chose and why.
- One sentence flagging that Spotify deprecated this endpoint in November 2024, so
  the features are not obtainable for new applications (→ ch. 9), which is also
  why a third-party corpus was necessary.

### 4.5 Label definition (~110 words)

- **`skipped = reason_end ∈ {fwdbtn, clickrow}`.** State it exactly.
- Justify both members — and the numbers make this easy. **`clickrow` is 0.57% of
  positives** (7,528 rows against `fwdbtn`'s 1,314,673). The label is `fwdbtn` in
  all but name, so the "two different acts" objection is immaterial and no
  sensitivity check is needed. State the split and move on.
- **The negative class is the one that needs defending, not the positive.** Only
  63.9% of non-skips are `trackdone`, i.e. genuine completions. A further 27.5%
  are `endplay` and 3.7% are `backbtn`, with the rest logouts, remote transfers
  and unexpected exits. Two consequences to state:
  - `endplay` does not mean the user listened through — it means playback stopped
    for a reason other than track completion. Treating 611,914 such rows as
    negatives is a choice, and a marker will ask about it.
  - `backbtn` ends the current track early, which is behaviourally closer to a
    skip than to a completion, yet it sits in the negative class.
  You do not need to change the labelling — argue that both are non-skips because
  neither is an explicit forward action by the user, which is what the label is
  defined to capture. But make the argument rather than leaving it implicit.
- Source: `Models/label_composition.csv`, produced by `Data Scripts/dataset_stats.py`.
- Base rate: **38.1% of training rows are skips.** Note this is high relative to
  published streaming figures, and explain why — 4.7's filter guarantees it.
- Cross-reference ch. 1's ambiguity caveat rather than repeating it.

### 4.6 Feature derivation (~180 words)

Fifteen features: ten audio, five behavioural. Group them and justify each group.

- **Behavioural**: `historical_skip_rate`, `historical_artist_skip_rate`,
  `shuffle`, `is_repeat_track`, `same_artist_as_prev`.
- **`historical_skip_rate` is leak-free by construction and you should say so
  explicitly.** It is an expanding mean over that user's *prior* plays of the same
  track, computed as `.shift(1).expanding()` — the current row never contributes
  to its own feature. Same for the artist variant. A marker checking for temporal
  leakage will look precisely here, and the shift is your answer.
- **`track_length` is estimated, not measured**, and this is the chapter's most
  important caveat. It is the maximum `ms_played` across plays that ended
  `trackdone`; for tracks never played to completion it is the maximum observed
  `ms_played`, floored at four minutes. `ms_played` is then clipped to it. Say
  what this means: for a rarely-completed track the estimate is a lower bound, and
  PWTS's time weight is computed from the ratio `ms_played / track_length`, so the
  loss inherits the error.
- Derived context: `hour`, `day_of_week`, `song_pos` — note which are used as
  model inputs and which only feed the loss.
- **`actively_selected` and `reason_start` were computed and then excluded** on the
  deployability constraint from ch. 5. Say they exist in the pipeline but never
  reach a model, so a reader of the code does not think you forgot.

### 4.7 Validity filtering (~100 words)

Three criteria, applied *after* the audio join:

1. **≥ 7 tracks** — shorter sessions cannot support a context/query split with a
   meaningful window.
2. **> 1 unique artist** — removes single-album and single-artist runs where the
   ordering task is degenerate.
3. **≥ 1 skip** — a session with no skips has no ranking signal to recover.

- **This is the pipeline's largest cut by far, and the chapter has to own it.**
  It removes 1,167,427 rows and **172,948 sessions — 61.5% of everything that
  survived the audio join.** Only 36.5% of constructed sessions reach a model.
  Report both figures.
- **Criterion 3 is the one to defend openly.** It guarantees every session
  contains a positive, which raises the observed skip rate and lifts the random
  floor. It is defensible — a session with no skips cannot be reordered to reduce
  skips, so it carries no signal about ordering — but it means your 38.1% base
  rate is not an estimate of the real skip rate, and no figure in the report
  should be read as one.
- The honest framing: the models are evaluated on the sessions where reordering
  could help, not on a representative sample of listening. That is the right
  population for the research question and the wrong one for any claim about how
  often people skip. Draw that line yourself.
- If you have the time, a one-line breakdown of which criterion removes what
  would strengthen this considerably — the three overlap, and a marker will
  wonder whether the length floor or the skip requirement is doing the work.

### 4.8 Normalisation and splitting (~150 words)

- **Nine continuous audio features are z-scored within session** (`mode` excluded,
  being binary). Give the reason: it removes between-session level differences so
  the model compares tracks *against the queue they sit in*, which is the actual
  decision a reorderer makes.
- **Check this before writing.** `dataset_splitter.py` prints "Normalising Dataset
  (per user)" but groups by `session_id`. The code is per-session; the message is
  wrong. Fix one of them and describe what the code does.
- Zero within-session standard deviation → NaN → filled with 0.0. One sentence.
- **Chronological split within each user**, 70/15/15 by session count — realised
  as exactly 0.700 / 0.150 / 0.150. Justify: a random split would let a model see
  a user's later behaviour while predicting their earlier behaviour.
- **Pre-empt the obvious objection.** The three splits' global date ranges overlap
  (training runs to 2026-03, testing starts 2022-09). That is not leakage: the
  split is chronological *per user*, and users' histories start at different
  times, so one user's test sessions can predate another's training sessions.
  Say this in the chapter — a marker who spots the overlap without the
  explanation will assume the worst.
- **Note the drift it produces**, because it matters for ch. 8. It is not confined
  to the label — every behavioural feature moves the same way across the splits:

  | | train | validation | test |
  |---|---|---|---|
  | skip rate | 0.3815 | 0.3462 | 0.3559 |
  | shuffle share | 0.6274 | 0.6085 | 0.5955 |
  | repeat-track share | 0.1526 | 0.1275 | 0.1203 |
  | same-artist-as-prev | 0.2040 | 0.1739 | 0.1601 |
  | mean session length | 33.5 | 31.4 | 30.6 |

  The test distribution is not the training distribution, and every reported test
  number sits under that shift. This is the honest cost of a chronological split
  and is worth one sentence saying you would rather have it than the leakage a
  random split would introduce.
- Source: `Models/dataset_stats.csv`, produced by `Data Scripts/dataset_stats.py`.

### 4.9 Evaluation protocol (~cross-reference only, ~40 words)

- Do not describe the protocol here — it belongs in ch. 7. One sentence pointing
  forward is enough.
- Do state the one fact that belongs to the data: of 16,280 test sessions, 11,695
  yield a valid context/query split and are scored. Say why the others do not, so
  the sample-size difference between ch. 4 and ch. 8 is explained where it first
  appears.

### 4.10 Tools (~60 words)

- PyTorch 2.x, scikit-learn, pandas, PyArrow, NumPy; SLURM job arrays on the
  Birkbeck DCS cluster for tuning and training.
- One sentence on why the cluster was necessary — 1,080 tuning configurations is
  not a laptop workload. Detail belongs in ch. 6.

### Before you write this chapter

- Resolve the "per user" / `session_id` discrepancy in `dataset_splitter.py`.
- Have the ethics reference number to hand.

## 5. Requirements specification & Design (~1,900 words) — 20% of marks

**Budget warning.** The eleven subsections below sum to roughly 1,900 words,
against the 1,100 originally allocated. That is defensible — this chapter carries
your contribution and now covers all four model families — but the 800 words have
to come from somewhere. Take them from ch. 1 and ch. 3 rather than from ch. 6,
which is worth 30% to this chapter's 20%. Revised suggestion: ch. 1 → 600,
ch. 3 → 1,400, ch. 5 → 1,900.

The chapter where your distinctive contribution lives. Chapter 4 described what
was built; this one has to argue *why it had to be built that way*. Every
decision below should read as a constraint producing a consequence, not as a
description of code.

### 5.1 The deployability requirement (~200 words)

*This is the spine of the chapter. Lead with it.*

- **The rule, stated once and plainly:** a feature may be used as a model input
  only if its value is knowable for **every unplayed track in the queue** — not
  merely for the next one.
- **Why the stricter form.** The system reorders the entire remaining queue after
  each listening event. `reason_start` for a track is determined only once the
  track immediately before it in the realised play order has finished. For any
  track beyond the very next slot, that has not happened and cannot be known.
- Name the failure it prevents: **train/serve skew.** Offline logs contain the
  field for every row post-hoc; the deployed system would not. A model trained on
  it would score well in evaluation and be unusable in production.
- Say that the softer reading — "it is available for the immediate next track, so
  use it there" — was considered and **rejected**, in favour of a single feature
  set valid across the whole queue-scoring task. That choice is the contribution.

### 5.2 Pricing the constraint (~180 words)

*Do not assert the trade-off. Quantify it. This is the strongest passage
available to you in the whole report.*

- `actively_selected` was the most predictive signal in the data:

  | | not skipped | skipped |
  |---|---|---|
  | `actively_selected = 0` | 59.6% | **40.4%** |
  | `actively_selected = 1` | 91.9% | **8.1%** |

  A fivefold difference, and a correlation with `skipped` of **−0.240** —
  stronger than any feature you kept.
- **State that you verified no proxy survived**, rather than assuming it. The
  strongest correlate among retained features is `shuffle` at −0.210, and
  conditioning on it does not remove the effect: skip rates are 32.2% vs 7.8%
  with shuffle off, and 45.0% vs 8.7% with shuffle on. If `shuffle` were carrying
  the signal those columns would converge. They do not.
- The contemporaneous measurement: removing it moved AUC from ~0.6192 to ~0.5779
  on an identical hyperparameter configuration. **Caveat it** — that configuration
  also carried the auxiliary head, so the figures are not comparable to your final
  table. Cite them as the evidence that informed the decision, not as a result.
- Close the paragraph with the point that reframes your whole results chapter:
  the modest headline numbers are partly the price of this constraint, knowingly
  paid.

### 5.3 The design evolution of the excluded features (~130 words)

*A decision with three stages reads far better than a flat exclusion.*

- **Stage 1:** `actively_selected` and `reason_start` included as inputs.
- **Stage 2:** excluded as inputs but retained as a **training-time-only
  auxiliary target** via `SASRec.aux_head` — the learning-under-privileged-
  information pattern. Legitimate because the target is never needed at inference.
- **Stage 3:** the auxiliary head removed entirely, taking `aux_criterion`,
  `aux_target`, `aux_weight`, and `reason_start` from `SessionDataset` and
  `collate_fn` with it.
- Give the reason for stage 3 honestly. Then point forward: privileged-information
  distillation — training a teacher with the oracle feature and distilling into a
  deployable student — is the natural continuation, and belongs in ch. 9.

### 5.4 Model selection (~150 words)

- **Two tiers, four families:** interpretable baselines (logistic regression,
  extremely randomised trees) and sequential neural models (SkipLSTM, SASRec).
- **Justify having baselines at all**, and cite Ferrari Dacrema et al. (2019) when
  you do: neural recommenders routinely fail to beat simple methods once tuned
  equally, so a comparison without strong baselines is unfalsifiable. Your results
  vindicate the decision — ExtraTrees closes 77% of the Random-to-best gap.
- **Why two of each.** Logistic regression gives a linear reference point and
  signed coefficients; ExtraTrees adds non-linear interactions and feature
  importances; the LSTM adds recurrence; SASRec adds self-attention. Each tier
  step isolates one capability.
- **All four score the same candidate set under the same protocol.** That is what
  makes the tiers comparable, and it is a design decision, not an accident —
  `ndcg_at_k`, `valid_splits`, `pick` and `fast_auc` are imported from `Model_lib`
  by the sklearn scripts too, so the baselines cannot silently diverge.

### 5.5 Baseline design (~200 words)

- **Both baselines are deliberately *flat*.** They score each track independently
  from its own feature vector, with no access to session order or to what happened
  earlier in the session. That is the point: they establish what is achievable
  *without* sequence modelling, so the neural tier's gain over them is the value
  of sequence modelling specifically.
- **Logistic regression:** `lbfgs`, `max_iter=1000`, `C=1.0`. **Say that `C` was
  not tuned and why** — with 15 features and 2.5M training rows the model sits far
  from the regularisation-sensitive regime, so the penalty is effectively inert.
  Stated, that reads as judgement; unstated, it reads as an omission.
- **ExtraTrees rather than Random Forest** — a deviation from the proposal, so
  justify it: extremely randomised trees draw split thresholds at random rather
  than searching for the optimal one, which lowers variance and trains
  substantially faster at this data scale.
- Grid over `max_depth`, `min_samples_leaf`, `n_estimators`.
- **Reconcile before writing:** `extratrees_grid_search_results.csv` contains only
  `max_depth=8` rows, with (8, 20, 100) best at 0.7666 — but `ExtraTreesClassifier.py`
  hardcodes `max_depth=15, min_samples_leaf=20, n_estimators=200`. The recorded
  search does not support the parameters in use. Either the CSV is a partial merge
  or the values came from elsewhere; find out which.
- Feature importances are a second output, used in ch. 8.

### 5.6 SkipLSTM design (~200 words)

- **Purpose: isolate recurrence.** Same 15 inputs, same status channel, same
  losses, same evaluation protocol as SASRec — only the sequence encoder differs.
  Say this explicitly; it is what makes 8.4 a comparison of *architectures* rather
  than of two unrelated systems.
- **The front end is identical to SASRec by design:** `feature_proj = Linear(15,
  H)` summed with `status_emb`, then dropout. Deliberate parity, so the comparison
  isolates the encoder rather than confounding it with input handling.
- **Input dropout is applied explicitly**, and the reason is a real trap avoided:
  `nn.LSTM` silently ignores its own `dropout` argument when `num_layers == 1`.
  Without a separate `emb_dropout`, `dropout_rate` would have been inert across a
  third of the grid — a tuned hyperparameter doing nothing.
- `pack_padded_sequence(..., enforce_sorted=False)` with `total_length` pinned to
  the collated width, so per-chunk predictions concatenate correctly during
  evaluation, where chunks have different maximum lengths.
- `fc = Linear(H, 1)` emitting **logits**, matching SASRec (→ 5.10).
- It accepts `boundaries` and deliberately ignores it — signature parity so the
  two models are swappable; the consequence is in 5.9.
- Tuned over `hidden_units`, `num_layers`, `dropout_rate`, `lr`. Selected for
  PWTS: 128 units, 1 layer, dropout 0.3, lr 1e-3 — **notably smaller than the
  DrRL winner** (256 units, 3 layers). Worth a sentence: the smaller model won.
- **Limitation to name here, not only in ch. 8:** two of three LSTM arms selected a
  configuration whose best epoch reached the tuning cap (`bce` 40/40, `pwts`
  18/20). The search may have favoured configurations that converge fastest rather
  than train best.

### 5.7 SASRec design and adaptation from the reference (~280 words)

*Cite the implementation you started from and say exactly what changed. Being
explicit about what you inherited is what makes the claim about what is yours
credible — and an examiner will look for this section.*

**Write it as a derivation, not an inventory.** One change forces all the others,
and a chain of consequences reads as engineering judgement where a list reads as
a changelog.

**The root change is the task.** The reference solves *next-item prediction over a
catalogue*: given a history, score every item and rank them. This model solves
*per-track binary skip prediction within a fixed candidate set the user has
already chosen*. Every difference below follows.

1. **The item embedding table has to go.** `nn.Embedding(item_num+1, H)` gives
   each catalogue item a vector looked up by ID — the reference is built on item
   identity. This model must score tracks absent from training, which a lookup
   table structurally cannot. `nn.Linear(15, H)` produces the same H-dimensional
   vector from the track's *properties* instead.
2. **Which makes the `sqrt(d)` scaling meaningless.** `seqs *= embedding_dim ** 0.5`
   exists to match embedding-lookup variance to the positional embeddings; a
   linear projection is already scaled by its initialisation. Removed.
3. **Which forces a new output head.** `predict()` dot-products the sequence state
   against candidate item embeddings; with no embeddings there is nothing to dot
   against. `nn.Linear(H, 1)` reads one scalar per position instead — which works
   because every position already *is* a track, the whole queue being fed through.
4. **Which makes negative sampling unnecessary.** The reference samples one
   negative per step because scoring a whole catalogue is infeasible. Here the
   candidate set is a queue: small, fixed, every track carrying an observed skip
   label. Full binary supervision at every query position; `pos_seqs`/`neg_seqs`
   disappear.
5. **Then something is added that the reference has no analogue for.** SASRec's
   sequence is item IDs and the interaction is implicit. Here the *outcome* of
   each played track is known, so `status_emb = nn.Embedding(3, H)` is **summed**
   into the representation — additive, so `H` is unchanged and the transformer
   blocks are untouched (→ 5.8).
6. **Then the mask breaks, and this is the real intellectual step.** The reference
   mask is causal only, one `(L, L)` matrix shared across the batch — correct when
   position *i* predicts item *i+1*. This model scores many positions at once, so
   under a causal-only mask a query position attends the query positions before
   it, and a track's score depends on which tracks happen to precede it in the
   queue as it currently stands. Reorder the queue and the scores change; but the
   scores produced the ordering. **The ranking would be circular.** Hence
   `causal | (qq & ~eye)`, and hence a per-sample `(N × heads, L, L)` mask, since
   `boundaries` differ by session (→ 5.9).
7. **And the loss becomes pluggable.** The reference hardcodes BCE over sampled
   logits. Here the model emits raw logits — required because DrRL saturates on
   values confined to [0,1] — and three objectives share one interface (→ 5.8).

**The sentence that ties it together:** *the reference identifies items; this model
characterises them.*

| Component | Reference SASRec | This project | Why |
|---|---|---|---|
| Item representation | `nn.Embedding(item_num+1, H, padding_idx=0)` | `nn.Linear(15, H)`; **no item embedding table** | Must score tracks absent from training |
| Embedding scaling | `seqs *= embedding_dim ** 0.5` | removed | Meaningless for a linear projection |
| Extra input channel | none | `status_emb = nn.Embedding(3, H)`, zero-init on *unknown* | Carries observed skip outcomes for played context tracks |
| Attention mask | causal only, `~torch.tril(ones)` | `causal \| (qq & ~eye)` | Makes scores independent of realised queue order |
| Mask shape | `(L, L)`, broadcast | `(N × heads, L, L)` via `repeat_interleave` | Boundaries differ per session |
| Output head | dot product with candidate item embeddings | `nn.Linear(H, 1)` — one scalar per position | Pointwise skip logit, not item retrieval |
| Training objective | BCE over one positive and one **sampled negative** | full binary label at every query position; BCE / PWTS / DrRL | Every unplayed track has a ground-truth label |
| Negative sampling | required | **removed entirely** | as above |
| Positional embedding | `Embedding(maxlen, H)` | `Embedding(maxlen+1, H, padding_idx=0)`, explicit position tensor, clamped | Padding index reserved; positions past `maxlen` clamped, not truncated |
| Normalisation order | post-LN | `norm_first=True` (pre-LN) | More stable at depth |
| Sequence handling | left-padded, fixed `maxlen` window | variable length, `lengths` passed explicitly | Sessions range 7–1,148 tracks |

- **State what was *not* changed**, in one sentence: `PointWiseFeedForward`
  (Conv1d, kernel size 1), the attention block loop, the layer-norm and residual
  structure, the `norm_first` option and the `MultiheadAttention` calls are all
  inherited unmodified.
- **Note that `SkipLSTM` is not derived from SASRec at all.** It is an independent
  baseline written to share the same `(x, lengths, status, boundaries)` signature
  so the two are directly swappable — which is also why it accepts `boundaries`
  and ignores it (→ 5.9).
- **Confirm the source before writing.** The class structure matches the PyTorch
  port (`pmixer/SASRec.pytorch`) rather than Kang & McAuley's TensorFlow release:
  `PointWiseFeedForward` as a named Conv1d class, `self.dev = args.device`,
  `~torch.tril(torch.ones((tl, tl), ...))`, the four parallel `ModuleList`s, and
  the `args.hidden_units` / `args.num_blocks` naming. Verify it, then cite the
  paper for the method and the repository for the implementation.
- **There is currently no attribution anywhere in the source.** Add a header
  comment to the `SASRec` class and to `PointWiseFeedForward` naming the
  repository, its licence and what changed. The modifications are substantial and
  genuinely yours, but substantial modification does not remove the need to credit
  the starting point.
- `maxlen=200` and `norm_first=True` were fixed rather than tuned. Say so.

### 5.8 The skip-status channel (~150 words)

- `nn.Embedding(3, hidden_units)` over three states — completed, skipped, unknown
  — **added** to the projected features rather than concatenated, so input
  dimensionality is unchanged.
- `nn.init.zeros_(self.status_emb.weight[2])` zero-initialises the *unknown*
  state, so an unplayed track begins as pure feature signal and the channel
  contributes nothing until the model learns it should.
- **Why this is deployable** while `actively_selected` is not: outcomes for tracks
  already played *earlier in the current session* are observed facts at prediction
  time. The constraint in 5.1 excludes future information, not past information.
  Make that distinction explicitly — it is the one a reader is most likely to
  challenge.
- **The consequence for ch. 8.5, stated here first:** because the channel carries
  observed skip outcomes regardless of feature set, a neural "audio-only" run is
  not audio-only in the sense the sklearn baselines are. The measured gap is
  0.5779 against the baselines' 0.5038 — that difference *is* the channel.

### 5.9 The context/query attention mask (~200 words)

*The most technical passage in the report. Worth the words.*

- `build_attn_mask` returns `causal | (qq & ~eye)`:
  - `causal` blocks position *i* from attending any *j > i*
  - `qq` blocks every query position from attending **any other query position**
  - `~eye` preserves the diagonal, so a query position still attends itself
- The effect: each unplayed track is scored from the context segment plus its own
  features alone. Its score does not depend on which other unplayed tracks
  surround it.
- **Frame this correctly, and this is the sentence to get right.** It is *not* a
  fix for future leakage — the causal mask already handles that. It makes the
  score **well-defined for reordering**. If a track's score depended on its
  neighbours among the unplayed set, then reordering the queue would change the
  scores that produced the ordering, and the ranking would be circular.
- **The asymmetry, named by you before a marker names it.** `SkipLSTM.forward`
  takes `boundaries` and deliberately does not use it — an LSTM's hidden state is
  strictly sequential, so every position sees all earlier positions, query
  positions included. **The LSTM therefore conditions on the realised queue
  order and is solving a strictly easier problem.** Say so here, and again in 8.4.
  A comparison you have disqualified yourself scores better than one presented as
  clean.

### 5.10 Loss function design (~180 words)

- Three objectives: **BCE** (baseline), **PWTS** (proposed here), **DrRL**
  (adapted from Zhang et al., AAAI 2025).
- **Models emit logits, not probabilities.** Give both reasons: DrRL is defined
  over unbounded ranking scores and its gradient saturates when applied to values
  confined to [0,1]; and `BCEWithLogitsLoss` is the numerically stable form. Note
  that AUC and NDCG are rank-based, so the sigmoid would not change them anyway.
- **PWTS, with its three weights and the reasoning for each:**
  - position: `1 + exp(−percentage_pos × 3)` — early-session errors are the ones
    a user actually notices
  - time: a ramp from 2 to 1 across `percentage_listened ≤ 0.1` — a near-instant
    skip is a stronger signal than a late one
  - historical: `1 + sigmoid((historical_skip_rate − 0.5) × 6)`
  - the combined weight is normalised to mean 1, **so with all three disabled it
    reduces exactly to BCE** — which is what makes the ablation a clean test
- Cross-reference forward to 8.6 rather than pre-empting it.

### 5.11 Output and requirements traceability (~120 words)

- The models emit one skip logit per track; the queue is reordered **ascending by
  predicted skip probability**.
- **State the NDCG convention explicitly**, because it inverts the usual one:
  relevance is *not skipped*, and the ranking is ascending. A reader who assumes
  the standard convention will misread every number in ch. 8.
- Close with a short numbered requirements table — requirement, the design
  decision that satisfies it, and the section where it is evaluated. Markers
  reward traceability, and it costs you eighty words.

### Before you write this chapter

- Decide your chapter map. The report currently calls Data Processing chapter 4
  and refers forward to "chapter 5" for excluded features and "chapter 6" for
  models. Make it consistent before writing cross-references.
- A block-mask diagram would earn marks — a small grid with context and query
  regions shaded, showing which cells are blocked. It communicates 5.6 in a way
  prose cannot.
- Every claim in 5.2 is measured and reproducible; quote the figures rather than
  writing "significantly more predictive".

## 6. Implementation (~1,050 words) — 30% of marks

**Not a tour of the code.** Select the decisions that required judgement and the
errors that required diagnosis. The brief asks explicitly for error analysis, and
this is the chapter where you have more of it than most projects ever produce —
six defects, each found by measurement rather than by a crash. Lead with that as
an asset rather than burying it at the end.

### 6.1 Structure and reuse (~110 words)

- `Model_lib.py` is a shared library, not a utility dump: `SessionDataset`,
  `collate_fn`, both architectures, the training loop and every metric.
- **The point worth making:** `ndcg_at_k`, `valid_splits`, `pick` and `fast_auc`
  are imported by `Log_regression.py` and `ExtraTreesClassifier.py` too. The sklearn
  baselines and the neural models are therefore scored by *the same functions*,
  not by two implementations that agree by inspection. That is what makes the
  nine-row table in ch. 8 a comparison rather than a collation.
- The four models share one call signature `(x, lengths, status, boundaries)`, so
  `evaluate.py` treats them interchangeably.

### 6.2 Performance engineering (~260 words)

*Three optimisations, each with a measured before-and-after. Give the numbers —
an optimisation without a measurement is an assertion.*

- **`fast_auc`** — AUC via the Mann-Whitney rank statistic rather than
  `sklearn.roc_auc_score`. Agrees with sklearn to **1.7 × 10⁻¹⁶** and is
  **14.3× faster**; it skips per-call input validation and never materialises the
  ROC curve. Justified because AUC is computed once per session per split point
  per epoch — tens of thousands of calls per training run, not one.
- **`LengthBucketSampler`** — batching sessions of similar length together.
  Padding waste fell from **6.23× to 1.01×** and training ran **3.2× faster**,
  with no change to the loss, because sessions range from 7 to 1,148 tracks and a
  naïve batch pads everything to its longest member.
- **Chunked corpus scan** in `audio_features_clean.py` — the audio-feature corpus
  is 11 GB across ten Parquet files. Each is read with column projection
  (11 of 17 columns), filtered to matching track IDs, and released before the next
  is opened, so peak memory holds only the matched subset rather than the corpus.
- **Chunked evaluation** — `evaluate_sequential` expands each session into one row
  per split point, which multiplies batch size; the forward pass is chunked to
  keep long sessions inside memory.
- Report these as engineering decisions with reasons, not as a list of tricks. The
  common thread is that each follows from a property of *this* data — session
  length variance, call frequency, corpus size.

### 6.3 Running the experiments (~200 words)

- **1,080 tuning configurations** as SLURM job arrays, split 54/54/162 for the
  LSTM and 162/162/486 for SASRec (→ 8.1 for why they differ).
- **One configuration per array task, each writing its own CSV**, merged
  afterwards by script. Say why: concurrent tasks appending to a shared file
  interleave and corrupt it, and a per-task file also makes a failed task visible
  as a missing file rather than as silently absent rows.
- **The environment failures are worth reporting**, and this is the section for
  them:
  - `sbatch --wrap` executes under `/bin/sh`, where `source` does not exist, so
    `source ~/venv/bin/activate` fails silently and the job falls through to the
    system interpreter. Calling the venv's `python3` by absolute path is the fix.
  - Deleting the virtual environment while an array was queued killed **742
    tasks** with exit 127 under `set -euo pipefail`.
  - `--gpus=1` requires `cons_tres`; `--gres=gpu:1` is the portable form.
  - SASRec/BCE at 256 hidden units and 16 heads exceeded GPU memory on the
    default partition and required `a40`.
- **Frame these as operational findings, not as apology.** A dissertation that
  reports what broke on a shared cluster is more useful than one that implies
  everything ran first time.

### 6.4 Defects found and corrected (~360 words)

*The strongest section available to you. Every one of these was found by
measurement, not by a crash — say so, because that is the distinction between
debugging and error analysis.*

**Two in DrRL:**

- **β stepped twice per batch.** `self.beta` is an `nn.Parameter`, so it reached
  the main Adam optimiser through `criterion.parameters()` *and* was stepped by
  its own SGD instance inside `update_beta`. The gradient left behind by the
  second was consumed by the first, moving β at **roughly 166× the intended
  rate**. Fixed by clearing `self.beta.grad` after the dedicated step. Note the
  cause: two optimisers is the reference implementation's design, and the bug came
  from wiring rather than from the method.
- **Gradient saturation on bounded outputs.** DrRL is defined over unbounded
  ranking scores; applied to sigmoid outputs the gradient fell from
  **4.28 × 10⁻⁴ to 1.04 × 10⁻⁴** as the logit spread grew from 1 to 6 — the loss
  weakened exactly as the model improved. Fixed by removing the output sigmoid
  across both architectures (→ 5.10).

**Two in the pipeline:**

- **Temporal leak in the historical rates.** `historical_skip_rate` and
  `historical_artist_skip_rate` sorted by the *string* `session_id` before the
  expanding mean. For users with ten or more sessions, string ordering puts
  `_10` before `_2`, so rows from later sessions entered the "past" window. Fixed
  by sorting on `ts`. **This is the most serious of the six** — it is silent, it
  inflates results, and it would have survived to submission undetected.
- **Early stopping firing before convergence.** In the PWTS ablation the variant
  ranking was perfectly separated by training length: the top seven variants were
  exactly the seven whose seeds all ran ≥15 epochs, the bottom eight exactly those
  with a seed stopping at 7 or 9. A minimum-epoch floor and a re-run removed the
  artefact entirely — correlation between training length and score fell from
  **+0.751 to −0.016**, and seed variance from ~0.0055 to 0.0008 (→ 8.6).

**Two in tooling:**

- **`evaluate.py` saved the last seed's per-session scores** while reporting the
  five-seed mean, so any paired test would have tested a model that was not the
  one in the results table. Detected because the three single-seed rows matched
  the table exactly and all six multi-seed rows did not, with the discrepancy
  tracking seed variance (worst on DrRL at 0.021 NDCG@5).
- **A silently ignored command-line flag.** `parse_known_args` discards
  unrecognised arguments, so `--feature-set` passed to a script that did not
  implement it was dropped without error, training the full-feature model and
  overwriting the headline checkpoints.

### 6.5 Reproducibility (~120 words)

- Five seeds per arm, **every checkpoint retained**, not only the best — the
  per-seed spread is what puts error bars on ch. 8.
- Validation loaders are length-sorted with `shuffle=False`, and `pick()` selects
  split points deterministically by even spacing rather than sampling, so
  evaluation has no run-to-run variation and no seed of its own.
- Selected hyperparameters are written to JSON per arm, so training and evaluation
  read the same configuration rather than restating it.
- **State honestly what is not reproducible:** UUIDs are regenerated on each
  pipeline run, so user and session identifiers differ between executions even
  though the data does not.

### Before you write this chapter

- Every number above is measured and in your notes — quote them rather than
  writing "significantly faster".
- Decide the split with ch. 5: *design decisions* there, *how they were realised
  and what went wrong* here. `total_length` pinning and the `nn.LSTM` dropout trap
  currently sit in 5.6 and belong here.
- Point to an appendix for the full tuning tables rather than reproducing them.

## 7. Testing and Evaluation (~1,300 words) — part of 30%

**This chapter justifies the measurement; chapter 8 reports what it measured.**
Keep results out of it. Its job is to make every number in ch. 8 credible before
the reader sees one, which is why it comes first.

### 7.1 Per-session rather than pooled scoring (~230 words)

- **The decision:** every metric is computed within a session and averaged across
  sessions, never pooled over all rows at once.
- **The reason, stated as a property of the task:** pooled AUC is rewarded for
  separating high-skip-rate sessions from low-skip-rate ones. A queue reorderer
  never makes that comparison — it only ever orders tracks *within* one session,
  so any credit for between-session discrimination is credit for a decision the
  system does not make.
- **Give the evidence rather than asserting it.** Your feature ablation contains a
  natural experiment:

  | | pooled AUC | per-session AUC |
  |---|---|---|
  | ExtraTrees, behavioural (5) | 0.7201 | 0.6051 |
  | ExtraTrees, audio (10) | 0.5499 | 0.5038 |
  | LogReg, behavioural (5) | 0.7082 | 0.5916 |
  | LogReg, audio (10) | 0.5060 | 0.5068 |

  The behavioural rows lose ~0.11 AUC when scored per session; the audio rows lose
  almost nothing. That is exactly the signature of `historical_skip_rate` acting
  as a proxy for session-level base rate — a between-session signal that pooled
  AUC banks and per-session scoring correctly discards.
- **Verify before quoting** the ~30.9% between-session variance figure from your
  earlier analysis; it is the cleanest single number for this argument but it is
  not currently reproducible from any saved artefact.
- Cite Zhou et al. (2018) if the per-session/pooled distinction has a source you
  have read; do not cite it otherwise.

### 7.2 Multi-point splitting (~190 words)

- Each session is scored at up to **five** context/query split points rather than
  one, and the results are averaged **within** a session before averaging across
  sessions — so every session contributes equally regardless of length.
- **The failure this prevents:** with a single split point, a session's score
  depends on where the draw happened to land. An unlucky split makes a competent
  model look poor on that session, and with ~12,000 sessions those draws do not
  simply cancel — they add variance that is indistinguishable from a real
  difference between models.
- **`pick()` is deterministic** — five evenly spaced points across the valid
  range, not a random sample. Evaluation therefore has no seed and no run-to-run
  variation, so a repeat of `evaluate.py` on the same checkpoints returns
  identical numbers. Say this: it is a reproducibility guarantee, not an
  implementation detail.
- State the constants and what they imply: `MIN_CONTEXT = 3`,
  `MIN_WINDOW_LENGTH = 5`, so a session needs **at least 8 tracks** to be
  scorable at all — one more than the 7-track filter in ch. 4 admits.

### 7.3 Metrics (~250 words)

- **NDCG@5 is primary.** State the convention explicitly and early, because it
  inverts the usual one: the queue is ordered **ascending** by predicted skip
  probability, and a track is **relevant if it was not skipped**. A reader who
  assumes the standard convention will misread every figure in ch. 8. Cite
  Järvelin & Kekäläinen (2002).
- **AUC secondary**, computed from the Mann-Whitney rank statistic (Hanley &
  McNeil, 1982) — see ch. 6 for the implementation.
- **NDCG@10 and P@5** as robustness checks: P@5 is the most directly
  interpretable — "of the next five tracks this system would play, how many are
  kept?"
- **`RankFirstSkip`** — how many tracks play before the listener hits a skip. Give
  it prominence: it is the only metric stated in units a reader understands
  without explanation, and it is the quantity a queue reorderer exists to
  increase.
- **MRR was measured and dropped.** It spanned only 0.767–0.854 across every model
  including Random, because with ~62% of tracks unskipped the first kept track is
  almost always at rank 1. Report that you tested it and rejected it — a metric
  you evaluated and discarded is worth more than one you never considered.
- **The random floor belongs here, not only in ch. 8.** NDCG@5 0.6881 and AUC
  0.5028 on the test set. Every metric needs its floor stated at the point it is
  defined, so the reader carries it into the results.

### 7.4 Experimental design (~250 words)

*Three stages, and the separation between them is what makes ch. 8 credible.*

- **Tuning** — 1,080 configurations, scored on validation only. State the split
  per arm and why the arms differ (→ 6.3).
- **Training** — the selected configuration retrained over **five seeds**, again
  selecting on validation.
- **Test** — scored **once**, after all selection was complete. Say this in those
  words. It is the single most important sentence in the chapter.
- **The selection metric is NDCG@5 smoothed over a three-epoch window, not the
  per-epoch maximum.** Give the reason: taking a maximum over epochs is biased
  upward, because it selects the epoch whose validation noise happened to be
  favourable. Smoothing removes most of that. Your raw per-epoch value swings by
  ~0.03, which is larger than the differences being compared.
- Early stopping on the smoothed metric with patience 5, max 100 epochs.
- **Identical protocol across all arms** — same batch size, patience, smoothing
  window, seed count and evaluation code. State the fixed constants explicitly
  (batch 64, patience 5, window 3, five seeds); an unstated constant reads as
  something you did not consider.
- Cite Shehzad & Jannach (2023) on unequal tuning budgets manufacturing wins, if
  you have read it — and note that your budgets were unequal *by hyperparameter
  count*, with the two worst arms receiving the largest allocations.

### 7.5 Resolution and significance (~200 words)

- **State the noise floor before any comparison.** Mean seed standard deviation
  across the six neural arms is ~0.0057 on NDCG@5, so differences below roughly
  **0.011** (2 sd) are not resolvable by inspection of the means alone.
- **Significance is by paired bootstrap over per-session scores**, 10,000
  resamples, not by comparing means to standard deviations. Explain why pairing
  matters: every model scores the *same* sessions, so the paired difference
  removes between-session variance, which is the dominant source of spread.
- Cite Smucker, Allan & Carterette (2007) — bootstrap and randomisation tests are
  preferred to t-tests for IR evaluation.
- **State the one comparison you cannot make.** The sklearn baselines scored
  11,694 sessions to the neural models' 11,695, so baseline-versus-neural
  differences are **unpaired**. Either reconcile the missing session or report
  those comparisons without a paired test — do not quietly pair them.

### 7.6 Known limitations of the protocol (~180 words)

*These belong here rather than ch. 8: they are properties of the measurement, not
of the results.*

- **Scoring begins at the committed slot.** The protocol scores from context
  length *c* onward, but a deployed reorderer cannot alter the track at *c* — that
  slot is already playing by the time the reorder completes. The reported figures
  therefore credit the model for ranking a position it could not control. With
  `MIN_WINDOW_LENGTH = 5` that is one of at least five positions, and it is the
  one NDCG discounts least.
- **No session is scored before its first skip.** `valid_splits` requires at least
  one skip in the context, because the status channel carries no discriminating
  signal otherwise. But that is precisely the window in which a deployed system
  would be asked to act, so cold-start-within-session behaviour is untested.
- **28.2% of test sessions yield no valid split** and are excluded entirely.
- **The test distribution differs from training** — skip rate 0.381 / 0.346 /
  0.356 across the splits, the cost of the chronological split (→ ch. 4).

### Before you write this chapter

- Nothing here requires the re-run. Every constant, condition and justification is
  readable from the code.
- Write it **before** ch. 8. A reader who has accepted the protocol reads the
  results as findings; one who meets the protocol afterwards reads them as claims
  needing defence.
- Resist putting any result in it. The random floor is the one exception, because
  it is a property of the data rather than of a model.

## 8. Results / Findings and Discussion (~2,000 words) — part of 30%

Your strongest chapter, and the one to draft first. Word budgets below sum to
~2,150; trim 8.1 and 8.8 if you run over.

**Blocking issue before 8.2–8.4 can be written:** `eval_per_session.npz` stored
only the *last* seed's per-session scores while `final_results.csv` reported the
5-seed mean, so any paired test run on the old file tests seed 4, not the table.
`evaluate.py` is patched; **re-run it** to regenerate the npz, then re-run the
bootstrap. Sanity check: after the fix, each neural arm's npz array mean must
equal its `final_results.csv` value exactly, as the three single-seed rows
already do.

### 8.0 Opening (~80 words)

- One paragraph stating what the chapter reports and in what order.
- State the **resolution threshold** you will apply and where it came from
  (2 × mean seed sd ≈ 0.011 on NDCG@5), so every later claim can be read against
  it without re-deriving it.
- Say up front that significance is by paired bootstrap over per-session scores,
  10,000 resamples — not by comparing means to standard deviations.

### 8.1 Hyperparameter search (~180 words)

- 1,080 configurations: 6 arms (2 architectures × 3 losses) × 180 configs, run as
  SLURM job arrays.
- **Verify before writing** the claim that only learning rate mattered — recompute
  it from `*_grid_search_*_results.csv` rather than repeating it from memory.
  A marker may ask, and a variance decomposition or a simple per-parameter
  spread table is cheap to produce.
- Report the selected configuration per arm from the `*_best_params.json` files,
  in a small table. Note the LSTM/PWTS winner (128 units, 1 layer, dropout 0.3,
  lr 1e-3) is *smaller* than the DrRL winner (256 units, 3 layers) — worth a
  sentence, since the smaller model won.
- **The budget was not equal across arms — state the real split**, because the
  actual numbers make a stronger argument than equality would:

  | arm | configs |
  |---|---|
  | SkipLSTM/bce | 54 |
  | SkipLSTM/pwts | 54 |
  | SkipLSTM/drrl | 162 |
  | SASRec/bce | 162 |
  | SASRec/pwts | 162 |
  | SASRec/drrl | 486 |
  | **total** | **1,080** |

  The split follows the number of tunable hyperparameters, not preference: DrRL
  adds `gamma` (×3) and SASRec adds `num_blocks`/`num_heads` over the LSTM's
  single `num_layers` (×3). Make the argument this way round: **the two worst
  arms received the largest search budgets** — SASRec/drrl got 486 configs, nine
  times SkipLSTM/pwts's 54, and still finished last. So DrRL's poor showing
  cannot be attributed to under-tuning, and the winning arm cannot be attributed
  to a search advantage. That is a stronger defence against the "everyone's a
  winner" critique than equal budgets would have been.
- Cross-reference the epoch cap in tuning as a known limitation (→ 8.7), do not
  discuss it here.

### 8.2 Final comparison (~350 words)

The nine-row test table is the centrepiece. Put the **Random row at the bottom
and refer to it in the first sentence.**

| model | AUC | NDCG@5 | NDCG@10 | P@5 | tracks before 1st skip |
|---|---|---|---|---|---|
| SkipLSTM/pwts | 0.6445 ±0.0025 | **0.7865** ±0.0030 | 0.8226 | 0.6964 | *(re-run)* |
| SkipLSTM/bce | 0.6427 ±0.0075 | 0.7800 ±0.0050 | 0.8175 | 0.8458 | 0.6913 |
| SkipLSTM/drrl | **0.6448** ±0.0210 | 0.7782 ±0.0100 | 0.8165 | 0.8457 | 0.6874 |
| SASRec/pwts | 0.6358 ±0.0024 | 0.7805 ±0.0018 | 0.8168 | 0.8492 | 0.6919 |
| SASRec/bce | 0.6333 ±0.0039 | 0.7772 ±0.0034 | 0.8144 | 0.8464 | 0.6897 |
| SASRec/drrl | 0.6137 ±0.0153 | 0.7598 ±0.0111 | 0.8011 | 0.8301 | 0.6774 |
| ExtraTrees | 0.6040 | 0.7643 | 0.8049 | 0.8337 | 0.6825 |
| LogReg | 0.5948 | 0.7522 | 0.7963 | 0.8171 | 0.6751 |
| Random | 0.5028 | 0.6881 | 0.7438 | 0.7671 | 0.6254 |

Points to make, in this order:

- **Everything beats the floor**, and by how much: the best arm gains 0.0984
  NDCG@5 over random, which is **31.5% of the 0.3119 that was available**. Report
  the fraction of headroom, not just the raw number — 0.7865 sounds far better
  than it is.
- **Most of that gain is not neural.** LogReg alone closes 65% of the
  Random→best gap; ExtraTrees closes 77%. The entire neural apparatus — two
  architectures, three losses, 1,080 tuning runs — buys the remaining 23%. State
  this plainly. It is the honest reading and it is what Ferrari Dacrema et al.
  would predict.
- **ExtraTrees (0.7643) beats SASRec/drrl (0.7598).** A tree baseline beating a
  tuned transformer arm is worth a sentence of its own.
- **The metrics disagree about the winner.** SkipLSTM/drrl has the highest raw
  AUC (0.6448) but ranks fourth on NDCG@5 — and its AUC sd is 0.0210, eight times
  PWTS's. Use this to justify having pre-registered NDCG@5 as primary rather than
  picking the metric that flatters a result.
- **Lead the user-facing claim with `RankFirstSkip`** — the rank of the first
  skipped track under the model's ordering, i.e. how many tracks play before the
  listener is interrupted. It is the only metric in the table stated in units a
  reader understands without explanation, and it is the quantity a queue reorderer
  exists to increase. Give it against the queue's own order, not only the random
  floor: "the listener reaches N tracks before a skip as the queue stands, and M
  once reordered" is the sentence the whole project is for.
- **MRR was removed** after it proved to span only 0.767–0.854 across every model
  including Random. Say that it was measured, found not to discriminate, and
  dropped — a metric you tested and rejected is worth more than one you never
  considered.
- Finish with the bootstrap column: which of these gaps survive pairing.

### 8.3 Loss functions (~300 words)

- **PWTS is best on both architectures** and the margin clears the threshold on
  the LSTM. Give the paired-bootstrap diff, 95% CI and p for PWTS vs BCE within
  each architecture.
- PWTS also has the **smallest seed variance** of the three losses on both
  architectures (0.0030 and 0.0018 vs DrRL's 0.0100 and 0.0111). Stability is a
  result, not a footnote — a loss that is reliably good is worth more than one
  that is occasionally better.
- **DrRL is worst, and unstable.** On SASRec it is 0.0207 below PWTS, far outside
  the threshold. Its seed sd is 3–8× the other losses'.
- Connect this to ch. 6: the two DrRL defects you found and fixed (β stepped by
  two optimisers at ~166× the intended rate; saturation on sigmoid outputs, fixed
  by moving to logits). Be careful with the claim — say the *implementation* was
  corrected and the arm still underperformed, not that DrRL is a bad loss. It was
  designed for a different setting (large candidate sets with sampled negatives),
  and your task has neither.
- **State the one caveat on the LSTM margin.** `SkipLSTM/bce`'s five seeds ran
  [45, 9, 15, 13, 7] epochs, and its grid shows the strongest longer-is-better
  relationship of the LSTM arms (+0.571; configs peaking ≤4 average 0.0058 lower).
  If that arm's mean is depressed by even 0.005, the LSTM PWTS-over-BCE gap of
  0.0065 largely closes. Read the LSTM margin as an upper bound — then note that
  the SASRec pairing, whose seeds all ran 30–55 epochs, is clean and points the
  same way (0.7805 vs 0.7772). The claim rests on SASRec; the LSTM corroborates
  it. Saying this yourself is stronger than having it put to you.
- One paragraph on **why PWTS might help**: it upweights early-session positions
  and near-instant skips, which is where a reordering system's errors are most
  visible to a user. Note this is a plausible mechanism you did not isolate — the
  ablation that would have tested it is confounded (→ 8.6).

### 8.4 Architecture (~250 words)

- Report the paired bootstrap for SkipLSTM vs SASRec at matched loss, all three
  pairs. **Do not report only the pair that favours your preferred conclusion.**
- The LSTM is ahead on the PWTS and DrRL pairings and roughly level on BCE. Note
  the direction is consistent, and that the DrRL pairing is the largest gap
  precisely because SASRec/drrl is the weakest arm — so it inflates any average
  taken over the three.
- **The block-mask asymmetry is the most important paragraph in this subsection.**
  SASRec's context/query mask removes its dependence on the realised order of the
  unplayed queue; the LSTM cannot implement it and therefore conditions on that
  order. The LSTM is solving a strictly easier problem. Any LSTM win is
  confounded by this, and you should say so before a marker does — a comparison
  you have correctly disqualified scores better than one you presented as clean.
- Interpretation: self-attention brought no benefit here. Offer the likely reason
  — sessions are short and the feature space is 15 continuous dimensions with no
  item embedding table, so there is little long-range structure for attention to
  exploit. Flag this as interpretation.

### 8.5 Feature ablation (~350 words)

Your most interesting result. Validation split, sklearn baselines only.

| model | features | pooled AUC | session AUC | session NDCG@5 |
|---|---|---|---|---|
| LogReg | audio (10) | 0.5060 | 0.5068 | 0.6963 |
| LogReg | behavioural (5) | 0.7082 | 0.5916 | 0.7531 |
| LogReg | all (15) | 0.7081 | 0.5942 | 0.7584 |
| ExtraTrees | audio (10) | 0.5499 | 0.5038 | 0.6938 |
| ExtraTrees | behavioural (5) | 0.7201 | 0.6051 | 0.7707 |
| ExtraTrees | all (15) | 0.7159 | 0.6029 | 0.7691 |
| Random floor | — | — | 0.5021 | 0.6960 |

- **Audio features alone are at chance.** Session AUC 0.5068 and 0.5038 against a
  0.5021 floor. Ten Spotify audio descriptors carry essentially no skip signal.
- The sharpest way to say it: **ExtraTrees on audio alone scores NDCG@5 0.6938,
  which is *below* the 0.6960 random floor.** Not "barely above chance" — at it.
- **Behavioural features carry all of the signal**, and **adding audio on top
  changes nothing**: 0.5916→0.5942 for LogReg, 0.6051→**0.6029** for ExtraTrees.
  The two models disagree in *sign*, and both differences are inside noise. That
  disagreement is the cleanest evidence that the audio contribution is zero
  rather than small — a real effect would move both the same way.
- **The pooled/per-session gap is a methodological finding in its own right.**
  Behavioural-only scores pooled AUC 0.7201 but session AUC 0.6051. Pooled AUC is
  inflated because it is rewarded for separating high-skip-rate sessions from
  low-skip-rate ones — a between-session distinction a queue reorderer never has
  to make. This justifies the per-session protocol in ch. 7 with evidence rather
  than assertion. Note the audio-only rows show almost no gap (0.5060 vs 0.5068),
  which is exactly what you would expect if the gap comes from
  `historical_skip_rate` proxying session-level base rate.
- **The caveat, stated by you not by the marker:** the sklearn baselines are the
  clean test. The neural arms receive observed skip outcomes through the
  status-embedding channel regardless of feature set, so a neural "audio-only" run
  is not audio-only in the same sense. Say this explicitly and cite it as the
  reason the ablation was run on the baselines.
- Discussion: set this against van den Oord et al. and Oramas et al. Your result
  does not refute content-based music recommendation — it says audio timbre
  descriptors do not predict *within-session skipping*, which is a different
  question from whether they predict *taste*. Making that distinction yourself is
  worth marks.

### 8.6 PWTS ablation (~250 words)

**Lead with the confound, not the ranking.** The ranking is not interpretable and
presenting it first invites the marker to read conclusions you then withdraw.

- 15 variants, 3 seeds each, SkipLSTM. Spread across the whole table is
  0.784–0.795 on validation NDCG@5 — about 0.011, roughly the resolution
  threshold. So the honest headline is: **PWTS is insensitive to its constants.**
- The confound, stated precisely: **the top 7 variants are exactly the 7 whose
  seeds all ran ≥15 epochs; the bottom 8 are exactly the 8 with a seed that
  stopped at 7 or 9.** The separation is perfect, with no overlap. The ranking is
  a report about early stopping firing, not about the loss constants.
- This matters because taken at face value the table says `no_time` and
  `no_historical` rank 1st and 2nd — i.e. that removing two of PWTS's three
  weights *improves* it. **That conclusion is not supported**, and saying so is
  the whole value of the subsection.
- Also note the variance signature that corroborates it: variants with all-long
  seeds have sd 0.0004–0.0018; those with a short seed have 0.0034–0.0045.
- What would fix it: a minimum-epoch floor before early stopping is allowed to
  trigger, or a fixed epoch budget with no early stopping. Put this in ch. 9.
- Do not delete the subsection. A designed experiment that returned an
  uninterpretable result, diagnosed correctly, is better evidence of competence
  than a clean result you did not stress-test.

### 8.7 Limitations (~250 words)

Enumerate, one or two sentences each. Being first to name these is worth marks.

- **The block-mask asymmetry** (→ 8.4) — the architecture comparison is not clean.
- **SASRec was epoch-capped during tuning**, so its selected configuration may be
  the one that trains fastest rather than the one that trains best.
- **`maxlen` was not enforced** at training time.
- **Patience interacts unevenly with the arms.** The effect is severe in the PWTS
  ablation (→ 8.6) but does *not* generalise to the headline table. Check before
  asserting otherwise: `SkipLSTM/drrl` peaked at epochs 1–4 across all five seeds
  yet scored 0.7891 on validation and the highest test AUC of any arm (0.6448),
  and two of the top ten configs in its 162-config grid also peaked at epoch 1.
  That is fast convergence, not truncation. The grid confirms it — LSTM/DrRL has
  the *weakest* best_epoch/NDCG correlation of any arm (+0.241, against +0.571
  for LSTM/bce and +0.783 for SASRec/drrl). Scope the limitation to where the
  evidence supports it (→ 8.3), rather than claiming it affects everything.
- **The neural feature ablation is confounded** by the status channel and was
  therefore not used for the headline claim.
- **The npz seed defect** (fixed, re-run required) — report it as an error found
  and corrected. The brief explicitly asks for error analysis.
- **Session-count mismatch**: the sklearn path scored 11,694 sessions to the
  neural path's 11,695, and `evaluate.py` warned rather than reconciled. Any
  baseline-vs-neural claim is therefore unpaired. Either recover the missing
  session or state that those comparisons are unpaired — do not quietly pair them.
- **Skips are ambiguous** — the ch. 1 caveat, now with the results behind it.
- **Single dataset, volunteer-sourced**, so no claim to generalisation.

### 8.8 Reflection (~200 words)

- What you would do differently, concretely: minimum-epoch floor; store per-seed
  per-session arrays from the start; fix the session-key reconciliation in
  `evaluate.py` rather than warning; run the feature ablation on a neural arm
  with the status channel disabled.
- What the deployability constraint cost, revisited now that the numbers exist.
- One paragraph on what the null results mean for practice: if audio features do
  not predict skipping, a production reorderer can be built from behavioural
  signals alone — which is cheaper, and survives the deprecation of Spotify's
  audio-features endpoint.
- Resist ending on apology. The project answered its question; the answer was
  partly negative.

### Figures to produce (none exist yet)

1. Bar chart, NDCG@5 by model with seed error bars and a horizontal Random line — 8.2
2. Grouped bars, feature ablation, session AUC with the 0.5021 floor marked — 8.5
3. Scatter, PWTS variant NDCG vs min epochs across seeds, showing the separation — 8.6
4. Pipeline dataflow diagram — ch. 5

## 9. Conclusions / Future Work (~650 words)

- Did the hypothesis hold? A project need not succeed to score well
- Each objective against its outcome
- Challenges: cluster access, the 742-task failure, compute constraints
- Future work: per-user standardisation; privileged-information distillation;
  raising the epoch caps; the neural feature ablation without the status
  confound
- What you would do differently

## References + Appendices (not counted)

- Harvard, alphabetical, everything cited in text
- **"Instructions for using the software/inspecting the code"** with the GitHub
  link — required, exact title
- **"AI tools and prompts used"** — required
- Full tuning tables, per-arm results, extended figures
- 5-minute demo video — optional but encouraged, and cheap marks

---

## Assets you already have

| Chapter | Artefact |
|---|---|
| 7, 8 | `Models/final_results.csv` — nine-row test table |
| 8.1 | `Models/{LSTM,sasrec}_grid_search_*_results.csv` — 1080 configs |
| 8.3 | `Models/test_log.csv` — 5-seed means and sds |
| 8.5 | `Models/feature_ablation_study.csv` — six rows, two models |
| 8.6 | `Models/pwts_experiments.csv` — 15 variants |
| 8.3 | `Models/eval_per_session.npz` — for paired significance tests |

## Still outstanding

- Paired bootstrap on the final results — needed before any loss claim
- Figures — none produced yet
- Session-count mismatch (12011/12014/12011) to reconcile before pairing
