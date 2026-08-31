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

## 1. Introduction (~700 words)

Write this chapter **last**. Every figure in it has to match the final results
table, and the framing only settles once you know what the results say.
Six paragraphs, budgeted below.

### 1.1 The problem (~120 words)

- Open on the concrete situation, not the field. A listener queues an album or a
  playlist on shuffle; the order is arbitrary; some tracks are skipped within
  seconds. **Do not open with "with the rise of streaming services" — the marker
  has read that opening forty times.**
- Streaming research overwhelmingly asks *what to play next*. Far less asks *in
  what order to play what the user has already chosen.*
- Frame queue reordering as a **ranking** problem, not a retrieval one: the
  candidate set is fixed, small, and already selected by the user.
- Draw the consequence immediately, because it justifies the architecture in
  ch. 5: there is no catalogue-scale cold start, and **no need for an item
  embedding table** — you only ever rank tracks already in hand.
- One sentence on why that is worth doing: the same listening session, reordered,
  produces fewer skips without adding or removing a single track.

### 1.2 Skip behaviour as the signal (~140 words)

- Explicit feedback — likes, saves, ratings — is sparse. Most listening produces
  no explicit signal at all.
- A skip is abundant, automatic and costless to collect: every session generates
  them, and the user does nothing extra.
- But it is **ambiguous**. A skip can mean dislike, wrong mood, heard-it-already,
  or an interruption with nothing to do with the track. One sentence here; it
  returns properly as a limitation in ch. 8. Do not bury it — a marker who spots
  the ambiguity before you name it reads the whole report as naive.
- State your operational definition of `skipped` in plain words (one sentence);
  defer the exact rule to ch. 4.
- **Name the random floor in the introduction.** Because most tracks are not
  skipped, random ordering already scores NDCG@5 ≈ 0.696 and MRR ≈ 0.772. Giving
  the floor up front reframes every later number and stops the marker reading a
  headline 0.77 as impressive. This is the single most important sentence in the
  chapter for the Evaluation marks.

### 1.3 The deployability constraint (~140 words)

*This is the project's distinctive move. Give it its own paragraph and claim it.*

- The rule in one sentence: **a feature may be used only if its value is knowable
  for a track that has not yet been played.**
- Two features were excluded by it — `actively_selected` and `reason_start`. Both
  describe how a track's playback *began*, so both exist only after the fact.
- Why it matters: a model using them scores well offline and cannot be deployed.
  Much of the skip-prediction literature, the WSDM Cup 2019 framing included,
  predicts skips for tracks that are *already playing* — a different task with a
  different information set.
- **State the cost honestly.** The constraint lowers the achievable ceiling. That
  is the point: you traded offline score for a system that could actually run,
  and you should present that trade as a design contribution rather than
  apologise for the resulting numbers.
- One sentence on what the constraint does *not* forbid: skip outcomes for tracks
  already played *earlier in the same session* are legitimately available at
  prediction time, and both neural models use them.

### 1.4 What was built (~100 words)

- Four model families in one sentence: logistic regression, extremely randomised
  trees, an LSTM, and a SASRec adaptation without item embeddings.
- Three loss functions across the two neural architectures — BCE, PWTS
  (position/time/history weighted), DrRL — giving six trained arms plus two
  baselines and a random floor.
- Scale, in one clause, because it evidences the Implementation marks: 1,080
  tuning configurations, five seeds per final arm, run on the departmental SLURM
  cluster.
- Evaluation in one sentence: per-session, at up to five evenly spaced split
  points, averaged within session before across sessions.
- Resist listing every metric here. NDCG@5 primary, AUC secondary; the rest
  belongs in ch. 7.

### 1.5 What was found (~130 words)

State the three findings plainly, one sentence each, with numbers.

- **Audio features alone are at chance.** Session AUC 0.507 (logistic regression)
  and 0.504 (extremely randomised trees) against a random floor of 0.502. Ten
  Spotify audio descriptors carry essentially no skip signal in this data — and
  ExtraTrees on audio alone scores NDCG@5 0.694, *fractionally below* the 0.696
  random floor. Worth naming: it is a cleaner result than "slightly above chance".
- **Behavioural features carry the signal.** Five behavioural features reach
  session AUC 0.592 / 0.605 and NDCG@5 0.753 / 0.771. Adding the ten audio
  features on top changes nothing (0.594 / 0.603) — inside seed noise.
- **Architecture matters less than expected.** The LSTM and SASRec are
  statistically indistinguishable under a paired bootstrap over per-session
  scores; PWTS is the best loss on both.
- Add one sentence on why null results are the strongest part of the report: they
  contradict a standing assumption (that content features drive music taste), and
  they come from an ablation you designed deliberately — not from a failure being
  explained after the fact.
- **Do not oversell.** The brief rewards honest evaluation.

### 1.6 Comparable systems (~90 words — brief, detail in ch. 3)

One sentence each, then one sentence on the gap.

- **Spotify smart shuffle / autoplay** — the commercial comparator. Reorders and
  extends a queue, but is proprietary and its objective function is unpublished,
  so it cannot be benchmarked against.
- **WSDM Cup 2019 Spotify Sequential Skip Prediction** — the academic comparator.
  Same signal, different task: predict skips within a session as it unfolds, with
  no reordering and no deployability constraint.
- **Automatic playlist continuation** (RecSys Challenge 2018) — the adjacent task.
  *Extends* a playlist with new tracks; this project *reorders* tracks the user
  already chose.
- The gap in one sentence: none of them convert a skip prediction into a queue
  order and then measure the ranking quality of that order.

### 1.7 Roadmap (~80 words)

- One sentence per chapter, written as prose. **Not a bulleted list** — a
  bulleted roadmap reads as filler and costs Presentation marks.
- Keep it factual and specific: "Chapter 4 describes the data pipeline and the
  chronological per-user split; Chapter 5 sets out three design decisions that
  follow from the deployability constraint; …"

### Before you write this chapter

- Reconcile the **session-count mismatch** (12,011 / 12,014 / 12,011) before
  quoting any exact figure. If it is a `dropna` interaction, say so in ch. 4 in
  one sentence rather than leaving a marker to notice the inconsistency.
- Every number here must match ch. 7 exactly, including rounding.
- Keep citations to one or two at most — the introduction is not the literature
  review.
- Do not state a hypothesis you do not go on to test. If you frame the project as
  testing whether audio features help, the ablation must be presented as the test
  of it, not as a side experiment.

## 2. Aims and Objectives (~400 words)

- Aim: rank unplayed queue tracks by predicted skip probability
- Objectives as workpackages with measurable outcomes — data pipeline, four
  models, three losses, evaluation protocol, feature ablation
- **State how success is measured**: per-session NDCG@5 as primary, AUC
  secondary, against the random-ordering floor
- Reviewers check objectives against outcomes, so phrase them so each maps to a
  results subsection

## 3. Background / Literature review (~1,600 words)

*Content and reference selection are yours. Structural notes only:*

- The brief asks for **critical analysis**, not summary — identify gaps and use
  them to justify your design decisions
- Address any marker feedback on the proposal
- Each design choice in ch. 5 should be traceable to something argued here
- Verify every reference against the publisher before it goes in

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
  `max_depth=8` rows, with (8, 20, 100) best at 0.7666 — but `Random_Forest.py`
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

## 6. Implementation (~1,000 words) — 30% of marks

- Selected points only, not a tour of the code
- Worth covering: multi-point evaluation; `fast_auc`; length-bucketed batching
  (6.23× → 1.01× padding, 3.2× faster); SLURM job arrays for 1080 configs
- Two defects found and fixed, both good material: DrRL's `beta` stepped twice
  by two optimisers at ~166× the intended rate; DrRL saturating on sigmoid
  outputs, fixed by moving to logits
- Point to the appendix for detail

## 7. Testing and Evaluation (~1,300 words) — part of 30%

- The brief wants **systematic** experimental design, not assorted results
- The evaluation protocol and its justification: per-session vs pooled, and the
  ~30.9% between-session variance that motivates it
- Multi-point splits, and why single-point sampling was inadequate
- Metric definitions, including NDCG@5's ascending-by-skip-probability ordering
- **The random floor** — and MRR's saturation at 0.767
- Experimental design: tuning → training (5 seeds) → held-out test
- Noise floor and the resolution threshold you will apply

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

| model | AUC | NDCG@5 | NDCG@10 | MRR | P@5 |
|---|---|---|---|---|---|
| SkipLSTM/pwts | 0.6445 ±0.0025 | **0.7865** ±0.0030 | 0.8226 | 0.8540 | 0.6964 |
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
- **MRR is nearly uninformative here.** It moves only from 0.7671 to 0.8540 across
  the entire table — the base rate compresses it. Say so; reporting a metric and
  then explaining why you do not rely on it reads as rigour.
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
