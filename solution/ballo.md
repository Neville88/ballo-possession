# Frame-level possession from automatic tracking (technical document)

**Team:** ballo  
**Members:** Gareth Neville Kisuze, Ogwang Andrew, Katende Derrick Elvan.

## 1. Approach

We treat possession as a segmentation problem, not a per-frame classification. Every graded frame
receives a score for each of the three states: dead ball (`D`), home (`H`), away (`A`). And a
Viterbi pass picks the single label sequence that maximises the summed scores excluding a fixed cost
for every change of state:

- switching between `H` and `A` costs 10 frames' worth of clear evidence;
- entering or leaving `D` costs 15.

A touch or a deflection usually produces too little contrary evidence to pay for two switches, so
possession stays with the team that had it. An opponent who controls the ball for longer pays
for it, and possession changes where the evidence turns, not after a smoothing delay. Frames with
no ball carry no team evidence and cost nothing to keep in the current state.

We chose this over windowed majority voting (the baseline) because voting smears every boundary by
half its window and cannot distinguish a deflection from an interception. The ranking metric scores
whole possessions by temporal overlap, so boundary placement and avoiding spurious possessions
matter more than per-frame accuracy.

The path is decoded in two passes, then tidied (§2.5). The code is deterministic and rule-based,
apart from the dead-ball histograms in §2.2 and the unused model in §4.

## 2. Evidence

### 2.1 Who is on the ball

For each frame where the ball is tracked, we take the distance from the ball to the nearest home
and nearest away player, in metres. The evidence for the nearer team is
`tanh((d_away − d_home) / 1.5 m) · exp(−d_nearest / 2 m) · exp(−max(v_ball − 4, 0) / 6)`:
strong when one team is clearly closer and someone is on the ball, and fading while the ball
travels faster than 4 m/s.

### 2.2 Dead ball, learned from the automatic tagging

A naive-Bayes log-likelihood ratio over five signals (medians, stoppage vs open play):

| signal | stoppage | play |
|---|---|---|
| mean player speed, 2 s average | 0.97 m/s | 1.80 m/s |
| players above 4 m/s, 2 s average | 0 | 1 |
| fraction of the last 4 s with a tracked ball | 0.00 | 1.00 |
| ball distance to the nearest line | 3.2 m | 13.7 m |
| ball speed | 2.9 m/s | 5.4 m/s |

Each signal's likelihood ratio is a 24-bin histogram ratio between reference windows (§3), with
add-one smoothing and a ±3 clip per signal. The summed log ratio is halved because the signals are
correlated. The model is **cross-fitted by half**: each half is scored by a model fitted on the
other half's windows.

### 2.3 Situation rules from the tracking

| rule | what it detects | effect |
|---|---|---|
| **Ball out of play** | Ball tracked more than 1 m off the pitch. Over the goal line, dead until it has settled for 1 s and is then kicked above 8 m/s for 3 frames. Over the touchline, dead until it is back inside the width above 3 m/s for 3 frames. Capped at 45 s, falling back to the last off-pitch frame plus the settle time. | +4 per frame to `D` |
| **Ball lost at a line** | Ball disappears within 3.5 m of a line for at least 5 s, and is not a keeper hold. | +3 per frame to `D` |
| **Keeper hold** | Ball vanishes for 1–25 s, last seen inside a team's own box, and its path projected over 0–0.5 s comes within 3 m of that team's keeper during the next second. | +0.5 to his team and −0.5 to `D` over the gap |
| **Keeper release** | After such a gap that was not a hold and was not lost within 3.5 m of a line, the ball's path back-projected over 0.5 s from where it reappears comes within 3 m of a keeper in his own box. | Same as a hold |
| **Restart in flight** | From 2 s before to 3 s after the first pass leaves a stoppage, when a player was clearly on the ball inside that window (ball under 4 m/s, within 1.5 m, nearer than the other team by 1.5 m). | In a second pass, +0.5 per frame to that team while the ball travels above 4 m/s, for up to 3 s after that control |

### 2.4 Smart Tagging

- **Phases.** In-possession and defending phases vote once per frame, and the majority side gets a
  small bonus (0.3). They are not stacked: stacked, they outweighed clear tracking evidence for
  10–15 s after a turnover.
- **Set pieces.** A set-piece tag is about 35 s long and **opens at the stoppage**: averaged over
  all 59 tags, player speed bottoms out and the ball disappears from about 4 s before the tag start
  to 4 s after, and play resumes 6–8 s in. We add dead-ball evidence from −4 s to +4 s, and evidence
  for the named team from +8 s to +20 s.
- **Goals.** The goal tag spans the attack and the celebrations, so the goal moment is the last
  frame inside the tag with the ball seen in a penalty box. Frames from there to the following
  kickoff are forced to `D`, if that kickoff is under 5 minutes away.
- **Kickoffs.** Frames from each half's first graded frame to its tagged kickoff are forced to `D`.

### 2.5 Tidying the path

Applied after decoding, within each half:

- A stoppage under 3 s with the same team on both sides becomes that team's possession, unless it
  falls in a set-piece dead window or a forced `D` stretch. Fetching the ball and restarting takes
  longer than that.
- Play under 2 s between two stoppages becomes `D`.
- A spell between two stoppages, with the ball seen on over half its frames and never above 8 m/s
  for 3 frames, becomes `D`. A restart is a kick or a throw.

## 3. How versions were chosen

**Public five minutes.** Used only to confirm the pipeline is correct. We never compared two
versions on it to choose between them.

**Reference windows.** Built from the automatic Smart Tagging, which covers all 106 graded minutes:

- **dead:** ±3 s around each set-piece tag start;
- **alive:** frames inside tagged phases, at least 8 s from any set piece, with the tagged team as
  the team in possession;
- **turnovers:** 2–6 s after each Attacking Transition tag, where the tagged team should have it.

**The tags are not truth.** On frames where one player is within 1 m of a slow ball and the nearest
opponent is more than 2 m further away, the tagged team agrees with that player's team only 80% of
the time.

**Watching the video.** The algorithm never reads the video, and no notes are an input to it. A team
member watched the tactical video next to a list of the model's segments
(`solution/review.py`) to find where it failed. This changed during the work:

- **First half.** The notes were descriptions of specific moments: what happened, where
  the model looked wrong, and sometimes which team had the ball. They were not frame-by-frame
  `H`/`A`/`D` labels. We used them to identify kinds of failure and design general rules. We also
  used them to check whether a rule fixed the moments described, and they influenced some limits and
  weights, including the 25 s keeper hold, the 8 m/s and 3 m/s restart speeds and the ball-out weight.
- **Second half.** The notes became summaries of error patterns. Rules were kept or dropped on
  football reasoning and the reference windows above, on each half separately, and not by counting
  against the summaries.

The notes are not in this repository.

## 4. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | all tracking evidence |
| Smart Tagging | evidence (§2.4) and the reference windows (§3) |
| tactical video | not read by the algorithm; watched to find failures (§3) |
| tactical homography | not used: the ATD is already in pitch coordinates |
| broadcast video, ATD, homography | not used: not synchronised to the tactical timeline, and a third of it has no usable homography |

**Tried and dropped:**

- **Team-shape model.** A logistic regression on shape features of both teams, trained on the tagged
  phases and cross-fitted by half. It lags turnovers by seconds while teams reorganise. It remains in
  `teammodel.py` with weight 0.
- **Crediting the passer's team on every ball in flight.** Tagged turnovers fell from 21 of 24 to 19
  of 24, so we limited it to restarts.
- **A still ball struck hard ends a stoppage.** Only 57 of 1,193 kicks come from a ball tracked as
  still, and the rule had no effect.
- **A keeper with the ball at his feet keeps it.** Turned tagged stoppages into play for little gain.
- **Detecting fouls from a still ball.** Tracked ball jitter and false triggers.
- **Goal moment from the ball crossing the goal line.** The ball is often untracked at that moment.

## 5. Training data

No labelled dataset was used to train anything, from this match or any other. The only fitted
components (the dead-ball histograms, and the unused team-shape model) are fitted on the automatic
Smart Tagging of the provided match, cross-fitted by half. No private ground truth and no event data
were used. For how notes from watching the video were used, see §3.

## 6. Possession rules implemented

- **Loose and contested balls** stay with the team that last had control unless the evidence pays
  for a switch.
- **Change of possession** needs about 10 frames (0.4 s) of clear control evidence in excess of the
  previous team's.
- **Passes in flight** carry little evidence, so a completed pass does not split a possession.
- **Challenges** are not modelled directly. The nearest player gets the evidence, so a challenge can
  read as a turnover.
- **Restarts** belong to the taker's team while the kick is in flight when the taker is seen, and to
  the team a set-piece tag names once play resumes.
- **Keepers** keep possession through a detected catch or hold, even though the slow tempo looks like
  a stoppage.
- **Dead ball** runs from the stoppage to the restart, and ownership of a dead ball is not modelled.

## 7. Results

**Public five minutes (pipeline check only):**

| | spellF1 | macroF1 |
|---|---|---|
| baseline_smooth50 | 0.3256 | 0.6207 |
| baseline_smooth125 (best baseline) | 0.4242 | 0.5812 |
| **ours** | **0.5091** | **0.7770** |

The public window has 17 possessions and the README shows it ranking two baselines in the wrong
order, so it does not show how much better a method is. 

**Full match against the Smart Tagging windows:**

| | dead | alive | team | turnovers | possessions | mean length |
|---|---|---|---|---|---|---|
| baseline_smooth50 | 0.583 | 0.764 | 0.505 | 0.792 | 669 | 7.6 s |
| ours, tags switched off | 0.866 | 0.712 | 0.491 | 0.708 | 260 | 13.8 s |
| **ours, submitted** | **0.921** | **0.734** | **0.589** | **0.875** | **245** | **14.9 s** |

The submitted model reads the same tags, so its row is flattered. The `team` column counts frames
labelled `D` as misses; among tagged open-play frames the submitted model labels as play, it agrees
with the tagged team about 80% of the time, in line with the tags' own limit in §3.

An earlier version scored higher on `alive` (0.815) and `team` (0.740), and had 215 possessions. It
stacked the phase tags, which made it follow them for 10–15 s after real turnovers. We accepted the
lower agreement in exchange for better timing and stoppage detection, without being able to measure
the effect on the private spellF1.

## 8. Limitations

- **Fouls, offsides and advantage.** There is no whistle signal, so these stoppages often start late,
  and the gap can be filled with possession, sometimes for the team that fouled. Quick free kicks can
  be missed entirely.
- **Quick throw-ins and free kicks** can look like play to every signal.
- **Long pauses** such as substitutions and set-ups can still contain fake possession when a kick is
  tracked during the pause.
- **Keeper with the ball.** Holds outside the detected patterns can be read as a stoppage, or given to
  an opponent nearby.
- **Headers and deflections.** The nearest player gets the evidence, so possession can switch early
  on a contact, or late after a clearance is won in the air.
- **Crowded boxes.** The nearest player is a weak proxy for control in a crowd.
- **Line crossings.** Stoppage starts around shots and balls near a line can be early or late,
  depending on how well the ball is tracked there. A ball drifting just over a line in tracking can
  create a stoppage that isn't real.
- **Short tidying thresholds** can remove a real, very quick stoppage.

With more time: a ball-control model trained on labelled touches from other matches; audio-based
whistle detection; and the broadcast feed for moments the tactical tracking loses the ball.

## 9. Reproduction

```bash
pip install -r solution/requirements.txt          # numpy 2.3.3, pandas 2.3.3; Python 3.13.1
# unpack the match data into data/ (see DATA.md)
python solution/run.py --team ballo --check
```

`--check` rebuilds twice, once from the raw tracking, and confirms the two files are byte-identical
before running the organisers' validator. There is no randomness anywhere in the pipeline. The
submitted file's SHA-256 is
`06d72eb684cab9f7d09ab084e268c782c7d528fb1ca62751f7aa9dad71684da3`.
