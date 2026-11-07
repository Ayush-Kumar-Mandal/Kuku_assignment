# How this project works, phase by phase

## The big picture

**The problem (Kuku assignment, Problem 1):** an editor watches every episode of a microdrama
and picks promo clips by hand. This tool does that job from the screenplay instead.

**What a good promo is:** a 20–60 second clip that:
- **opens on a hook**, something in the first 3 seconds that stops a scrolling viewer,
- **ends on a cliffhanger**, so the viewer wants to know what happens next,
- **makes sense on its own** to someone who has never seen the show.

**What the tool does:** it reads all 50 screenplay episodes, finds strong moments, cuts them
into promos, scores the promos and ranks them. A human editor then approves or fixes them,
and the tool remembers what the editor liked for next time.

```
 screenplays (.txt)
        │
        ▼
   ┌─────────┐   ┌──────────┐   ┌────────┐   ┌────────┐   ┌───────────┐   ┌──────┐
   │  Parse  │ → │  Reader  │ → │ Cutter │ → │ Joiner │ → │ Evaluator │ → │ Rank │
   └─────────┘   └──────────┘   └────────┘   └────────┘   └───────────┘   └──────┘
                 finds moments  1-episode    multi-ep      scores 1–10      top 20
                                promos       promos        weak → redo        │
                                                                              ▼
                                       ┌────────────┐            ┌────────────────┐
                                       │   Memory   │ ← learns ← │ Editor review  │
                                       └────────────┘            └────────────────┘
                                     (used in the next run)      approve/reject/fix
```

The project was built in three phases:

| Phase | What it adds |
|---|---|
| **Phase 1** | The core promo finder: read the scripts, find promos, score, rank, run, measure quality |
| **Phase 2** | An editor in the loop: a human approves, rejects or asks for fixes |
| **Phase 3** | Long-term memory: the tool learns the editor's taste across runs |
| **Phase 4** | Better evaluation: an independent AI labeller, an agreement check, and a made-up show with known answers |

### Folder map

| Folder | What's inside |
|---|---|
| [input/](input/) | The 50 screenplay episodes (the input pack), plus `input/synthetic/` for made-up test shows |
| [parser/](parser/) | Turns raw screenplay text into clean data |
| [config/](config/) | All settings in YAML files, with no hard-coded numbers |
| [llm/](llm/) | The code that talks to AI models (Groq, Gemini) |
| [agents/](agents/) + [prompts/](prompts/) | The AI "workers" and their instructions |
| [graph/](graph/) | The pipeline that connects every step (LangGraph) |
| [ranking/](ranking/) | Picks and orders the final promos |
| [memory/](memory/) | Remembers editor feedback between runs |
| [evaluate/](evaluate/) | Measures how good the picks are and what they cost |
| [app/](app/) | The Streamlit web app |
| [run.py](run.py) | Runs everything from the command line |
| [tests/](tests/) | Automated tests |
| [output/](output/) | Results of runs (`output/runs/`) and evaluations (`output/eval/`) |

---

## Phase 1: The core promo finder

**Goal:** given a folder of screenplays, produce a ranked list of promos with exact start and
end timestamps, a hook, a cliffhanger and a line on why each one works.

Phase 1 was built in 8 small steps.

### Step 1: Read the scripts
Each episode is a text file. Every line of dialogue or action has a timestamp:

```
**MAYA (Angry):**
Start: (00:04.00) | End: (00:07.50)
"How could you do this to me?"
```

- The **parser** reads each file line by line and turns it into clean data. Each line becomes
  one item with who speaks, what they say, and its start and end time in seconds.
- The patterns it looks for live in [config/parser.yaml](config/parser.yaml). If the script
  format changes, you edit that file, not the code.
- The parser also builds a list of **every real timestamp**. Later, any cut point the AI invents
  that isn't on this list is rejected.
- Output: one JSON file per episode in `output/parsed/`.

Key files: [parser/screenplay_parser.py](parser/screenplay_parser.py), [parser/models.py](parser/models.py)

### Step 2: Keep all settings in one place
[config/settings.yaml](config/settings.yaml) holds every rule and number:
- Promo rules: 20–60 s long, hook in the first 3 s, cliffhanger in the last 10 s.
- Which AI model each agent uses, plus backup models.
- Score weights, ranking limits, retry counts and so on.

**Swapping the model is easy.** You change one line in the config, or pass a flag such as
`--model reader=groq:openai/gpt-oss-120b`. The brief asked for this.

Key file: [config/loader.py](config/loader.py). It checks the settings at startup, so a typo
fails right away.

### Step 3: Talk to AI models safely
Every AI call goes through one function, `LLMClient.structured()` in [llm/client.py](llm/client.py).
- It works with **Groq** and **Google Gemini**, both on free tiers.
- The AI must answer in a **fixed structure** (JSON). If the answer is malformed, the tool sends
  the errors back and asks again, up to 3 times.
- **Fact checks** ([llm/checks.py](llm/checks.py)) catch mistakes. Made-up timestamps, cuts in
  the middle of a sentence and promos that are too short or too long are all rejected.
- When a free quota runs out, it **switches to a backup model**. If no backup is left, the run
  pauses safely and can be resumed later.
- Every call is logged to `logs/llm/`, with tokens counted.

### Step 4: The AI team (agents)
Each agent is a separate AI worker with one job. Their instructions live in [prompts/](prompts/).

| Agent | Job |
|---|---|
| **Reader** | Reads episodes in order and marks up to 3 hooks and 3 cliffhangers per episode. It keeps a running story summary. |
| **Cutter** | Turns one episode's moments into up to 2 promos, each one continuous clip. |
| **Joiner** | Stitches 2–4 clips from different episodes into up to 6 multi-part promos. |
| **Evaluator** | Scores each promo 1–10 for hook, cliffhanger and standalone. It sees **only what a viewer would see**, with no show name or plot summary, so "makes sense on its own" is tested fairly. |

The code computes ids, durations and timestamps itself; it never trusts the AI with those.

There are also **stub agents** ([agents/stubs.py](agents/stubs.py)). They are fake agents that
need no API key and cost nothing. They are used for dry runs and tests.

Key file: [agents/llm_agents.py](agents/llm_agents.py)

### Step 5: Connect it all as a pipeline (the graph)
[graph/](graph/) wires the steps together with LangGraph. The diagram is in [docs/graph.md](docs/graph.md).
- **Revision loop:** a promo passes only if its score is at least 7.0 and every individual score
  is at least 6. A weak promo goes back to the Cutter or Joiner with a fix suggestion, up to
  2 times. After that, the best version is kept.
- **Checkpoints:** progress is saved after every step to SQLite (`checkpoints/`). If the run stops
  because of the internet, a quota or Ctrl+C, `--resume` continues where it left off.

Key files: [graph/nodes.py](graph/nodes.py), [graph/routing.py](graph/routing.py)

### Step 6: Rank the promos
[ranking/rank.py](ranking/rank.py) picks the final list.

1. **Score:** `final = 0.4 × hook + 0.35 × cliffhanger + 0.25 × standalone`. The hook counts most
   because it decides whether people stop scrolling.
2. **Small bonus for early episodes:** a promo from episode 1 leads viewers to the start of the
   show. The bonus is at most +15%.
3. **Remove duplicates:** if two promos share at least 50% of their screen time, only the better
   one is kept.
4. **Limits:** at most 2 promos per episode and 20 in total. Anything scoring below 6.0 is dropped.

Every dropped promo is recorded with the reason, so nothing disappears silently.

### Step 7: Run it (command line and web app)
- **Command line:** [run.py](run.py). See "How to run" below.
- **Web app:** [app/streamlit_app.py](app/streamlit_app.py). Start a run, watch its progress live,
  browse the ranked promos and see the full history ("trace") of any promo. The **Evaluation**
  page shows the quality reports from Phase 4, and the **Memory** page shows the editor's rules.
- **Outputs** go to `output/runs/<run_id>/`:
  - `promos.json` / `promos.csv`: the ranked promos (episode, segment timestamps, hook,
    cliffhanger, why it works)
  - `report.md`: an easy-to-read report
  - `raw/`: every moment and every promo, including the ones that were dropped
  - `traces/`: the full history of each promo

### Step 8: Check quality and cost
[evaluate/](evaluate/) answers the question "is it actually good?"
- **Blind labels** ([evaluate/labels.py](evaluate/labels.py)): good moments are marked *before*
  looking at the tool's picks.
- **Recall funnel** ([evaluate/compare.py](evaluate/compare.py)): for each labelled moment, it
  checks whether the Reader noticed it, whether a promo was made, and whether it made the final
  list. This shows *where* good moments get lost.
- **"Would you post it?"** ([evaluate/judge.py](evaluate/judge.py)): a yes/no for each pick. A "no"
  must give a reason, such as "needs earlier context" or "cuts mid-action".
- **Cost** ([evaluate/cost.py](evaluate/cost.py)): turns real token counts into the cost per show
  and per week, on free and paid tiers.

**Honest result** from the final 50-episode run, with editor review and memory on
([output/eval/final-50/evaluation.md](output/eval/final-50/evaluation.md)):
- In review, the editor approved 8 of the first 20 picks (40%). The 12 back-filled promos were all
  approved.
- Judged again afterwards, more slowly, 12 of 19 final picks would be posted (**63%**). The 7 "no"s:
  hook only works visually 2, needs earlier context 2, cuts mid-action 1, resolves too much 1,
  other 1. So a quick approval in review is not the same as "I'd post it".
- Against the first labels (made by Claude in chat: 8 moments in 5 episodes), the Reader noticed
  all 8 and promos were made for 5, but **none made the final list**.
- Against the Gemma labels from Phase 4 (10 moments in episodes 2–9), the Reader noticed 9, promos
  were made for 7, and **only 2 made the final list**.
- Of the 10 labelled moments that got a promo but missed the list: the editor rejected 4, 3 fell
  below the top-20 cut, and 3 scored under the 6.0 minimum. Good moments are lost **after** the
  promo is made.
- Neither label set was made by a human editor, so treat these numbers as a rough guide.
  Phase 4 explains how to check them.

---

## Phase 2: The editor in the loop

**Goal:** keep a human in control. The AI suggests, and the editor decides.

- After ranking, the run **pauses** and waits for the editor.
- For each promo the editor picks one of these:
  - **Approve**: it stays in the list.
  - **Reject**: it's removed, and the next-best promo takes its slot.
  - **Fix + comment**: for example "start 2 seconds earlier, the slap is cut off".
  - **Keep before**: undo a fix if the new version is worse.
- A **feedback agent** (`feedback_fix`) turns the editor's plain words into a precise instruction
  with real timestamps. The promo is then remade, re-scored, and shown **before vs after**.
- Limits: at most 5 review rounds and 2 fixes per promo, so the loop always ends.
- The editor can review in the web app's **Review tab** or in the terminal (`--review`).

Key files: [app/review.py](app/review.py), the `editor_review` and `feedback` steps in [graph/nodes.py](graph/nodes.py)

---

## Phase 3: Long-term memory

**Goal:** the tool gets better with every show by learning what the editor likes.

- Every editor decision is saved in a small database ([memory/store.py](memory/store.py), file
  `memory/memory.db`). The feedback log can only be added to; it can't be edited or deleted.
- A **learning agent** (`feedback_learn`) looks at the feedback and suggests **rules**, such as
  "don't end on a resolved argument".
  - A rule needs at least **2 pieces of feedback** behind it.
  - New rules **wait for the editor to switch them on** in the web app's Memory page. The AI
    can't quietly change its own behaviour.
- Each rule has a **confidence** score: `(supports + 1) / (supports + contradicts + 2)`.
  A rule that keeps getting contradicted is flagged for review.
- In the next run, the top 8 rules and a few approved and rejected examples are added to the
  agents' instructions. Rules can apply to **this show**, **this genre** or **all shows**.
- To check whether memory helps, the approval rate is compared between runs with memory and
  runs without it ([evaluate/runs.py](evaluate/runs.py)).

---

## Phase 4: Better evaluation (checking the checker)

**The problem:** quality is measured against **labels**, a list of the moments a good editor would
promo. The first labels were made by Claude in chat: only 8 moments, made once by hand, and never
checked against a real editor. If the labels are wrong, every number built on them is wrong.

**The idea:** use two different AIs, one to make labels and one (the tool) to be checked. Then check
the labeller itself on a show where the right answers are known for certain.

```
  made-up show with planted moments ──► answer key (exact)
          │                                   │
          ├──► the tool ──────► compare ◄─────┤   how good is the tool?
          │                                   │
          └──► AI labeller ───► agreement ◄───┘   can we trust the labeller?
                    │
                    └──► labels for the real show (once trusted)
```

### Step A: Pick a labeller from a different model family
- The tool already uses Gemini, gpt-oss and qwen. **Gemma 4** (`gemma-4-31b-it`) is the only free
  model family it doesn't use, so it is the labeller. That way the tool is not graded by itself.
- It is set in `models.labeler` in [config/settings.yaml](config/settings.yaml), with
  `gemma-4-26b-a4b-it` as the backup.

### Step B: The AI labeller
[evaluate/ai_labels.py](evaluate/ai_labels.py), with instructions in [prompts/labeler.yaml](prompts/labeler.yaml).
- For each episode it sees exactly what a human labeller would: the screenplay lines, the episode
  logline and the loglines of earlier episodes. It **never sees the tool's picks**.
- It marks every strong 20–60 s stretch (up to 4 per episode), with a type and a reason why.
- The same fact checks as the agents apply. A start or end time must be a real line boundary, the
  length must fit, and moments can't overlap. Any problem is sent back to the AI to fix.
- It writes a separate file, `output/eval/labels/<show>-ai.yaml`, so it can never overwrite human
  labels. The file records that it is AI-made, which model made it and which prompt version was
  used, and it is locked when finished.
- If the labeller used a model that also ran in the tool, the report **warns** that the tool was
  partly graded by itself.

### Step C: The agreement check
[evaluate/agreement.py](evaluate/agreement.py) compares two label files: a trusted **reference**
(human or planted) and a **candidate** (the AI labeller).
- **Recall:** how many of the reference moments the AI also found.
- **Precision:** how many of the AI's moments are in the reference.
- Two moments match when they share at least 50% of the shorter one's screen time.
- **Verdict:**
  - "trust it on the other episodes" needs recall ≥ 70% and precision ≥ 50%;
  - "don't rely on it" otherwise;
  - "too few reference moments to say" below 10 reference moments.
- The thresholds are in `evaluation.agreement` in the settings.

### Step D: A made-up show with planted answers
[evaluate/synthetic_show.py](evaluate/synthetic_show.py), with settings in [config/synthetic.yaml](config/synthetic.yaml).
1. An AI writes a **show bible**: the premise, the characters and one beat per episode.
2. **The code (not the AI) picks the answers:** 1–2 moments per episode, each with a type
   (betrayal, reveal, romance…) and a position. The choice is random but seeded, so it can be
   repeated.
3. The AI writes each episode as plain script lines and tags the planted lines `[P1]`. The rest
   of the episode is kept ordinary on purpose.
4. **The code adds every timestamp** from word counts and writes the files in the exact screenplay
   format.
5. Checks before anything is saved:
   - each planted moment is 20–60 s long and opens on dialogue or action;
   - every file passes the real parser at 100%.
6. The answer key is saved as `output/eval/labels/<show>.yaml`, marked "planted" and locked.

What the answer key does and doesn't tell you:
- Recall against it is **exact**, because those moments were planted by design.
- Precision is only **approximate**, because the writer may also write strong moments it wasn't
  asked for.

### Step E: Results
The made-up show "The Glass Heir" has 8 episodes and 13 planted moments.

**The tool vs the answer key** ([output/eval/glass-heir-e1/evaluation.md](output/eval/glass-heir-e1/evaluation.md)):

| Stage | Found | Rate |
|---|---|---|
| The Reader noticed it | 11/13 | 85% |
| A promo was made for it | 11/13 | 85% |
| It made the final top list | 8/13 | 62% |

- 8 of the 9 ranked picks are planted moments. The 9th wasn't planted, but the AI labeller marked
  the same stretch too, so it is probably a real strong moment the writer added.
- Ranking is again where most moments are lost. Two of the five misses were quiet romance scenes.
- **The models overlapped.** Some of the tool's models also helped write the show, so the report
  marks recall as an upper bound.

**The AI labeller vs the answer key** ([output/eval/agreement/the-glass-heir-ai__vs__the-glass-heir.md](output/eval/agreement/the-glass-heir-ai__vs__the-glass-heir.md)):
- Recall **92%** (12/13), precision **86%** (12/14). The verdict is "trust it on the other episodes".
- Its only miss was a romance moment that the tool also missed. That moment doesn't end on a
  cliffhanger, so the writer probably wrote it weakly.

**On the real show,** Gemma was compared with the Claude labels on episodes 14–20: recall 62%
(5/8), precision 83%. That is too few moments for a verdict, and both sides are AI. Gemma missed
the quiet romance and the subtle hints.

**What this means:** the made-up show is easier than the real one, because everything outside the
planted moments is deliberately ordinary. So the 92% is a best case. The real test is to compare
the labeller with **a human editor's labels** on 3–5 real episodes, and that step needs a person.

---

## Phase 5: Refining the prompts from the measured failures

**Goal:** aim each prompt at the mistakes the final run (`final-50`) actually made, not at
general advice. The final run itself used the version 1 prompts; everything below came after it.

| Measured in `final-50` | Prompt change (version 2) |
|---|---|
| Wrong length: 54 of 99 retries | `cutter`: a 5-step method (hook → cliffhanger → end time inside the given window → length check aiming for the middle of the range → stands-alone check) |
| Editor: "needs earlier context" (2), rejected the cross-episode promo | `reader`, `cutter`, `joiner`: prefer clips where names and relationships are said; in a joined promo each clip must follow from the one before |
| Editor: "hook only works visually" (2) | `reader`, `cutter`, `evaluator`: the hook must land in spoken words |
| Editor: "resolves too much", "cuts mid-action" | `cutter`, `joiner`, both revise prompts: named as red flags |
| Evaluator scores barely matched the editor (rejected 8.1s, approved 6.85s) | `evaluator`: what a 1–3, 4–6, 7–8 and 9–10 looks like for each score, plus the editor's red flags |
| 12 rejects without comments taught memory nothing | `feedback_learn`: look for what uncommented rejects have in common |

**Test** (episodes 1–5, review and memory off, old vs new prompts; runs `prompts-v1-ep1-5`
and `prompts-v2-ep1-5`):
- **Same judge, better promos:** scored by the old Evaluator prompt, the new prompts' 6 ranked
  promos average **7.72** (lowest 7.1) against **7.35** (lowest 5.4) for the old ones.
- **Passed the Evaluator first time:** 3 of 9 against 1 of 9. That run's Evaluator was a
  different model (the gpt-oss quota had run out, so every call went to Qwen), so this number
  mixes the prompt with the model.
- **Length is not fixed by the prompt.** Wrong-length rejections on the first cuts were 5 with
  the old prompts and 4, 2 and 4 in three runs with the new ones: within the noise. The first
  version of step 4 also swapped "too long" for "too short", which led to the "aim for the
  middle" wording. A real fix is in code: let the model choose lines, and compute the times.
- **A new mistake, then fixed:** the step-by-step wording made the Cutter put a time or a
  shortened id into the moment-id field. One added rule removed it in the next run.
- **Not tested:** the new Evaluator prompt's own scores (the daily quota ran out), and the
  memory learner (it needs an editor's review).

Small samples: 6–9 promos per run. Treat this as a direction, not a result.

---

## How to run

```bash
# 1. Free dry run with fake agents: no API key needed
python run.py --input "input/My Show" --show "My Show" --genre "romance drama" --agents stub

# 2. Real run on a few episodes (needs GROQ_API_KEY / GOOGLE_API_KEY in .env)
python run.py --input "input/My Show" --show "My Show" --genre "romance drama" --episodes 1-5

# 3. Web app
streamlit run app/streamlit_app.py
```

- Resume a stopped or paused run: `python run.py --resume <run_id>`. Add `--review` to review
  in the terminal.
- Run the tests: `pytest`

Evaluation commands (Phase 4):

```bash
# Label real episodes yourself, blind (human labels)
python -m evaluate.labels --input "<screenplays folder>" --show "<show>" --episodes 2-5

# AI labeller (Gemma); re-run the same command to continue if it stops
python -m evaluate.ai_labels --input "<screenplays folder>" --show "<show>" --genre "romance drama" --episodes 2-9

# Agreement between a trusted label file and the AI labels
python -m evaluate.agreement --reference output/eval/labels/<show>.yaml --candidate output/eval/labels/<show>-ai.yaml

# Make a show with planted answers, run the tool on it, compare
python -m evaluate.synthetic_show --show "The Glass Heir" --genre "romance drama" --episodes 8 --seed 7
python run.py --input "input/synthetic/the-glass-heir" --show "The Glass Heir" --genre "romance drama" --no-review --no-memory
python -m evaluate.compare --run <run_id>
```

---

## Known limits

- **It works from text only.** A hook that is purely visual, like a look or a slow reveal, can
  score lower than it deserves.
- **Ranking loses good moments.** In the full run, the labelled moments were found but didn't
  make the top list.
- **The labels are still not human.** The AI labeller scored well on the made-up show (92% recall),
  but it misses quiet romance on the real show. It needs a check against an editor's own labels.
- **Quiet romance is a blind spot** for both the tool and the labeller.
- **The made-up show is easier than real scripts**, so scores on it are a best case.
- **Free-tier quotas** limit how many shows can run per day. The cost report shows the
  bottleneck and what paid models would cost.
