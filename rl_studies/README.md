# RL studies

This folder contains readable RL loops and their experiment configs. **Energy–mobility is the only implemented nexus here.** Start with the [repository guide](../README.md) for installation. In `train.py`, set `MODE`, `NEXUS`, and `JUSTICE`; set `DISTRIBUTIVE_TYPE` for distributive justice or `PREFERENCE_PROFILE` for recognitional justice. The selector block at the top of that file lists the implemented choices, and `python -m rl_studies.train --list-options` prints the current YAML-defined types, profiles, priorities and seeds without training. The study loader reads the matching YAML and passes the resolved settings and simulator parameters to the `justnexus_v2g` library builder. The library constructs the day suite, observations, reward and environment without looking up study files. Implemented energy–mobility selections are base PPO, three distributive PPO types, procedural PPO, declared-preference recognitional PPO, intertemporal PPO, a dedicated distributive–recognitional PPO, and the epistemic-motivated priority sweep.

Each folder has one commented `config.yaml`. [Base](configs/energy_mobility/base/config.yaml) contains the three active reward weights, PPO settings, and both fixed and variable day sections. [Distributive](configs/energy_mobility/distributive/config.yaml) contains all three selectable types and their weights, including Gini's ranked-satisfaction weights. [Procedural](configs/energy_mobility/procedural/config.yaml) contains its two added reward weights. [Recognitional](configs/energy_mobility/recognitional/config.yaml) contains named preference profiles, ordered household objectives, and the action-gap weight. [Intertemporal](configs/energy_mobility/intertemporal/config.yaml) contains its two reward weights. [Combined distributive–recognitional](configs/energy_mobility/distributive_recognitional/config.yaml) contains its own ordered household preferences and five combined-specific coefficients. [Epistemic](configs/energy_mobility/epistemic/config.yaml) contains the explicit mobility/grid priorities and independent training seeds. All reward weights must be finite and nonnegative. Observation size, soft-grid operation, normalization references, and reward timing are fixed by the library's selected formulation. [`config.py`](config.py) reads the relevant sections and passes parsed values to the library. `--config-dir` and `--scenario` can select alternate file locations.

## Fixed or variable days

`MODE` chooses the day regime independently of nexus and justice:

| Mode | Training | Validation | Test |
| --- | --- | --- | --- |
| `fixed` | Repeat the bundled reference day | The same day, one deterministic episode per check | The same day again; in-sample replay only |
| `variable` | Draw from 64 train days uniformly with replacement at each reset | All 16 distinct validation days, once per check | All 32 reserved days, once after model selection |

There is **one training environment** and a separate validation environment during training; the 64/16/32 counts are distinct day configurations, not parallel PPO environments. Each episode has 48 half-hour steps. A variable day changes household load, PV output and selected departure times, while retaining the five households, trip-energy requirements, simulator physics, five actions and per-departure normalized base reward. Observation size follows the formulation: **52 for base, recognitional and epistemic; 54 for all three distributive types; 57 for procedural; 56 for combined distributive–recognitional; 62 for intertemporal**, in either day mode. It does not mean an episode has a variable number of steps. The library generates days in memory from the [scenario’s `day_variation` recipe](../justnexus_v2g/configs/scenario_energy_mobility.yaml) and the seed/count settings in [base `config.yaml`](configs/energy_mobility/base/config.yaml). With the bundled recipe and seeds, all 64/16/32 generated simulator parameter sets match the previous day suite. The run record stores the generation settings and sampled train-day IDs. This mode does **not** use the Phase 3B 64-feature matched-welfare environment or reproduce its trained results.

### Changing the variable-day recipe

Edit `day_variation` in the scenario YAML to choose which household load fields (`base_kw`, `morning_peak_kw`, `evening_peak_kw`) and PV fields (`peak_kw`, `taper_kw_per_step`) receive a uniform multiplier. Each household draws its own load and PV factors once per generated day. `factor_range: [low, high]` sets each range. Departure options are keyed by the scenario’s household labels; the chosen step also sets disconnection start, while return time stays fixed. `all_distinct: true` requires separate departure times. The library rejects unknown fields, missing household labels, impossible distinct assignments, and invalid ranges. Variable mode requires this section; fixed mode uses only the reference scenario values.

Edit `modes.variable.days` in the base study config for the generation seed, training-sampling seed, and train/validation/test day counts. The train script snapshots both the scenario and resolved settings with each run. Older run snapshots with the previous variation format remain loadable.

## Single-policy model selection and reserved test

For base, distributive, procedural, recognitional, intertemporal and combined studies, `train.py` calls PPO once on the training environment. Fixed mode validates every 10,000 steps; variable mode validates every 262,144 steps. Both also validate the final trained model. Validation uses deterministic actions and the mean undiscounted episode return across its complete validation split. The runner saves:

- `final_model.zip`: weights after the last training update;
- `best_validation_model.zip`: the globally highest validation mean (first strict best wins ties);
- `window_checkpoints/`: the best validation model in each 500,000-step window (latest within a 1e-6 tie), with a manifest in `validation.json`;
- `selected_model.zip`: a copy of the global validation best, used by the test command.

The selected model is the global validation best; the 500K window checkpoints are also retained for inspection. Variable-day test outcomes cannot affect checkpoint selection. Fixed-day validation and test revisit the training day, so their scores do not measure day generalization. Results on the reserved variable-day split are still within this synthetic day generator, not a population claim.

## Commands

From the repository root, follow the [installation instructions](../README.md). In an existing Python environment, the editable install is:

```bash
python -m pip install -e './justnexus_v2g[rl,test]'
```

The commands below assume the `justnexus-v2g` Conda environment from the repository guide. If you installed into another environment, replace `conda run -n justnexus-v2g python` with that environment's `python`. Start by listing choices or inspecting the default fixed-day base problem; neither command trains a policy:

```bash
conda run -n justnexus-v2g python -m rl_studies.train --list-options
conda run -n justnexus-v2g python -m rl_studies.train --inspect
```

For a **full base run**, choose one mode. Each command trains for the configured 3,000,000 requested PPO steps and prints its new run directory when complete:

```bash
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.train --justice base --mode fixed
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.train --justice base --mode variable
```

Run either line independently; they are two separate trainings. After one finishes, use the printed directory as `RUN`:

```bash
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.test RUN --inspect
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.test RUN
```

`train.py --inspect` shows the resolved observation, reward and config for one selected study. `test.py --inspect RUN` shows that run's saved mode, nexus, justice, optional variant and test size without evaluating a model. The test command needs **only the run directory**: it reads the training snapshots and validation selection, so there are no separate mode, justice or preference options to keep in sync. `test.py` writes `test.json` only once per run.

### Files created by a single-policy run

Training does not print a progress bar. While it runs, `monitor.csv` records each completed **training** episode (`r` is its return, `l` its step count, and `t` elapsed seconds). `validation.json` is updated after each validation check; its `evaluations` list contains every check's training step, deterministic per-day return, and mean return. Fixed mode checks every 10,000 steps on the same reference day; variable mode checks every 262,144 steps across all 16 validation days. `best_validation_model.zip` tracks the highest mean validation return, and `window_checkpoints/` holds the best model in each 500,000-step window. PPO finishes a full 512-step rollout, so a 3,000,000-step request reaches 3,000,320 steps and the final validation can add a short extra window. Training ends by writing `final_model.zip`, copying the validation best to `selected_model.zip`, and writing `run.json`. The separate test command then produces `test.json` with per-day and mean metrics. `resolved_config.json` and `scenario.yaml` snapshot the exact inputs, and generated `runs/` and `oracle_cache/` files are ignored by Git.

For a quick wiring check instead of a full run, add `--smoke` to the training command. It uses 64 CPU training steps; its scores are not performance results.

For variable days, select the mode; the study loader supplies its config and the library generates the day splits:

```bash
conda run -n justnexus-v2g python -m rl_studies.train --mode variable --inspect
conda run -n justnexus-v2g python -m rl_studies.train --mode variable --smoke
conda run -n justnexus-v2g python -m rl_studies.test /path/to/completed/variable/run
```

To select distributive justice, set `JUSTICE = "distributive"` and `DISTRIBUTIVE_TYPE` in `train.py`, or use the command line:

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice distributive --distributive-type mean_deviation --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice distributive --distributive-type max_min --mode variable --smoke
conda run -n justnexus-v2g python -m rl_studies.train --justice distributive --distributive-type generalized_gini --mode fixed --smoke
conda run -n justnexus-v2g python -m rl_studies.test /path/to/completed/distributive/run
```

The three types share the historical **54-value** distributive observation: the base 52 values plus normalized worst and mean excess mobility-burden context. Every type charges weighted `−F_t/74` at each household departure and weighted grid overload/command mismatch at every step. Only at the last departure does it add its selected dimensionless term: mean deviation `−w_w W/0.8−w_e E/0.24`, max–min `−w_m max_i b_i`, or generalized Gini `w_g(Σ_j r_j s_(j)−1)` with ascending satisfaction and default rank weights `r=(.40,.25,.17,.11,.07)`. All distributive weights default to `1.0`. Here `b_i` is household shortfall divided by its own trip requirement and `s_i=1−b_i`. The max–min and Gini types are **additions** to the common shortfall reward. With all weights at 1, the mean-deviation loss ranges from 0 to 1.75, while the other two losses range from 0 to 1; equal weights therefore do not imply equal reward strength. For Gini, `rank_weights_low_to_high` must contain five nonnegative weights summing to one; changing them changes whose satisfaction has greater rank priority.

## Procedural discharge authority

Set `JUSTICE = "procedural"` in `train.py`, or use `--justice procedural`. This selects the single procedural formulation. It retains the three normalized base reward terms, adds five household authority-availability features to the observation (57 values total), and applies a hard discharge-authorization gate before physical limits. Charging remains permitted for connected households.

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice procedural --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice procedural --mode variable --inspect
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.train --justice procedural --mode fixed
conda run --no-capture-output -n justnexus-v2g python -m rl_studies.test RUN
```

The [procedural config](configs/energy_mobility/procedural/config.yaml) exposes two nonnegative weights, both defaulting to 1.0. The incidence term counts raw discharge requests below −0.001 kW at prohibited household opportunities **before** the gate and divides by all prohibited opportunities at that step. The consistency term compares binary discharge activity within current authority-status groups. Each step's two losses are divided by the 48-transition horizon. The older archived authority-adjustment reward is not part of this formulation; the gate still reports its corrections as diagnostics. Validation selects the model by mean episode return. The test report includes each day's pre-gate incidence and mean stepwise consistency, then an arithmetic mean across test days. Comparing a base policy under the same authorization gate requires a separate matched-gate evaluation; an ordinary base `test.json` has no procedural group.

## Recognitional preferences and offline oracles

Set `JUSTICE = "recognitional"` and `PREFERENCE_PROFILE = "declared_needs_v1"` in `train.py`, or use `--justice recognitional --preference-profile declared_needs_v1`. The default profile defines an ordered objective list for each household. Each household has **a priority list**, not a single preference: all first minimize trip shortfall; then the reliable commuter prioritizes a 4 kWh after-trip reserve, the price-flexible worker the bill, the solar battery conserver EV throughput, the low-power affordability home incremental bill, and the battery autonomy home discharge avoidance. Later objectives break ties in the written order, and squared power is always the final tie-breaker. The [recognitional config](configs/energy_mobility/recognitional/config.yaml) shows every complete order. Add another entry under `profiles` to select a different named profile; within a profile, reorder or omit supported intermediate objectives (`minimum_reserve_shortfall`, `minimum_bill`, `minimum_incremental_bill`, `minimum_discharge`, `minimum_throughput`); reserve shortfall requires a positive `reserve_kwh`. The first `minimum_trip_shortfall` and final `minimum_squared_power` are fixed for this oracle method. `minimum_incremental_bill` subtracts that day's no-action bill, a constant, so at the same priority position it chooses the same action as `minimum_bill`; its separate name records the household's stated concern. The library validates the full list and checks every scenario label. These are declared experimental preferences, not measured household choices.

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice recognitional --preference-profile declared_needs_v1 --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice recognitional --preference-profile declared_needs_v1 --mode variable --smoke
```

For each selected day, the library solves five household-alone, full-day lexicographic oracles and verifies each schedule by replaying it in the simulator. Fixed mode uses one bundle. Variable mode prepares 64 training bundles and 16 validation bundles before PPO; when a train day repeats, its existing bundle is reused. `test.py` prepares the 32 reserved test bundles only after training has selected its model. Oracle bundles are cached in ignored `rl_studies/oracle_cache/`, keyed by day parameters, resolved preferences, and solver version; run provenance records each bundle hash. The study config and scenario are snapshotted with the run, so the same preferences and days can be rebuilt if the cache is removed. Preparing variable-day oracles can take several minutes on the first run.

The recognitional policy retains the base **52-value observation**. Let `o_it` be the day-specific oracle power and `a_it` the simulator-applied power. Each household has its own positive divisor `D_i = Σ_t o_it² Δt`. The per-step recognitional reward is `−w_r Σ_i (a_it−o_it)² Δt / D_i`, added to the three base terms; `action_gap_weight` sets `w_r` and defaults to `0.2`. The training gap is not clipped; the reporting alignment clips `1−Σ_t (a_it−o_it)² Δt / D_i` to `[0,1]`. These full-day oracle references have future-day information; the PPO does not receive their schedules in its observation. The test report includes this alignment for each day and its mean across days.

The energy–mobility [scenario labels](../justnexus_v2g/configs/scenario_energy_mobility.yaml) use the household names as machine-friendly IDs. Historical label provenance is preserved in the original research repository; only labels changed, so simulator parameters and oracle schedules remain the same.

## Joint distributive–recognitional controller

Set `JUSTICE = "distributive_recognitional"` or use `--justice distributive_recognitional`. Its [dedicated config](configs/energy_mobility/distributive_recognitional/config.yaml) specifies the complete ordered household preferences and five combined-specific weights: two for mobility disparity, one for the per-step oracle action gap, and two for final alignment-gap disparity. The shared three base weights and PPO/day settings still come from base `config.yaml`; editing pure distributive or pure recognitional configs does not silently change this joint study.

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice distributive_recognitional --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice distributive_recognitional --mode variable --smoke
conda run -n justnexus-v2g python -m rl_studies.test /path/to/completed/combined/run
```

The observation has **56 values**: base 52, two cumulative normalized mobility-burden context values, and two cumulative normalized oracle-gap disparity values. Each household's mobility shortfall is charged at departure; worst and mean above-average mobility-burden penalties are paid once at the last departure. The household-normalized squared applied-action oracle gap is charged each step without clipping. For alignment disparity only, each household's accumulated normalized gap is capped at 1; the worst and mean above-average gap penalties are paid once at episode end. Both disparity pairs use fixed divisors **0.8 and 0.24**. This is the original combined mean-deviation formulation, not a generic choice among the three pure distributive types. Fixed mode prepares one oracle bundle; variable mode prepares day-matched bundles for train, validation and, after model selection, reserved test. The joint observation/reward composition follows the historical experiment; the clean variable-day protocol extends it without claiming historical variable-day results.

## Intertemporal battery use and reserve

Set `JUSTICE = "intertemporal"` in `train.py`, or select it on the command line. The [intertemporal config](configs/energy_mobility/intertemporal/config.yaml) contains two nonnegative weights; both default to `1.0` as in the historical energy–mobility experiment. Edit them to change the relative penalties. Fixed and variable day modes use the same formulation and 62-value observation.

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice intertemporal --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice intertemporal --mode variable --smoke
```

For household `i`, the daily controlled-throughput cap is `(trip_requirement_i + reserve_i) / charge_efficiency_i + allowance_i`. The scenario sets both the reserve target and throughput allowance to 10% of battery capacity; the five reserve targets sum to **28.5 kWh**. The 62 observations are the base 52 plus each household's remaining throughput-budget fraction and signed battery-reserve margin fraction. Trip energy is not counted as controlled throughput. The base three reward terms remain active; intertemporal justice adds `−w_throughput × incremental_excess_kWh / 28.5` on each step and `−w_reserve × terminal_reserve_deficit_kWh / 28.5` once at the end of the episode. These fixed divisors reproduce the historical normalized reward; neither term is clipped. Variable days change load, PV, and departure times while leaving the battery capacities, requirements, reserve settings, and these reference scales unchanged.

## Epistemic-motivated priority sweep

Set `JUSTICE = "epistemic"` or use `--justice epistemic`. The [epistemic config](configs/energy_mobility/epistemic/config.yaml) lists explicit `priority_weights` λ and `training_seeds`; it does not ask for an ambiguous policy count. The study uses the base **52-value observation** and base simulator, with no native epistemic objective-profile modifier. At priority λ, the two base mobility/grid weights are redistributed as `λ × total` and `(1−λ) × total`, where `total` is their sum in base `config.yaml`; the base mismatch weight remains fixed. With bundled base weights, the training reward is `−2λ S/74−2(1−λ)G/267.9392226234346−M/160`, reproducing the historical energy–mobility sweep.

```bash
conda run -n justnexus-v2g python -m rl_studies.train --justice epistemic --mode fixed --inspect
conda run -n justnexus-v2g python -m rl_studies.train --justice epistemic --mode variable --smoke
conda run -n justnexus-v2g python -m rl_studies.test /path/to/completed/epistemic/sweep
```

The bundled 11 priorities and three seeds request **33 independent PPO trainings**. Each retains the best validation checkpoint in every 500K-step window; at 3M requested steps, that is six checkpoint candidates per run and **198 candidates** overall. The runner evaluates every candidate on the validation split, identifies candidates not dominated on **mean raw departure shortfall and mean raw squared import overload**, and writes `validation_portfolio.json`. The test command checks its saved hash, evaluates **all saved candidates** on the reserved test split, and labels a descriptive test Pareto front. The validation-selected portfolio remains the selected set; test outcomes never change its membership. Command mismatch is a training penalty, not a Pareto axis. Fixed mode's validation and test reuse the training day; variable mode uses separate 64/16/32 day sets. The historical experiment compared all 198 candidates on the reference day; the variable-day validation/test split is a new extension, not a reproduced historical result. `--smoke` limits this sweep to the first two configured priorities and first seed, each with 64 training steps, so its results are wiring checks only.

Remove `--smoke` for a full training run. Each run gets a new timestamped directory under `rl_studies/runs/`; explicit `--output` paths must not exist. `resolved_config.json` and `scenario.yaml` snapshot the exact inputs used by that run. `run.json` records their hashes, sampled training days and model selection; `validation.json` records every check and per-day return for single-policy studies. For those studies, `test.py` reloads the snapshots, checks their hashes, evaluates `selected_model.zip` on the test split, and writes `test.json` once. The epistemic portfolio has per-policy `validation.json` files and one parent `validation_portfolio.json`, with the same snapshot independence. A smoke run uses 64 CPU steps and validation checks at steps 32 and 64; its scores only check the wiring, not performance.

## Base formulation and test metrics

The base observation has seven system values (time sine/cosine, total load and PV, previous requested/applied total EV power, grid exchange) and nine values per household (previous requested/applied power, connection, SOC, time to departure, required departure energy, battery capacity, charge limit, discharge limit): `7 + 5×9 = 52`. The action is five signed kW requests. Soft import permits overload, which is penalized rather than hard-clipped.

The base reward is `-w_s S_t/74 - w_g G_t/267.9392226234346 - w_m M_t/160`, with departure shortfall `S_t` charged at each household's departure, squared import overload `G_t` charged each step above 9 kW, and authorized-to-applied physical command mismatch `M_t` charged each step. The three weights in base `config.yaml` default to `1.0` and apply to every justice selection. These divisors are fixed references for this scenario, not universal bounds. Distributive selections add one terminal term as described above.

`test.json` reports each deterministic test day's episode return and normalized metric groups, plus an arithmetic mean across days. Every model, including base and epistemic candidates, gets normalized mobility shortfall (`S/74`), normalized squared import overload (`G/267.9392226234346`), mismatch (`M/160`), household shortfall fractions, worst and mean above-average burden, worst household burden, normalized terminal reserve deficit, and normalized excess controlled throughput. A fixed declared preference profile supplies day-specific reference oracles for clipped mean alignment and worst above-average alignment burden; its resolved preferences are saved in the run snapshot. These evaluation measures do not change training rewards or validation selection. Procedural reports include the pre-gate prohibited-request rate and mean stepwise discharge-consistency and authority-compliance scores. The rate is undefined when a day has no prohibited opportunities. For the epistemic sweep, test reports normalized axes for every candidate, a descriptive `test_front`, and `validation_selected` for the portfolio chosen before testing. Definitions are implemented in the library's [metric calculator](../justnexus_v2g/src/justnexus_v2g/rl/metrics.py).

Metric families are independent of the justice used for training: for example, an intertemporal run's `mean_metrics.distributive` reports its household mobility-burden disparity. `test.json` records `trained_justice` alongside those results. Base, distributive, intertemporal and epistemic training does not prepare offline oracles; their test command prepares or loads the declared reference oracles only to calculate recognitional alignment. Recognitional and joint training already require day-matched oracles for their rewards. Cached oracle bundles are keyed by day and preferences, so policies tested on the same days reuse them. To compare test reports across runs, use the same scenario, generated day split, normalizers and declared oracle preference profile; keep day and training-seed variation distinct.
