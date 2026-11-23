# JustNexus V2G

Python library for synthetic residential vehicle-to-grid simulation and configurable reinforcement-learning problems. The simulator, Gymnasium environment, day generator, reward wrappers, and test metrics live under `src/justnexus_v2g/`. Start with the [repository guide](../README.md) for installation and an example run.

The sibling [`rl_studies/`](../rl_studies/README.md) contains the runnable PPO training and test scripts, study configurations, and the supported **energy–mobility** study. Its README describes fixed and variable days, justice choices, validation checkpoints, and test reports.

The [energy–mobility scenario](configs/scenario_energy_mobility.yaml) contains the reference day and an optional `day_variation` recipe. The recipe names the load and PV fields to scale, their uniform factor ranges, and allowed departure steps by household label. The study config chooses fixed or variable mode and sets the generated day counts and seeds; the library validates and generates the days.

From the repository root, install the library and RL dependencies:

```bash
python -m pip install -e './justnexus_v2g[rl,test]'
python -m rl_studies.train --list-options
python -m rl_studies.train --nexus energy_mobility --justice base --mode fixed --inspect
```

`--list-options` and `--inspect` do not train a policy. For full training and one-shot testing, use the commands in the [RL studies README](../rl_studies/README.md). Generated runs and oracle caches stay under `rl_studies/` and are ignored by Git.

## Build a problem from Python

The study loader selects and resolves YAML files; the library builder receives those inputs explicitly. From the repository root:

```python
from rl_studies.config import load_study_inputs
from justnexus_v2g.rl import make_rl_problem

inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice=None)
problem = make_rl_problem(
    mode="fixed",
    nexus="energy_mobility",
    justice=None,
    configuration=inputs.configuration,
    source_params=inputs.source_params,
)
try:
    observation, info = problem.env.reset(seed=0)
    print(problem.env.observation_space, problem.env.action_space)
finally:
    problem.env.close()
```

The CLI names this choice `base`; the lower-level Python API represents it with `justice=None`. For another formulation, select it in `load_study_inputs` and pass the same `justice` (and any required `distributive_type` or `preference_profile`) to `make_rl_problem`. The builder also accepts `split="train"`, `"validation"`, or `"test"`; the study loader supplies the day configuration for the chosen mode. The [study guide](../rl_studies/README.md) lists each option and its config file.
