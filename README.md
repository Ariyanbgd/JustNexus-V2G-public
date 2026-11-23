# JustNexus V2G

JustNexus V2G is a synthetic residential vehicle-to-grid (V2G) simulator and a set of reinforcement learning (RL) studies about operational and justice outcomes. The public study in this repository covers the **energy–mobility** nexus: EV charging and discharge affect household travel needs and neighborhood grid loading.

| Folder | What it contains |
| --- | --- |
| [`justnexus_v2g/`](justnexus_v2g/README.md) | Importable Python library: scenario, simulator, Gymnasium environment, RL problem builders, day generator, and test metrics. |
| [`rl_studies/`](rl_studies/README.md) | Experiment configurations and readable PPO training and test entry points. |
| [`JustNexus_Offline_Demo/`](JustNexus_Offline_Demo/README.md) | Self-contained interactive demo with precomputed one-day policy replays and printable household cards. |

The library and studies support base operation, five justice dimensions (distributive, procedural, recognitional, intertemporal, and epistemic), and a distributive–recognitional combination. Distributive justice has three selectable formulations. Studies can use the reference day (`fixed`) or separate generated training, validation, and test days (`variable`). See the [study guide](rl_studies/README.md) for the available choices and the meaning of each evaluation metric.

## Install

Use Python 3.11 or newer. From the repository root, either create the small Conda environment and install the local package:

```bash
conda env create -f environment.yml
conda activate justnexus-v2g
python -m pip install -e './justnexus_v2g[rl,test]'
```

Or use an existing Python environment and run only the `pip install` line. The Conda file selects Python and pip; package dependencies are declared in [`pyproject.toml`](justnexus_v2g/pyproject.toml). Run the study commands from the repository root.

## First study

List the supported choices, then inspect the base problem without training:

```bash
python -m rl_studies.train --list-options
python -m rl_studies.train --nexus energy_mobility --justice base --mode fixed --inspect
```

A full fixed-day base run uses the configured 3 million requested PPO steps:

```bash
python -m rl_studies.train --nexus energy_mobility --justice base --mode fixed
python -m rl_studies.test /path/to/the/completed/run
```

The training command prints its run directory. Pass that directory to `test`; the test command reads the saved configuration and validation-selected model. The [study guide](rl_studies/README.md) explains variable days, other justice choices, checkpoints, and the normalized metrics in `test.json`. A fixed-day test replays the training day, while variable mode reserves distinct test days.

## Offline stakeholder demo

Open [`energy_mobility_demo.html`](JustNexus_Offline_Demo/energy_mobility_demo.html) in a recent browser. No Python installation or network connection is needed. The [demo README](JustNexus_Offline_Demo/README.md) explains the controls, indicators, and five household cards.

The demo embeds precomputed replays from an earlier version of the experiments. It contains no PPO checkpoint files and does not train or evaluate policies in the browser. Its one-day results and displayed indicators should be read as a discussion aid; use `rl_studies.test` for the current study's normalized test metrics.

## Licenses

The Python code in `justnexus_v2g/` and `rl_studies/` is released under the [MIT License](LICENSE). The HTML demo, embedded replays, household-card PDF, and README in `JustNexus_Offline_Demo/` are separately licensed under [CC BY 4.0](JustNexus_Offline_Demo/LICENSE.md).
