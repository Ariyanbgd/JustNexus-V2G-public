# JustNexus offline energy–mobility demo

This interactive demo shows how different control priorities affect five households with electric vehicles and a shared neighborhood grid. It replays **one synthetic day** using results prepared from an earlier version of the experiments. You can explore the schedules and compare outcomes without installing Python or training a model.

## Open the demo

Download or clone this repository, then open [energy_mobility_demo.html](energy_mobility_demo.html) from your computer in a recent desktop browser. You can double-click the file; no server, Python installation, or internet connection is required. The HTML contains its data, code, styles, and image asset. The [household cards](JustNexus_Household_Cards_A4_Updated.pdf) are an optional printable companion; each card describes one household's travel need, preferences, and permission to discharge its EV.

If your browser restricts local HTML files, you can start a server **on your own computer** from the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8765/JustNexus_Offline_Demo/energy_mobility_demo.html` on that same computer. `127.0.0.1` means your own machine; this command does not depend on a server operated by the project.

## Explore the interface

1. Start on **Introduction**. Click a house to see its EV, travel schedule, electricity demand, solar generation, preferences, and discharge permission. **EVs left idle** shows the reference day without EV charging or discharge.
2. Choose one of the five justice tabs. Select a policy, then click **Reveal outcomes** to see its full-day indicators. Use **Play**, the time controls, or the timeline to inspect what happens across the 48 half-hour steps.
3. Compare policies within a tab. The **1:1 policy** gives grid and travel equal numerical training weights; it is a reference for comparison, not a claim of fairness.
4. In **Epistemic**, open the Pareto chart to explore additional grid–travel compromises. Green points are nondominated among the candidates shown; gray points are dominated on the two plotted measures. Selecting a point replays its saved schedule.

The scenario has a 9 kW neighborhood import target. Imports above that target are allowed and measured as grid overload. Households can have different travel requirements, EV capacities, preferences, and discharge permissions.

## What the choices illustrate

| View | What to compare |
| --- | --- |
| **Epistemic** | Grid priority, travel priority, and intermediate policies. The full chart includes 198 evaluated checkpoint candidates, with 67 nondominated on the plotted outcomes. |
| **Distributive** | The five households' shares of missing trip energy. **Share mobility burden** aims to make the household shortfall fractions more similar. |
| **Recognitional** | How closely the applied schedules follow five separately optimized household plans. **Fully respect household needs** combines those plans without coordinating their grid use; **Balance household needs and grid** makes a grid-aware compromise. |
| **Procedural** | Whether discharge requests respect each household's permission and whether households with the same authority state receive consistent treatment. |
| **Intertemporal** | End-of-day energy reserve and battery energy cycled beyond a household-specific daily cap. |

**Share the compromise** is a separate policy combining distributive and recognitional objectives. It appears where those outcomes are compared. The five household oracle plans are optimization references, not a sixth justice policy.

## Reading the indicators

- **Missing trip energy** is the energy needed but unavailable when an EV departs, in kWh. A household shortfall percentage divides that home's missing energy by its own trip-energy requirement.
- **Squared grid overload** adds the squared amount of import above 9 kW over the day, in kW²h. It is the grid measure in the Pareto plot; lower is better.
- **Plan alignment** compares a household's applied EV-power schedule with its separately optimized reference schedule. It is scaled to 0–100% relative to that household's idle-to-reference gap; 100% means the schedules match. It does **not** mean 100% of travel or other needs were met.
- **Discharge compliance and consistency** summarize the demo's permission and equal-treatment indicators. They describe how the controller handles discharge, separately from its travel or grid outcome.
- **Reserve deficit and throughput excess** show how much end-of-day battery energy falls short of a reserve target and how much controlled charging/discharging exceeds a daily cap. Trip consumption is excluded from controlled throughput.

Read the household indicators alongside neighborhood totals: a policy can improve a total while changing who bears the burden. The interface displays the measures for each saved replay; clicking a different policy does not recalculate or retrain it.

## Data, scope, and reproducibility

The demo HTML embeds policy frames, plotted candidate points, scenario values, and reference plans. Its one-day setup corresponds to the bundled [energy–mobility scenario](../justnexus_v2g/configs/scenario_energy_mobility.yaml). Its outputs came from an earlier experiment version on a single fixed day. The original PPO checkpoints, episode CSVs, and HTML build script are **not** distributed here. This folder provides a replay, not a pipeline for recreating the embedded numbers.

The current [RL studies](../rl_studies/README.md) support fixed and variable days, three distributive formulations, and normalized test metrics. The demo does not offer those options, and its displayed indicators use earlier presentation definitions. For current experiments and test reports, use `rl_studies.train` and `rl_studies.test`. Neither this synthetic scenario nor the demo establishes measured preferences of real households or a globally optimal policy.

The demo files are licensed under [CC BY 4.0](LICENSE.md).
