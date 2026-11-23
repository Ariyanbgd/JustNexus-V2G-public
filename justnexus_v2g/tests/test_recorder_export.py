from pathlib import Path

from justnexus_v2g.controllers.heuristic import AlwaysChargeController
from justnexus_v2g.core.params import (
    EVParams,
    HouseholdParams,
    SimulationParams,
    SystemParams,
)
from justnexus_v2g.engine.recorder import EpisodeRecorder
from justnexus_v2g.engine.simulator import Simulator


def make_params(n_households: int = 3, episode_steps: int = 4) -> SimulationParams:
    return SimulationParams(
        n_households=n_households,
        dt_hours=0.5,
        episode_steps=episode_steps,
        households=[
            HouseholdParams(
                ev=EVParams(
                    battery_capacity_kwh=60.0,
                    max_charge_kw=7.0,
                    max_discharge_kw=7.0,
                    charge_efficiency=0.95,
                    discharge_efficiency=0.95,
                    initial_energy_kwh=30.0,
                )
            )
            for _ in range(n_households)
        ],
        system=SystemParams(
            max_import_kw=50.0,
            max_export_kw=50.0,
        ),
    )


def test_recorder_can_export_csv(tmp_path: Path):
    params = make_params()
    simulator = Simulator(params)
    controller = AlwaysChargeController(charge_power_kw=4.0)
    recorder = EpisodeRecorder()

    simulator.reset()
    simulator.run_episode(controller, recorder=recorder)

    out_path = tmp_path / "episode.csv"
    recorder.to_csv(out_path)

    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "grid_exchange_kw" in text
    assert "h0_ev_energy_kwh" in text
