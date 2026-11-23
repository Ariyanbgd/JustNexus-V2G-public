from justnexus_v2g.controllers.heuristic import AlwaysChargeController
from justnexus_v2g.core.params import (
    EVParams,
    HouseholdParams,
    SimulationParams,
    SystemParams,
)
from justnexus_v2g.engine.recorder import EpisodeRecorder
from justnexus_v2g.engine.simulator import Simulator


def make_params(n_households: int = 3, episode_steps: int = 6) -> SimulationParams:
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


def test_recorder_collects_episode_rows():
    params = make_params(episode_steps=4)
    simulator = Simulator(params)
    controller = AlwaysChargeController(charge_power_kw=4.0)
    recorder = EpisodeRecorder()

    simulator.reset()
    results = simulator.run_episode(controller, recorder=recorder)

    assert len(results) == 3
    assert len(recorder.rows) == 3
    assert recorder.rows[0]["t"] == 1

    # Existing household/state fields
    assert "h0_ev_energy_kwh" in recorder.rows[0]
    assert "h0_battery_throughput_kwh" in recorder.rows[0]
    assert "h0_discharge_burden_kwh" in recorder.rows[0]
    assert "h0_deferral_count" in recorder.rows[0]

    # Existing grid-constraint fields
    assert "grid_constrained" in recorder.rows[0]
    assert "pre_grid_total_ev_kw" in recorder.rows[0]
    assert "post_grid_total_ev_kw" in recorder.rows[0]
    assert "grid_ev_reduction_kw" in recorder.rows[0]

    # New system-level tariff/cost fields
    assert "import_tariff_eur_per_kwh" in recorder.rows[0]
    assert "export_tariff_eur_per_kwh" in recorder.rows[0]
    assert "imported_energy_kwh" in recorder.rows[0]
    assert "exported_energy_kwh" in recorder.rows[0]
    assert "import_cost_eur" in recorder.rows[0]
    assert "export_revenue_eur" in recorder.rows[0]
    assert "net_energy_cost_eur" in recorder.rows[0]

    # New household-level attributed cost fields
    assert "h0_net_household_power_kw" in recorder.rows[0]
    assert "h0_imported_energy_kwh" in recorder.rows[0]
    assert "h0_exported_energy_kwh" in recorder.rows[0]
    assert "h0_import_cost_eur" in recorder.rows[0]
    assert "h0_export_revenue_eur" in recorder.rows[0]
    assert "h0_net_energy_cost_eur" in recorder.rows[0]


def test_recorder_summary_returns_basic_stats():
    params = make_params(episode_steps=4)
    simulator = Simulator(params)
    controller = AlwaysChargeController(charge_power_kw=4.0)
    recorder = EpisodeRecorder()

    simulator.reset()
    simulator.run_episode(controller, recorder=recorder)

    summary = recorder.summary()

    assert summary["num_steps"] == 3
    assert "max_grid_exchange_kw" in summary
    assert "min_grid_exchange_kw" in summary
    assert "num_grid_constrained_steps" in summary
    assert "max_grid_ev_reduction_kw" in summary

    # New total system-level cost summary fields
    assert "total_import_cost_eur" in summary
    assert "total_export_revenue_eur" in summary
    assert "total_net_energy_cost_eur" in summary

    # Existing household summary fields
    assert "h0_final_ev_energy_kwh" in summary
    assert "h0_final_battery_throughput_kwh" in summary
    assert "h0_final_discharge_burden_kwh" in summary
    assert "h0_final_deferral_count" in summary

    # New household final attributed cost summary field
    assert "h0_final_net_energy_cost_eur" in summary