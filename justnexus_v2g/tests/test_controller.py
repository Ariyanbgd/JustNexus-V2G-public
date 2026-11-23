from justnexus_v2g.controllers.heuristic import (
    AlwaysChargeController,
    EnergyAwareController,
)
from justnexus_v2g.core.params import (
    EVParams,
    HouseholdParams,
    SimulationParams,
    SystemParams,
)
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


def test_always_charge_controller_returns_one_command_per_household():
    params = make_params()
    simulator = Simulator(params)
    state = simulator.reset()

    controller = AlwaysChargeController(charge_power_kw=4.0)
    action = controller.act(state)

    assert len(action.ev_power_commands_kw) == params.n_households


def test_run_episode_returns_results_until_done():
    params = make_params(episode_steps=4)
    simulator = Simulator(params)
    controller = AlwaysChargeController(charge_power_kw=4.0)

    simulator.reset()
    results = simulator.run_episode(controller)

    assert len(results) == 3
    assert results[-1].done is True
    assert results[-1].state.t == 3


def test_energy_aware_controller_charges_when_price_is_low_with_grid_headroom():
    params = make_params(n_households=3)
    simulator = Simulator(params)
    state = simulator.reset()

    controller = EnergyAwareController(
        charge_power_kw=6.0,
        low_price_threshold_eur_per_kwh=0.20,
        target_grid_usage_kw=12.0,
    )

    action = controller.act(state)

    assert action.ev_power_commands_kw == [3.1, 3.1, 3.1]


def test_energy_aware_controller_idles_when_price_is_high():
    params = make_params(n_households=3, episode_steps=48)
    simulator = Simulator(params)
    simulator.reset()
    for _ in range(34):
        result = simulator.step(
            AlwaysChargeController(charge_power_kw=0.0).act(simulator.state)
        )
    state = result.state

    controller = EnergyAwareController(
        charge_power_kw=6.0,
        low_price_threshold_eur_per_kwh=0.20,
        target_grid_usage_kw=12.0,
    )

    action = controller.act(state)

    assert action.ev_power_commands_kw == [0.0, 0.0, 0.0]


def test_energy_aware_controller_limits_charging_to_grid_headroom():
    params = make_params(n_households=2)
    simulator = Simulator(params)
    state = simulator.reset()

    controller = EnergyAwareController(
        charge_power_kw=7.0,
        low_price_threshold_eur_per_kwh=0.20,
        target_grid_usage_kw=4.0,
    )

    action = controller.act(state)

    assert action.ev_power_commands_kw == [1.15, 1.15]
