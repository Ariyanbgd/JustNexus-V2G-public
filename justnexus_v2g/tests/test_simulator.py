from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.params import (
    EVParams,
    HouseholdParams,
    LoadProfileParams,
    MobilityProfileParams,
    PVProfileParams,
    SimulationParams,
    SystemParams,
)
from justnexus_v2g.engine.simulator import Simulator
from justnexus_v2g.exogenous.demand import household_demand_profile
from justnexus_v2g.exogenous.mobility import (
    ev_connected_profile,
    time_to_departure_profile,
    trip_energy_profile,
)
from justnexus_v2g.exogenous.pv import household_pv_profile
from justnexus_v2g.exogenous.tariffs import export_tariff_profile, import_tariff_profile


def make_params(
    n_households: int = 3,
    episode_steps: int = 48,
    initial_energy_kwh: float = 30.0,
    max_import_kw: float = 50.0,
    max_export_kw: float = 50.0,
) -> SimulationParams:
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
                    initial_energy_kwh=initial_energy_kwh,
                )
            )
            for _ in range(n_households)
        ],
        system=SystemParams(
            max_import_kw=max_import_kw,
            max_export_kw=max_export_kw,
        ),
    )


def test_simulator_reset_uses_configured_initial_energy():
    params = make_params(initial_energy_kwh=12.0)
    simulator = Simulator(params)
    state = simulator.reset()

    assert state.households[0].ev_energy_kwh == 12.0
    assert state.households[1].ev_energy_kwh == 12.0
    assert state.households[2].ev_energy_kwh == 12.0


def test_simulator_reset_seed_initializes_reproducible_rng():
    simulator_a = Simulator(make_params(n_households=3))
    simulator_b = Simulator(make_params(n_households=3))
    simulator_c = Simulator(make_params(n_households=3))

    simulator_a.reset(seed=123)
    simulator_b.reset(seed=123)
    simulator_c.reset(seed=456)

    assert simulator_a.last_seed == 123
    assert simulator_b.last_seed == 123
    assert simulator_c.last_seed == 456
    assert simulator_a.rng.random() == simulator_b.rng.random()
    assert simulator_a.rng.random() != simulator_c.rng.random()




def test_reset_randomizes_initial_energy_when_range_configured():
    params = make_params(n_households=1, initial_energy_kwh=30.0)
    params.households[0].ev.initial_energy_min_kwh = 10.0
    params.households[0].ev.initial_energy_max_kwh = 20.0
    simulator = Simulator(params)

    state_a = simulator.reset(seed=123)
    energy_a = state_a.households[0].ev_energy_kwh
    state_b = simulator.reset(seed=123)
    energy_b = state_b.households[0].ev_energy_kwh
    state_c = simulator.reset(seed=456)
    energy_c = state_c.households[0].ev_energy_kwh

    assert 10.0 <= energy_a <= 20.0
    assert energy_a == energy_b
    assert energy_a != energy_c


def test_always_connected_mobility_profile_has_no_trip_or_departure_countdown():
    params = make_params(n_households=1, episode_steps=96)
    params.cycle_steps = 48
    params.households[0].mobility.always_connected = True

    assert ev_connected_profile(params, 0, 20) is True
    assert time_to_departure_profile(params, 0, 20) == 0.0
    assert trip_energy_profile(params, 0, 16) == 0.0


def test_simulator_reset_creates_correct_number_of_households():
    params = make_params(3)
    simulator = Simulator(params)
    state = simulator.reset()

    assert state.t == 0
    assert len(state.households) == 3

    assert state.system.total_ev_kw == 0.0
    assert state.system.total_load_kw > 0.0
    assert state.system.total_pv_kw >= 0.0
    assert state.system.grid_exchange_kw == (
        state.system.total_load_kw + state.system.total_ev_kw - state.system.total_pv_kw
    )

    for household in state.households:
        assert household.ev_energy_kwh == 30.0
        assert household.requested_ev_power_kw == 0.0
        assert household.applied_ev_power_kw == 0.0
        assert household.trip_energy_kwh == 0.0
        assert household.battery_throughput_kwh == 0.0
        assert household.discharge_burden_kwh == 0.0
        assert household.was_deferred is False
        assert household.deferral_count == 0
        assert household.time_to_departure_hours >= 0.0
        assert household.required_departure_energy_kwh >= 0.0
        assert household.load_kw >= 0.0
        assert household.pv_kw >= 0.0


def test_simulator_step_advances_time():
    params = make_params(3)
    simulator = Simulator(params)

    state0 = simulator.reset()
    result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))
    state1 = result.state

    assert state0.t == 0
    assert state1.t == 1
    assert result.done is False
    assert len(state1.households) == 3
    assert state1.system.grid_exchange_kw == (
        state1.system.total_load_kw
        + state1.system.total_ev_kw
        - state1.system.total_pv_kw
    )


def test_grid_import_constraint_scales_charging():
    params = make_params(max_import_kw=5.0)
    simulator = Simulator(params)
    simulator.reset()

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[7.0, 7.0, 7.0]))

    assert result.constraints.grid_constrained is True
    assert result.constraints.pre_grid_total_ev_kw == 21.0
    assert result.constraints.post_grid_total_ev_kw < 21.0
    assert result.state.system.grid_exchange_kw <= 5.0 + 1e-9


def test_grid_export_constraint_scales_discharging():
    params = make_params(max_export_kw=1.0)
    simulator = Simulator(params)
    simulator.reset()

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[-7.0, -7.0, -7.0]))

    assert result.constraints.grid_constrained is True
    assert result.constraints.pre_grid_total_ev_kw == -21.0
    assert result.constraints.post_grid_total_ev_kw > -21.0
    assert result.state.system.grid_exchange_kw >= -1.0 - 1e-9


def test_charging_updates_throughput():
    params = make_params(3)
    simulator = Simulator(params)
    simulator.reset()
    result = simulator.step(SimulatorAction(ev_power_commands_kw=[4.0, 0.0, 0.0]))
    h0 = result.state.households[0]

    assert h0.applied_ev_power_kw == 4.0
    assert h0.battery_throughput_kwh == 2.0
    assert h0.discharge_burden_kwh == 0.0


def test_discharging_updates_throughput_and_discharge_burden():
    params = make_params(3)
    simulator = Simulator(params)
    simulator.reset()
    result = simulator.step(SimulatorAction(ev_power_commands_kw=[-4.0, 0.0, 0.0]))
    h0 = result.state.households[0]

    assert h0.applied_ev_power_kw == -4.0
    assert h0.battery_throughput_kwh == 2.0
    assert h0.discharge_burden_kwh == 2.0


def test_deferral_is_counted_when_service_is_too_small():
    params = make_params(3, initial_energy_kwh=10.0)
    simulator = Simulator(params)
    simulator.reset()

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.5, 0.0, 0.0]))
    h0 = result.state.households[0]

    assert h0.ev_connected is True
    assert h0.required_departure_energy_kwh > h0.ev_energy_kwh
    assert h0.was_deferred is True
    assert h0.deferral_count == 1


def test_charging_increases_energy_when_connected_with_efficiency():
    params = make_params(3)
    simulator = Simulator(params)
    state0 = simulator.reset()
    result = simulator.step(SimulatorAction(ev_power_commands_kw=[4.0, 0.0, 0.0]))
    state1 = result.state

    assert state0.households[0].ev_connected is True
    assert result.action.ev_power_commands_kw[0] == 4.0
    assert state1.households[0].requested_ev_power_kw == 4.0
    assert state1.households[0].applied_ev_power_kw == 4.0
    assert state1.households[0].ev_energy_kwh == 31.9
    assert result.constraints.clipped_by_connection[0] is False
    assert result.constraints.clipped_by_power_limit[0] is False
    assert result.constraints.clipped_by_energy_limit[0] is False


def test_trip_depletes_energy_on_departure():
    params = make_params(3, episode_steps=40)
    simulator = Simulator(params)
    simulator.reset()

    result = None
    for _ in range(16):
        result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))

    assert result is not None
    h0 = result.state.households[0]
    assert h0.ev_connected is False
    assert h0.trip_energy_kwh == 8.0
    assert h0.ev_energy_kwh == 22.0


def test_discharging_decreases_energy_with_efficiency():
    params = make_params(3)
    simulator = Simulator(params)
    simulator.reset()
    result = simulator.step(SimulatorAction(ev_power_commands_kw=[-4.0, 0.0, 0.0]))
    state1 = result.state

    assert state1.households[0].requested_ev_power_kw == -4.0
    assert state1.households[0].applied_ev_power_kw == -4.0
    assert round(state1.households[0].ev_energy_kwh, 6) == round(
        30.0 - (4.0 * 0.5 / 0.95), 6
    )


def test_requested_and_applied_power_differ_when_not_connected():
    params = make_params(3)
    simulator = Simulator(params)
    simulator.reset()

    for _ in range(17):
        result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))
        state = result.state

    assert state.households[0].ev_connected is False

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[4.0, 0.0, 0.0]))
    state = result.state
    assert state.households[0].requested_ev_power_kw == 4.0
    assert state.households[0].applied_ev_power_kw == 0.0
    assert result.constraints.clipped_by_connection[0] is True
    assert result.constraints.clipped_by_power_limit[0] is False
    assert result.constraints.clipped_by_energy_limit[0] is False


def test_requested_and_applied_power_differ_when_clipped_by_power_limit():
    params = make_params(3)
    simulator = Simulator(params)
    simulator.reset()

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[100.0, 0.0, 0.0]))
    state = result.state
    assert state.households[0].requested_ev_power_kw == 100.0
    assert state.households[0].applied_ev_power_kw == 7.0
    assert result.constraints.clipped_by_connection[0] is False
    assert result.constraints.clipped_by_power_limit[0] is True
    assert result.constraints.clipped_by_energy_limit[0] is False


def test_requested_and_applied_power_differ_when_clipped_by_energy_limit():
    params = make_params(3, episode_steps=100)
    simulator = Simulator(params)
    simulator.reset()

    while True:
        result = simulator.step(SimulatorAction(ev_power_commands_kw=[7.0, 0.0, 0.0]))
        h0 = result.state.households[0]
        if h0.ev_energy_kwh >= 60.0:
            break

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[7.0, 0.0, 0.0]))
    h0 = result.state.households[0]

    assert h0.requested_ev_power_kw == 7.0
    assert h0.applied_ev_power_kw == 0.0
    assert result.constraints.clipped_by_energy_limit[0] is True


def test_done_becomes_true_on_last_step():
    params = make_params(3, episode_steps=2)
    simulator = Simulator(params)
    simulator.reset()

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))
    assert result.done is True
    assert result.state.t == 1


def test_step_after_done_raises_clear_error():
    params = make_params(3, episode_steps=2)
    simulator = Simulator(params)
    simulator.reset()
    simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))

    try:
        simulator.step(SimulatorAction(ev_power_commands_kw=[0.0, 0.0, 0.0]))
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "Episode is already finished" in str(e)


def test_simulator_rejects_inconsistent_household_count():
    params = make_params(3)
    params.households = params.households[:2]

    try:
        Simulator(params)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_grid_allocation_is_pro_rata_even_with_previous_clipping_burden():
    params = make_params(max_import_kw=5.0)
    simulator = Simulator(params)
    simulator.reset()
    simulator.cumulative_grid_clipping_burden_kwh = [10.0, 0.0, 0.0]

    result = simulator.step(SimulatorAction(ev_power_commands_kw=[7.0, 7.0, 7.0]))

    applied = result.constraints.applied_ev_power_kw
    assert applied[0] == applied[1] == applied[2]


def test_configured_household_profiles_drive_exogenous_state():
    params = SimulationParams(
        n_households=1,
        dt_hours=0.5,
        episode_steps=8,
        households=[
            HouseholdParams(
                ev=EVParams(
                    battery_capacity_kwh=40.0,
                    max_charge_kw=7.0,
                    max_discharge_kw=7.0,
                    charge_efficiency=0.95,
                    discharge_efficiency=0.95,
                    initial_energy_kwh=20.0,
                ),
                load=LoadProfileParams(
                    base_kw=1.2,
                    evening_peak_kw=0.8,
                    evening_start=2,
                    evening_end=4,
                ),
                pv=PVProfileParams(
                    peak_kw=2.0,
                    peak_step=2,
                    generation_start=1,
                    generation_end=3,
                    taper_kw_per_step=0.5,
                ),
                mobility=MobilityProfileParams(
                    departure_step=3,
                    disconnected_start=3,
                    disconnected_end=5,
                    required_departure_energy_kwh=26.0,
                    trip_energy_kwh=9.0,
                ),
            )
        ],
        system=SystemParams(max_import_kw=50.0, max_export_kw=50.0),
    )

    simulator = Simulator(params)
    state0 = simulator.reset()
    assert state0.households[0].load_kw == 1.2
    assert state0.households[0].pv_kw == 0.0
    assert state0.households[0].time_to_departure_hours == 1.5
    assert state0.households[0].required_departure_energy_kwh == 26.0

    result = None
    for _ in range(3):
        result = simulator.step(SimulatorAction(ev_power_commands_kw=[0.0]))

    assert result is not None
    household = result.state.households[0]
    assert household.ev_connected is False
    assert household.load_kw == 2.0
    assert household.pv_kw == 1.5
    assert household.trip_energy_kwh == 9.0
    assert household.ev_energy_kwh == 11.0


def test_cycle_steps_repeat_daily_exogenous_profiles():
    params = make_params(n_households=1, episode_steps=96)
    params.cycle_steps = 48

    assert household_demand_profile(params, 0, 34) == household_demand_profile(
        params, 0, 82
    )
    assert household_pv_profile(params, 0, 23) == household_pv_profile(params, 0, 71)
    assert import_tariff_profile(params, 34) == import_tariff_profile(params, 82)
    assert export_tariff_profile(params, 23) == export_tariff_profile(params, 71)
    assert ev_connected_profile(params, 0, 16) == ev_connected_profile(params, 0, 64)
    assert trip_energy_profile(params, 0, 16) == trip_energy_profile(params, 0, 64)


def test_dynamic_import_discount_export_tariff_tracks_import_below_it():
    params = make_params(n_households=1)
    params.system.export_tariff_mode = "dynamic_import_discount"
    params.system.export_tariff_discount_eur_per_kwh = 0.02

    for t in (1, 20, 36):
        assert export_tariff_profile(params, t) == (
            import_tariff_profile(params, t) - 0.02
        )


def test_configured_morning_household_demand_peak():
    params = make_params(n_households=1)
    profile = params.households[0].load
    profile.base_kw = 0.2
    profile.morning_peak_kw = 1.8
    profile.morning_start = 13
    profile.morning_end = 20

    assert household_demand_profile(params, 0, 12) == 0.2
    assert household_demand_profile(params, 0, 13) == 2.0
    assert household_demand_profile(params, 0, 20) == 2.0
    assert household_demand_profile(params, 0, 21) == 0.2


def test_scenario_specific_import_tariff_periods_override_default_profile():
    params = make_params(n_households=1)
    params.system.import_tariff_default_eur_per_kwh = 0.11
    params.system.import_tariff_periods = [
        {"start_step": 12, "end_step": 18, "eur_per_kwh": 0.33}
    ]

    assert import_tariff_profile(params, 10) == 0.11
    assert import_tariff_profile(params, 12) == 0.33
    assert import_tariff_profile(params, 18) == 0.33
    assert import_tariff_profile(params, 19) == 0.11
