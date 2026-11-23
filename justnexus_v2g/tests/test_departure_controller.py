from justnexus_v2g.controllers.heuristic import DepartureAwareController
from justnexus_v2g.core.state import HouseholdState, SimulatorState, SystemState


def make_state(households: list[HouseholdState]) -> SimulatorState:
    return SimulatorState(
        t=0,
        households=households,
        system=SystemState(
            total_load_kw=0.0,
            total_pv_kw=0.0,
            total_ev_kw=0.0,
            grid_exchange_kw=0.0,
            import_tariff_eur_per_kwh=0.0,
            export_tariff_eur_per_kwh=0.0,
            imported_energy_kwh=0.0,
            exported_energy_kwh=0.0,
            import_cost_eur=0.0,
            export_revenue_eur=0.0,
            net_energy_cost_eur=0.0,
                ),
    )


def test_departure_aware_controller_idle_when_not_connected():
    controller = DepartureAwareController()

    state = make_state(
        [
            HouseholdState(
                ev_connected=False,
                ev_energy_kwh=10.0,
                time_to_departure_hours=1.0,
                required_departure_energy_kwh=20.0,
                load_kw=0.0,
                pv_kw=0.0,
                requested_ev_power_kw=0.0,
                applied_ev_power_kw=0.0,
                trip_energy_kwh=0.0,
                battery_throughput_kwh=0.0,
                discharge_burden_kwh=0.0,
                was_deferred=False,
                deferral_count=0,
                net_household_power_kw=0.0,
                imported_energy_kwh=0.0,
                exported_energy_kwh=0.0,
                import_cost_eur=0.0,
                export_revenue_eur=0.0,
                net_energy_cost_eur=0.0,
            )
        ]
    )

    action = controller.act(state)
    assert action.ev_power_commands_kw == [0.0]


def test_departure_aware_controller_idle_when_requirement_already_met():
    controller = DepartureAwareController()

    state = make_state(
        [
            HouseholdState(
                ev_connected=True,
                ev_energy_kwh=25.0,
                time_to_departure_hours=4.0,
                required_departure_energy_kwh=20.0,
                load_kw=0.0,
                pv_kw=0.0,
                requested_ev_power_kw=0.0,
                applied_ev_power_kw=0.0,
                trip_energy_kwh=0.0,
                battery_throughput_kwh=0.0,
                discharge_burden_kwh=0.0,
                was_deferred=False,
                deferral_count=0,
                net_household_power_kw=0.0,
                imported_energy_kwh=0.0,
                exported_energy_kwh=0.0,
                import_cost_eur=0.0,
                export_revenue_eur=0.0,
                net_energy_cost_eur=0.0,
            )
        ]
    )

    action = controller.act(state)
    assert action.ev_power_commands_kw == [0.0]


def test_departure_aware_controller_uses_normal_power_when_gap_exists():
    controller = DepartureAwareController(
        normal_charge_power_kw=4.0,
        urgent_charge_power_kw=7.0,
        urgent_horizon_hours=2.0,
    )

    state = make_state(
        [
            HouseholdState(
                ev_connected=True,
                ev_energy_kwh=10.0,
                time_to_departure_hours=5.0,
                required_departure_energy_kwh=20.0,
                load_kw=0.0,
                pv_kw=0.0,
                requested_ev_power_kw=0.0,
                applied_ev_power_kw=0.0,
                trip_energy_kwh=0.0,
                battery_throughput_kwh=0.0,
                discharge_burden_kwh=0.0,
                was_deferred=False,
                deferral_count=0,
                net_household_power_kw=0.0,
                imported_energy_kwh=0.0,
                exported_energy_kwh=0.0,
                import_cost_eur=0.0,
                export_revenue_eur=0.0,
                net_energy_cost_eur=0.0,
            )
        ]
    )

    action = controller.act(state)
    assert action.ev_power_commands_kw == [4.0]


def test_departure_aware_controller_uses_urgent_power_when_departure_is_close():
    controller = DepartureAwareController(
        normal_charge_power_kw=4.0,
        urgent_charge_power_kw=7.0,
        urgent_horizon_hours=2.0,
    )

    state = make_state(
        [
            HouseholdState(
                ev_connected=True,
                ev_energy_kwh=10.0,
                time_to_departure_hours=1.5,
                required_departure_energy_kwh=20.0,
                load_kw=0.0,
                pv_kw=0.0,
                requested_ev_power_kw=0.0,
                applied_ev_power_kw=0.0,
                trip_energy_kwh=0.0,
                battery_throughput_kwh=0.0,
                discharge_burden_kwh=0.0,
                was_deferred=False,
                deferral_count=0,
                net_household_power_kw=0.0,
                imported_energy_kwh=0.0,
                exported_energy_kwh=0.0,
                import_cost_eur=0.0,
                export_revenue_eur=0.0,
                net_energy_cost_eur=0.0,
            )
        ]
    )

    action = controller.act(state)
    assert action.ev_power_commands_kw == [7.0]