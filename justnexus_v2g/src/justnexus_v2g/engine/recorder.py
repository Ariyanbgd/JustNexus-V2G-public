from dataclasses import dataclass, field
from pathlib import Path
import csv

from justnexus_v2g.core.types import StepResult


@dataclass
class EpisodeRecorder:
    """Stores per-step episode logs in simple row format."""
    rows: list[dict] = field(default_factory=list)

    def record(self, result: StepResult) -> None:
        state = result.state
        constraints = result.constraints

        row = {
            "t": state.t,
            "done": result.done,
            "total_load_kw": state.system.total_load_kw,
            "total_pv_kw": state.system.total_pv_kw,
            "total_ev_kw": state.system.total_ev_kw,
            "grid_exchange_kw": state.system.grid_exchange_kw,
            "pre_intervention_grid_exchange_kw": state.system.pre_intervention_grid_exchange_kw,
            "unmet_demand_kwh": state.system.unmet_demand_kwh,
            "pv_curtailment_kwh": state.system.pv_curtailment_kwh,
            "import_tariff_eur_per_kwh": state.system.import_tariff_eur_per_kwh,
            "export_tariff_eur_per_kwh": state.system.export_tariff_eur_per_kwh,
            "imported_energy_kwh": state.system.imported_energy_kwh,
            "exported_energy_kwh": state.system.exported_energy_kwh,
            "import_cost_eur": state.system.import_cost_eur,
            "export_revenue_eur": state.system.export_revenue_eur,
            "net_energy_cost_eur": state.system.net_energy_cost_eur,
            "grid_constrained": constraints.grid_constrained,
            "pre_grid_total_ev_kw": constraints.pre_grid_total_ev_kw,
            "post_grid_total_ev_kw": constraints.post_grid_total_ev_kw,
            "grid_ev_reduction_kw": (
                constraints.pre_grid_total_ev_kw - constraints.post_grid_total_ev_kw
            ),
        }

        for i, household in enumerate(state.households):
            row[f"h{i}_unmet_demand_kwh"] = household.unmet_demand_kwh
            row[f"h{i}_pv_curtailment_kwh"] = household.pv_curtailment_kwh
            row[f"h{i}_connected"] = household.ev_connected
            row[f"h{i}_ev_energy_kwh"] = household.ev_energy_kwh
            row[f"h{i}_requested_ev_power_kw"] = household.requested_ev_power_kw
            row[f"h{i}_applied_ev_power_kw"] = household.applied_ev_power_kw
            row[f"h{i}_time_to_departure_hours"] = household.time_to_departure_hours
            row[f"h{i}_required_departure_energy_kwh"] = (
                household.required_departure_energy_kwh
            )
            row[f"h{i}_trip_energy_kwh"] = household.trip_energy_kwh
            row[f"h{i}_departure_energy_kwh"] = household.departure_energy_kwh
            row[f"h{i}_departure_energy_shortfall_kwh"] = (
                household.departure_energy_shortfall_kwh
            )
            row[f"h{i}_battery_throughput_kwh"] = household.battery_throughput_kwh
            row[f"h{i}_discharge_burden_kwh"] = household.discharge_burden_kwh
            row[f"h{i}_was_deferred"] = household.was_deferred
            row[f"h{i}_deferral_count"] = household.deferral_count
            row[f"h{i}_load_kw"] = household.load_kw
            row[f"h{i}_pv_kw"] = household.pv_kw
            row[f"h{i}_net_household_power_kw"] = household.net_household_power_kw
            row[f"h{i}_imported_energy_kwh"] = household.imported_energy_kwh
            row[f"h{i}_exported_energy_kwh"] = household.exported_energy_kwh
            row[f"h{i}_import_cost_eur"] = household.import_cost_eur
            row[f"h{i}_export_revenue_eur"] = household.export_revenue_eur
            row[f"h{i}_net_energy_cost_eur"] = household.net_energy_cost_eur

        self.rows.append(row)

    def summary(self) -> dict:
        if not self.rows:
            return {
                "total_unmet_demand_kwh": 0.0,
                "total_pv_curtailment_kwh": 0.0,
                "num_steps": 0,
                "max_grid_exchange_kw": 0.0,
                "min_grid_exchange_kw": 0.0,
                "num_grid_constrained_steps": 0,
                "max_grid_ev_reduction_kw": 0.0,
                "total_import_cost_eur": 0.0,
                "total_export_revenue_eur": 0.0,
                "total_net_energy_cost_eur": 0.0,
                "h0_final_ev_energy_kwh": 0.0,
                "h0_final_battery_throughput_kwh": 0.0,
                "h0_final_discharge_burden_kwh": 0.0,
                "h0_final_deferral_count": 0,
                "h0_final_net_energy_cost_eur": 0.0,
            }

        grid_values = [row["grid_exchange_kw"] for row in self.rows]
        grid_reductions = [row["grid_ev_reduction_kw"] for row in self.rows]
        import_costs = [row["import_cost_eur"] for row in self.rows]
        export_revenues = [row["export_revenue_eur"] for row in self.rows]
        net_costs = [row["net_energy_cost_eur"] for row in self.rows]
        num_grid_constrained_steps = sum(
            int(row["grid_constrained"]) for row in self.rows
        )
        last = self.rows[-1]

        return {
            "total_unmet_demand_kwh": sum(row["unmet_demand_kwh"] for row in self.rows),
            "total_pv_curtailment_kwh": sum(row["pv_curtailment_kwh"] for row in self.rows),
            "num_steps": len(self.rows),
            "max_grid_exchange_kw": max(grid_values),
            "min_grid_exchange_kw": min(grid_values),
            "num_grid_constrained_steps": num_grid_constrained_steps,
            "max_grid_ev_reduction_kw": max(grid_reductions),
            "total_import_cost_eur": sum(import_costs),
            "total_export_revenue_eur": sum(export_revenues),
            "total_net_energy_cost_eur": sum(net_costs),
            "h0_final_ev_energy_kwh": last["h0_ev_energy_kwh"],
            "h0_final_battery_throughput_kwh": last["h0_battery_throughput_kwh"],
            "h0_final_discharge_burden_kwh": last["h0_discharge_burden_kwh"],
            "h0_final_deferral_count": last["h0_deferral_count"],
            "h0_final_net_energy_cost_eur": last["h0_net_energy_cost_eur"],
        }

    def to_csv(self, path: str | Path) -> None:
        if not self.rows:
            raise ValueError("No rows recorded; cannot export CSV.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(self.rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)