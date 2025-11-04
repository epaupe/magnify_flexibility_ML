import numpy as np
from src.mpc import init_mpc_with_armax, update_mpc_parameters, solve_mpc

class RB: # Rule-Based controller
    """Simple rule-based controller.
    Turns heating on if temperature < T_min.
    Turns heating off if temperature >= T_max.
    """
    def __init__(self, n_zones, T_min, T_max):
        # TODO: Enable arrays T_min and T_max (e.g. at night). Require timestamp
        self.zones = n_zones
        self.T_min, self.T_max = T_min, T_max
        self.heating_on = np.ones(n_zones, dtype=bool)

    def rule_based_control_by_zone(self, xt):
        xt = np.asarray(xt)
        # turn on where currently off AND below min
        to_on = (~self.heating_on) & (xt < self.T_min)
        # turn off where currently on  AND above max
        to_off = self.heating_on  & (xt >= self.T_max)
        self.heating_on[to_on]  = True
        self.heating_on[to_off] = False
        return self.heating_on.astype(int) # ut

    def predict(self, observations):
        xt = observations.x[-1]  # Get the current state
        ut = self.rule_based_control_by_zone(xt)
        return ut

class MPC: # Model Predictive Controller
    """MPC controller using an ARMAX model."""
    def __init__(self, armax_config, target_temperature, T_min, T_max, history_length, horizon_length, objective='tracking'):
        # TODO: Enable arrays T_min and T_max (e.g. at night). Require timestamp
        #       For now, T_min and T_max are fixed for the whole horizon.
        self.mpc = init_mpc_with_armax(armax_config, target_temperature, T_min, T_max, history_length, horizon_length, objective)
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.results = {
            'temperature': [],
            'control_action': [],
            'ambient_temperature': [],
            'solar_irradiance': [],
            'solving_time': [],
            'slack': [],
        }

    def predict(self, observations, solver_name='gurobi'):
        x = observations.x     # state history
        u = observations.u     # control history
        T_amb = observations.a # ambient temperature (history and forecast)
        Q_irr = observations.s # solar irradiance (history and forecast)
        update_mpc_parameters(self.mpc, x, u, T_amb, Q_irr, self.history_length)
        self.solving_time = solve_mpc(self.mpc, solver_name)
        u_optimal = [self.mpc.u[0, k].value for k in self.mpc.zone_range]
        return u_optimal

    def get_values(self):
        x = {t: {r: self.mpc.x[t, r].value for r in self.mpc.zone_range} for t in self.mpc.time_state}
        u = {t: {r: self.mpc.u[t, r].value for r in self.mpc.zone_range} for t in self.mpc.time_input}
        a = {t: self.mpc.a[t].value for t in self.mpc.time_input}
        s = {t: {j: self.mpc.s[t, j].value for j in self.mpc.solar_terms} for t in self.mpc.time_input}
        slack = {t: {r: self.mpc.slack[t, r].value for r in self.mpc.zone_range} for t in self.mpc.time_horizon}
        return x, u, a, s, slack
    
    def save_episode(self):
        x, u, a, s, slack = self.get_values()
        self.results['temperature'].append(x)
        self.results['control_action'].append(u)
        self.results['ambient_temperature'].append(a)
        self.results['solar_irradiance'].append(s)
        self.results['solving_time'].append(self.solving_time)
        self.results['slack'].append(slack)
    