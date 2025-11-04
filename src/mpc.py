import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from functools import partial

def armax_model(mpc, t, r, feat_length, pos_T1, pos_u1, pos_T_amb, pos_s1, dynamics_lags_list, armax_coef, armax_intercept):
    if t < 1:
        return pyo.Constraint.Skip
    expr = 0
    for i, lag in enumerate(dynamics_lags_list):
        idx = i * feat_length
        expr += sum(armax_coef[idx + pos_T1 + rr, r] * mpc.x[t - lag, rr] for rr in mpc.zone_range)
        expr += sum(armax_coef[idx + pos_u1 + rr, r] * mpc.u[t - lag, rr] for rr in mpc.zone_range)
        expr += armax_coef[idx + pos_T_amb, r] * mpc.a[t - lag]
        expr += sum(armax_coef[idx + pos_s1 + j, r] * mpc.s[t - lag, j] for j in mpc.solar_terms)
    expr += armax_intercept[r]
    return mpc.x[t, r] == expr

def T_bounds_min(mpc, t, r):
    return mpc.x[t, r] >= mpc.T_min[t, r] - mpc.slack[t, r]

def T_bounds_max(mpc, t, r):
    return mpc.x[t, r] <= mpc.T_max[t, r] + mpc.slack[t, r]

def init_mpc_with_armax(armax_config, target_temperature, T_min, T_max, history_length, horizon_length, objective='tracking'):
    zones = armax_config.get('n_zones')
    solar_terms = armax_config.get('solar_terms')
    alpha_factor = 3 / (horizon_length) # for upper and lower bounds

    mpc = pyo.ConcreteModel()
    
    # Sets
    mpc.zone_range = pyo.RangeSet(0, zones - 1) # [0, 9]
    mpc.time_horizon = pyo.RangeSet(0, horizon_length)
    mpc.time_horizon_input = pyo.RangeSet(0, horizon_length-1)
    mpc.time_state = pyo.RangeSet(-history_length, horizon_length)
    mpc.time_input = pyo.RangeSet(-history_length, horizon_length - 1)
    mpc.solar_terms = pyo.Set(initialize=range(solar_terms), ordered=True)
    
    # Variables
    mpc.u = pyo.Var(mpc.time_input, mpc.zone_range, bounds=(0, 1))
    mpc.x = pyo.Var(mpc.time_state, mpc.zone_range)
    mpc.slack = pyo.Var(mpc.time_horizon, mpc.zone_range, within=pyo.NonNegativeReals)
    
    # External inputs
    mpc.a = pyo.Param(mpc.time_input, mutable=True, default=0.0)
    mpc.s = pyo.Param(mpc.time_input, mpc.solar_terms, mutable=True, default=0.0)
    mpc.T_min = pyo.Param(mpc.time_horizon, mpc.zone_range, initialize=lambda m, t, r: T_min[t, r], mutable=False)
    mpc.T_max = pyo.Param(mpc.time_horizon, mpc.zone_range, initialize=lambda m, t, r: T_max[t, r], mutable=False)
    mpc.T_target = pyo.Param(mpc.time_horizon, mpc.zone_range, default=target_temperature, mutable=False)
    
    # Constraints
    mpc.dynamics = pyo.Constraint(mpc.time_horizon, mpc.zone_range, rule=partial(
        armax_model,
        feat_length=armax_config.get('feat_length'),
        pos_T1=armax_config.get('pos_T1'),
        pos_u1=armax_config.get('pos_u1'),
        pos_T_amb=armax_config.get('pos_T_amb'),
        pos_s1=armax_config.get('pos_s1'),
        dynamics_lags_list=armax_config.get('dynamics_lags_list'),
        armax_coef=armax_config.get('armax_coef'),
        armax_intercept=armax_config.get('armax_intercept'),
    ))
    mpc.bound_min = pyo.Constraint(mpc.time_horizon, mpc.zone_range, rule=T_bounds_min)
    mpc.bound_max = pyo.Constraint(mpc.time_horizon, mpc.zone_range, rule=T_bounds_max)

    # Objective
    if objective=='tracking':
        mpc.objective = pyo.Objective(expr=
            sum(
                sum(mpc.slack[t, r] for t in mpc.time_horizon) +
                sum((mpc.x[t, r] - mpc.T_target[t, r])**2 for t in mpc.time_horizon)
                for r in mpc.zone_range
            ), sense=pyo.minimize
        )
    elif objective=='upper_bound':
        mpc.objective = pyo.Objective(expr=
            sum(
                100*sum(mpc.slack[l, r] for l in mpc.time_horizon)
                - sum(mpc.u[t, r] * pyo.exp(- alpha_factor * t) for t in mpc.time_horizon_input)
                for r in mpc.zone_range
            ), sense=pyo.minimize
        )
    elif objective=='lower_bound':
        mpc.objective = pyo.Objective(expr=
            sum(
                100*sum(mpc.slack[l, r] for l in mpc.time_horizon)
                + sum(mpc.u[t, r] * pyo.exp(- alpha_factor * t) for t in mpc.time_horizon_input)
                for r in mpc.zone_range
            ), sense=pyo.minimize
        )
    else:
        raise ValueError("Invalid objective type. Choose 'tracking', 'upper_bound', or 'lower_bound'.")
    
    return mpc

def update_mpc_parameters(mpc, history_T, history_u, T_amb, Q_irr, history_length):
    # Fix current state at t=0
    for r in mpc.zone_range:
        mpc.x[0, r].fix(history_T[-1,r])

    # Fix historical values for t < 0
    for r in mpc.zone_range:
        for i, t in enumerate(range(-history_length, 0)):
            mpc.u[t, r].fix(history_u[i, r])
            mpc.x[t, r].fix(history_T[i, r])
    
    # Unfix future values (good practice)
    for r in mpc.zone_range:
        for t in mpc.time_horizon:
            mpc.slack[t, r].unfix()
            if t > 0:
                mpc.x[t, r].unfix()
            if t < len(mpc.time_horizon) -1:
                mpc.u[t, r].unfix()
    
    # Update weather parameters
    for i, t in enumerate(mpc.time_input):
        mpc.a[t] = T_amb[i]
        for j in mpc.solar_terms:
            mpc.s[t, j] = Q_irr[i, j]

def solve_mpc(mpc, solver_name='gurobi'):
    results = SolverFactory(solver_name).solve(mpc, tee=False)
    return results.solver.wallclock_time #results.solver.time

def predict(mpc, history_T, history_u, T_amb, Q_irr, history_length):
    update_mpc_parameters(mpc, history_T, history_u, T_amb, Q_irr, history_length)
    solve_mpc(mpc)
    u_optimal = [mpc.u[0, k].value for k in mpc.zone_range]
    return u_optimal

def get_values(mpc):
    u_values = {t: {r: pyo.value(mpc.u[t, r]) for r in mpc.zone_range} for t in mpc.time_input}
    x_values = {t: {r: pyo.value(mpc.x[t, r]) for r in mpc.zone_range} for t in mpc.time_state}
    return u_values, x_values

def get_values_as_list(mpc):
    x = [[mpc.x[t, r].value for r in mpc.zone_range] for t in mpc.time_state]
    u = [[mpc.u[t, r].value for r in mpc.zone_range] for t in mpc.time_input]
    a = [mpc.a[t].value for t in mpc.time_input]
    s = [[mpc.s[t, j].value for j in mpc.solar_terms] for t in mpc.time_input]
    slack = [[mpc.slack[t, r].value for r in mpc.zone_range] for t in mpc.time_horizon]
    return x, u, a, s, slack

