import os
import datetime
import numpy as np
from dataclasses import dataclass, field

"""Environment for building temperature control using ARMAX model."""

from src.utils import ( 
    load_armax_model,
    load_weather_data,
    add_solar_bins, 
    prepare_weather_data,
    get_weather_forecasts
)

def armax_step(obs, armax_config):
    # Get the current time step
    t = obs.history_length + 1

    # Get ARMAX parameters
    n_zones = armax_config['n_zones']
    feat_length = armax_config['feat_length']
    pos_T1 = armax_config['pos_T1']
    pos_u1 = armax_config['pos_u1']
    pos_T_amb = armax_config['pos_T_amb']
    pos_s1 = armax_config['pos_s1']
    dynamics_lags_list = armax_config['dynamics_lags_list']
    armax_coef = armax_config['armax_coef']
    armax_intercept = armax_config['armax_intercept']

    # Check if t is within the valid range
    if t < max(dynamics_lags_list):
        raise ValueError(
            f'Time step {t} is too early for ARMAX model.'
            f'Minimum required time step is {max(dynamics_lags_list)}.'
            )
    
    # Run one step of the ARMAX model
    xt = np.zeros((1, n_zones))
    for r in range(n_zones):
        xt[:,r] += armax_intercept[r,0]
        for i, lag in enumerate(dynamics_lags_list):
            idx = i * feat_length
            xt[:,r] += sum(armax_coef[idx + pos_T1 + rr, r] * obs.x[t - lag, rr] for rr in range(n_zones))
            xt[:,r] += sum(armax_coef[idx + pos_u1 + rr, r] * obs.u[t - lag, rr] for rr in range(n_zones))
            xt[:,r] += armax_coef[idx + pos_T_amb, r] * obs.a[t - lag]
            xt[:,r] += sum(armax_coef[idx + pos_s1 + j, r] * obs.s[t - lag, j] for j in range(obs.s.shape[1]))
    return xt

@dataclass
class Observations:
    n_zones: int
    history_length: int
    a: np.ndarray
    s: np.ndarray
    x: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x = np.zeros((self.history_length + 1, self.n_zones)) + 21.0
        self.u = np.zeros((self.history_length + 1, self.n_zones))

    def update_action(self, ut):
        self.u = np.vstack([self.u[1:], np.asarray(ut).reshape(1, -1)])

    def update_state(self, xt, a, s):
        self.x = np.vstack([self.x[1:], np.asarray(xt).reshape(1, -1)])
        self.a = a
        self.s = s

class Env:
    def __init__(self,
                 building_id,
                 climate_id,
                 start_time=None,
                 end_time=None,
                 history_hours=8,
                 horizon_hours=24,
                 steps_per_hour=4,
                 controls_per_hour=4):
        self.building_id = building_id
        self.climate_id = climate_id
        self.history_length = history_hours * steps_per_hour  # 8 hours → 32 steps
        self.horizon_length = horizon_hours * steps_per_hour  # 24 hours → 96 steps
        self.steps_per_hour = steps_per_hour # For ARMAX model
        self.control_step = datetime.timedelta(hours=1 / controls_per_hour)

        # Load ARMAX
        self.armax_config = load_armax_model(
            building_folder_path=os.path.join('armax_models', 'archetypes', building_id),
            steps_per_hour=steps_per_hour
        )
        self.n_zones = self.armax_config['n_zones']

        # Load & prepare weather
        weather = load_weather_data(
            file_path=os.path.join('armax_models','archetypes','meteo','pop_weighted'),
            climate_id=climate_id
        )
        weather = add_solar_bins(weather, steps_per_hour=steps_per_hour)
        self.weather, self.idx_map, self.col_map = prepare_weather_data(weather)

        self.start_time = start_time if start_time else weather.index[0]
        self.end_time = end_time if end_time else weather.index[-1]
        self.current_time = start_time
        self.terminated = False

    def reset(self):
        self.current_time = self.start_time
        self.terminated = False
        # Build initial weather histories
        T_amb, Q_irr = get_weather_forecasts(
            current_time=self.current_time,
            array=self.weather,
            idx_map=self.idx_map,
            col_map=self.col_map,
            history_length=self.history_length,
            horizon_length=self.horizon_length,
        )
        # Init Observations
        self.obs = Observations(
            n_zones=self.n_zones,
            history_length=self.history_length,
            a=T_amb,
            s=Q_irr
        )
        # Init results history
        self.history = {
            'timestamp': [],
            'temperature': [],
            'control_action': [],
            'ambient_temperature': [],
            'solar_irradiance': [],
            'reward': []
        }
        return self.obs, {}

    def step(self, actions):
        # Update observations with the current action
        self.obs.update_action(actions)

        # Predict next temperature with ARMAX
        xt = armax_step(self.obs, self.armax_config)

        # Get next-step weather measures and forecasts
        T_amb, Q_irr = get_weather_forecasts(
            current_time=self.current_time,
            array=self.weather,
            idx_map=self.idx_map,
            col_map=self.col_map,
            history_length=self.history_length,
            horizon_length=self.horizon_length,
        )

        # Update observations with the new state and weather
        self.obs.update_state(xt=xt, a=T_amb, s=Q_irr)

        # Basic reward (e.g. -squared error around 21°C)
        reward = -np.sum((xt - 21.0)**2)

        # Store history for plotting
        self.history['timestamp'].append(self.current_time)
        self.history['temperature'].append(np.asarray(xt).ravel()) # Flatten
        self.history['control_action'].append(np.asarray(actions).ravel())
        self.history['ambient_temperature'].append(T_amb[self.history_length])
        self.history['solar_irradiance'].append(Q_irr[self.history_length])

        # Advance time & check termination
        self.current_time += self.control_step
        if self.current_time >= self.end_time:
            self.terminated = True

        return self.obs, reward, {}, self.terminated, False
    