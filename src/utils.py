import os
import re
import datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from typing import Callable, Tuple

def round_nb(x, sig=3):
    # rounds to a given number of significant digits
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
# Vectorize the function so it works on arrays
round_vec = np.vectorize(round_nb)

def get_list_of_buildings(models_dir): #parses building folder names to create a table of building attributes (type, age, climate, id, folder).
    '''Get the list if available buildings'''
    # Find type, age, climate, and number. 
    data = []
    for name in models_dir:
        match = re.match(r'ep_(\w+)_age_(\d+)_climate_(\d+)_(\d+)$', name)
        if match:
            data.append(match.groups() + (name,))
    return pd.DataFrame(data, columns=['type', 'age', 'climate', 'id', 'folder'])


def load_armax_model(building_folder_path: str, 
                        steps_per_hour: int = 4,
                        lag_hours_list:  list = [0.5, 1, 2, 8],
                        solar_terms: int = 9) -> Dict:
    '''
    Load ARMAX model parameters. Each building has 10 models calibrated for 
    different climate zone.
    
    Args:
        building_folder_path: Path to building archetype folder
        steps_per_hour: Time discretization (4 = 15min steps)
        lag_hours_list: Custom lag hours (default: [0.5, 1, 2, 8])
        lag hours are used to compute lag terms in the ARMAX model.
        solar_terms: Number of solar irradiance terms
        
    Returns:
        Dict containing all ARMAX model configuration
    '''
    
    # Find the filenames for weights and bias/intercept in the building folder
    weight_path, bias_path = _get_weight_and_bias_paths(building_folder_path)
    # Build full paths to the weights and intercept files
    weights_path = os.path.join(building_folder_path, weight_path)
    intercepts_path = os.path.join(building_folder_path, bias_path)
    # Load parameter names, ARMAX coefficients, and intercepts from CSV files
    param_names, armax_coef, armax_intercept = _load_armax_parameters(weights_path, intercepts_path)

    # Find all zone temperature variable names (e.g., T1, T2, ...) for multi-zone buildings
    zone_names = [name for name in param_names if re.match(r'^T\d+$', name)]
    n_zones = len(zone_names)  # Number of zones in the building

    # Get the column positions for key variables in the parameter list
    pos_T1 = param_names.get_loc('T1')        # Position of first zone temperature
    pos_u1 = param_names.get_loc('u1')        # Position of first zone control input (valve)
    pos_T_amb = param_names.get_loc('T_amb')  # Position of ambient temperature
    pos_s1 = param_names.get_loc('s1')        # Position of first solar bin

    # Validate lag hours: only allow 0.5, 1, 2, 8 hours
    if not all(lag in [0.5, 1, 2, 8] for lag in lag_hours_list):
        raise ValueError(
                f'Invalid lag hours in {lag_hours_list}. Only 0.5, 1, 2, 8 are allowed.')
    lag_hours_list = np.array(lag_hours_list)
    # Convert lag hours to number of time steps (e.g., 0.5h * 4 steps/h = 2 steps)
    lag_terms_list = (lag_hours_list * steps_per_hour).astype(int).tolist()
    # List of lag terms for ARMAX dynamics (always includes lag 1)
    dynamics_lags_list = [1] + lag_terms_list
    # Feature vector length: number of parameters per lag
    feat_length = len(param_names) // len(dynamics_lags_list)

    # Build the ARMAX model configuration dictionary
    armax_model_config = {
        # Model parameters for optimization
        'armax_coef': armax_coef,               # ARMAX model coefficients (used in prediction/optimization)
        'armax_intercept': armax_intercept,     # ARMAX model intercepts (bias terms)
        'feat_length': feat_length,             # Number of features per lag (used to build input vector)
        'solar_terms': solar_terms,             # Number of solar irradiance bins/terms (for solar input encoding), it is discreetized
        
        # Parameter Positions of key variables in the feature vector
        'pos_T1': pos_T1,                       # Index of first zone temperature in feature vector
        'pos_u1': pos_u1,                       # Index of first zone control input in feature vector
        'pos_T_amb': pos_T_amb,                 # Index of ambient temperature in feature vector
        'pos_s1': pos_s1,                       # Index of first solar bin in feature vector
        
        # Time-related parameters for optimization horizon and history
        'dynamics_lags_list': dynamics_lags_list, # List of lag steps used in ARMAX model (e.g., [1, 2, 4, 8, 32])
        'steps_per_hour': steps_per_hour,         # Number of time steps per hour (discretization), 4 means 15min steps
        
        # Metadata for configuration and reproducibility
        'n_zones': n_zones,                       # Number of zones (rooms) in the building (for multi-zone control)
        'building_folder': building_folder_path,   # Path to the building folder (for reference)
        'lag_hours_list': lag_hours_list.tolist(), # List of lag hours used (for documentation)
        'param_names': param_names.tolist(),       # List of all parameter names (for reference)
        
        # Model validation info
        'n_parameters': len(armax_coef),           # Total number of ARMAX parameters
        'model_type': 'ARMAX'                      # Model type identifier (for compatibility)
    }
    
    # Return the configuration dictionary for use in the optimization framework
    return armax_model_config

def _get_weight_and_bias_paths(folder_path: str) -> Tuple[str, str]:
    '''Find weight and bias files in the folder'''
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f'Folder {folder_path} does not exist.')
        
    list_files = os.listdir(folder_path)
    weight_path, bias_path = None, None
    
    for file_ in list_files:
        if 'weights' in file_.lower():
            weight_path = file_
        elif 'bias' in file_.lower():
            bias_path = file_

    if not weight_path or not bias_path:
        raise FileNotFoundError(f'Could not find files in {folder_path}.')
    
    return weight_path, bias_path

def _load_armax_parameters(weights_path: str, intercepts_path: str
                           ) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    '''Load ARMAX parameters from CSV files'''
    weights_df = pd.read_csv(weights_path, index_col=0)
    intercept_df = pd.read_csv(intercepts_path, index_col=0)
    param_names = weights_df.columns
    armax_coef = round_vec(weights_df.to_numpy().T)
    armax_intercept = round_vec(intercept_df.to_numpy())
    
    return param_names, armax_coef, armax_intercept

def load_weather_data(file_path: str, climate_id: int = 0) -> pd.DataFrame:
    """
    This function loads weather data from an EPW file for a given climate zone and parses it into a pandas DataFrame.
    It extracts the timestamp, ambient temperature, and solar irradiance for each time step, resamples the data to a 15-minute resolution,
    and interpolates missing values.
    """

    file_path = os.path.join(file_path, f'current_CL_{climate_id}_weighted.epw')

    with open(file_path, 'r') as input_file:
        all_lines = input_file.readlines()

    # Find start of data section
    header_index = next(
        (i for i, line in enumerate(all_lines) if 'DATA PERIODS' in line), None
    )
    if header_index is None:
        raise ValueError("Header line 'DATA PERIODS' not found in the file.")

    time_vec, temp_vec, irrd_vec = [], [], []
    for line in all_lines[header_index + 1:]:
        elements = [e.strip() for e in line.split(',')]
        time_ = datetime.datetime(
            int(elements[0]),        # year
            int(elements[1]),        # month
            int(elements[2]),        # day
            int(elements[3]) % 24,   # hour
            0, 0
        )
        ambient_temp = float(elements[6])       # ambient temperature in °C
        solar_irradiance = float(elements[13])  # solar radiation in W/m2

        time_vec.append(time_)
        temp_vec.append(ambient_temp)
        irrd_vec.append(solar_irradiance)

    weather_df = pd.DataFrame({
                                'time': time_vec,
                                'T_amb': temp_vec,
                                'irrad': irrd_vec
                            }).set_index('time')

    # Resample to 15-minute resolution
    weather_df = weather_df.resample('15min')
    
    # Interpolate missing values
    weather_df = weather_df.interpolate(method='linear', max_periods=3)

    return weather_df

def add_solar_bins(weather_df, steps_per_hour=4):
    solar_irradiance = weather_df['irrad'].to_numpy()
    bins = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
    sol_mat = np.tile(solar_irradiance.reshape(-1, 1), (1, len(bins)-1))
    hours = ((np.arange(len(solar_irradiance)) // steps_per_hour) % 24).reshape(-1, 1)
    enc_mat = (hours >= bins[:-1]) & (hours < bins[1:])
    Q_irr = np.multiply(sol_mat, enc_mat)

    colnames = [f'solar_bin_{low}-{high}' for low, high in zip(bins[:-1], bins[1:])]
    weather_df[colnames] = Q_irr
    return weather_df

def prepare_weather_data(df):
    """Convert weather DataFrame to NumPy array and create index/column maps."""
    array = df.to_numpy()
    idx_map = {ts: i for i, ts in enumerate(df.index)}
    col_map = {col: i for i, col in enumerate(df.columns)}
    return array, idx_map, col_map

def get_weather_forecasts(
    current_time,
    array,
    idx_map,
    col_map,
    history_length,
    horizon_length,
    T_col='T_amb',
    Q_prefix='solar_bin_'
):
    # turn datetime → integer index
    current_index    = idx_map[current_time]
    start = current_index - history_length
    end   = current_index + horizon_length

    # slice out T_amb column by name
    T_amb = array[start:end, col_map[T_col]]

    # find all solar‐bin columns by their prefix
    Q_cols = [ci for c, ci in col_map.items() if c.startswith(Q_prefix)]
    Q_irr  = array[start:end, Q_cols]

    return T_amb, Q_irr

def get_weather_forecasts_slower(
                        current_time: datetime.datetime,
                        weather_df: pd.DataFrame,
                        history_hours: int = 8, 
                        horizon_hours: int = 24,
                        steps_per_hour: int = 4,
                        solar_bins: list = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23], 
                        ) -> Tuple[np.ndarray, np.ndarray]:
    
    # Find the index of the current time
    if current_time not in weather_df.index:
        raise ValueError(f'{current_time} is not in the weather_df index.')
    current_index = weather_df.index.get_loc(current_time)
    idx_start = current_index - history_hours * steps_per_hour
    idx_end = current_index + horizon_hours * steps_per_hour

    # Check if we have enough data
    required_history_steps = history_hours * steps_per_hour
    required_horizon_steps = horizon_hours * steps_per_hour
    if current_index < required_history_steps:
        print(f'Warning: Not enough data. Required: {required_history_steps}',
              f'available: {current_index}.')
    if current_index + required_horizon_steps > len(weather_df):
        print('Warning: Not enough data. Required end index:',
              f'{current_index + required_horizon_steps}, available: {len(weather_df)}')

    # Extract the relevant data block
    data_block = weather_df.iloc[idx_start:idx_end, :]

    #  Temperature Forecast
    T_amb = data_block['T_amb'].to_numpy()

    # Solar Forecast
    Q_irr = data_block['irrad'].to_numpy()

    # Create an encoding matrix using one-hot encoding for solar terms
    bins = np.array(solar_bins)
    sol_mat = np.tile(Q_irr.reshape(-1, 1), (1, len(bins)-1))
    hours = ((np.arange(idx_start, idx_end) // steps_per_hour) % 24).reshape(-1, 1)
    enc_mat = (hours >= bins[:-1]) & (hours < bins[1:])
    Q_irr = np.multiply(sol_mat, enc_mat)

    return T_amb, Q_irr


def plot_weather_forecasts(T_amb, Q_irr):
    colors =[
        '#FF6F61',  # Saturated Red
        '#89CFF0',  # Saturated Blue
        '#FFB347',  # Saturated Orange
        '#FFD700',  # Saturated Yellow
        '#77DD77',  # Saturated Green
        '#CBAACB',  # Saturated Purple
        '#FF69B4',  # Saturated Pink
        '#F08080',  # Light Coral
        '#CD5C5C',  # Indian Red
        '#E9967A',  # Dark Salmon
        '#FFA07A',  # Light Salmon
    ]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6), dpi=300)
    ax[0].plot(T_amb, label='T_amb', color=colors[0])
    ax[0].set_title('Ambient Temperature Forecast')
    ax[0].set_ylabel('Temperature (°C)')
    ax[0].legend()
    for i in range(Q_irr.shape[1]):
        ax[1].plot(Q_irr[:, i], label=f'Solar Term {i+1}', color=colors[i+1])
    ax[1].set_title('Solar Irradiance Forecast')
    ax[1].set_xlabel('Time Steps')
    ax[1].set_ylabel('Irradiance (W/m²)')
    ax[1].legend()
    for ax_ in ax:
        ax_.grid()
    plt.tight_layout()
    plt.show()

def plot_episode(
                timestamps,
                building_temperature,
                valve_control,
                ambient_temperature, 
                solar_irradiance,
                T_min=None,
                T_max=None,
                slack_variable=None,
                labels=None,
                ):
    colors =[
        '#FF6F61',  # Saturated Red
        '#89CFF0',  # Saturated Blue
        '#FFB347',  # Saturated Orange
        '#FFD700',  # Saturated Yellow
        '#77DD77',  # Saturated Green
        '#CBAACB',  # Saturated Purple
        '#FF69B4',  # Saturated Pink
        '#F08080',  # Light Coral
        '#CD5C5C',  # Indian Red
        '#E9967A',  # Dark Salmon
        '#FFA07A',  # Light Salmon
    ]
    fontsize = 12
    subfontsize = 12
    lw = 2

    building_temperature = np.array(building_temperature)
    valve_control = np.array(valve_control)
    ambient_temperature = np.array(ambient_temperature)
    solar_irradiance = np.array(solar_irradiance)

    # Create the figure and axis objects
    labels = labels or [f'Zone {i+1}' for i in range(valve_control.shape[1])]
    _, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
    for i in range(3):
        axs[i].grid()
        axs[i].tick_params(axis='both', which='major', labelsize=12)

    # Plot temperatures
    for i in range(building_temperature.shape[1]):
        color_idx = i % len(colors)
        axs[0].plot(timestamps, building_temperature[:, i], 
                    color=colors[color_idx], lw=lw, label=labels[i])
    if T_min is not None and T_max is not None:
        axs[0].fill_between(
            timestamps, T_min, T_max, color='gray', alpha=0.1#, label='Comfort bounds'
        )
    axs[0].set_ylabel('Building temperature (°C)', fontsize=subfontsize)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=5, fontsize=fontsize)

    # Plot valve opening
    for i in range(valve_control.shape[1]):
        color_idx = i % len(colors)
        axs[1].plot(timestamps, valve_control[:, i], color=colors[color_idx], lw=lw)
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_ylabel('Valve opening (%)', fontsize=subfontsize)
    if slack_variable is not None:
        slack_variable = np.array(slack_variable)
        axs1 = axs[1].twinx()  # create a twin axis for the second plot
        for i in range(slack_variable.shape[1]):
            color_idx = i % len(colors)
            axs1.plot(timestamps, slack_variable[:, i], 
                      color=colors[color_idx], linestyle=':', lw=lw)
        axs1.set_ylabel('Slack variable', fontsize=subfontsize)

    # Plot solar irradiance and outside temperature
    axs[2].plot(timestamps, solar_irradiance, color='#FFD700', lw=lw)
    axs[2].set_ylabel('Solar Irradiance (W/m²)', color='#FFC700', fontsize=fontsize)
    axs2 = axs[2].twinx()  # create a twin axis for the second plot
    axs2.plot(timestamps, ambient_temperature, color='#FF6F61', lw=lw)
    axs2.set_ylabel('Outside Temperatures (°C)', color='#FF6F61', fontsize=fontsize)
    axs[2].set_xlabel('Time', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


