import os
import re
import datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Callable, Sequence

def plot_power_bounds_for_episode(upper_power_bound, lower_power_bound, zone_id, episode_id, env):
    """
    Visualize the upper and lower power bounds for a chosen zone and episode.

    Parameters
    ----------
    upper_power_bound : np.ndarray
        Array of shape (n_zones, n_episodes, horizon_length)
    lower_power_bound : np.ndarray
        Same shape as upper_power_bound
    zone_id : int
        Index of the zone to visualize (0 to n_zones-1)
    episode_id : int
        Index of the MPC episode (0 to n_episodes-1)
    env : Env
        The environment instance (used to extract horizon and time step info)
    """
    
    # Extract relevant data
    upper = upper_power_bound[zone_id, episode_id, :]
    lower = lower_power_bound[zone_id, episode_id, :]
    horizon_length = env.horizon_length
    
    # Build time axis in hours
    time_axis = np.arange(horizon_length) / env.steps_per_hour  # e.g., 0 to 24h
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(time_axis, upper, label='Upper Power Bound', color='tab:red', linewidth=2)
    plt.plot(time_axis, lower, label='Lower Power Bound', color='tab:blue', linewidth=2)
    plt.fill_between(time_axis, lower, upper, color='tab:gray', alpha=0.2)
    
    # Labels and title
    plt.xlabel("Lead Time [hours]")
    plt.ylabel("Power [kW]")
    plt.title(f"Simulated Flexibility Bounds for Zone {zone_id+1} — Episode {episode_id+1}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def extract_daily_zone_bounds(upper_power_bound, lower_power_bound, env, save_dir=None):
    """
    Extract (96, horizon_length) arrays from episode sequences for each zone
    and save them with day numbers from 1–365 (or 366 in leap years).

    Parameters
    ----------
    upper_power_bound : np.ndarray
        Shape (n_zones, n_episodes, n_horizon)
    lower_power_bound : np.ndarray
        Same shape as upper_power_bound
    env : Env
        Environment object containing building_id, climate_id, start_time, end_time
    save_dir : str, optional
        Directory to save arrays as .npy files.

    Returns
    -------
    daily_bounds : dict
        Keys like "build{building_num}_clim{climate_id}_day{day}_zone{j}_UB"
    """
    n_zones, n_episodes, n_horizon = upper_power_bound.shape
    block_size = 96  # one day of 15-min episodes
    n_blocks = n_episodes // block_size

    daily_bounds = {}

    # --- Extract metadata ---
    building_num = int(env.building_id.split('_')[-1])
    climate_id = env.climate_id

    for j in range(n_zones):          # Loop over zones
        for b in range(n_blocks):     # Loop over daily blocks
            start_idx = b * block_size
            end_idx = start_idx + block_size

            ub_chunk = upper_power_bound[j, start_idx:end_idx, :]
            lb_chunk = lower_power_bound[j, start_idx:end_idx, :]

            # Compute day of year (1–365 or 366)
            current_date = (env.start_time + datetime.timedelta(days=b)).date()
            day_of_year = current_date.timetuple().tm_yday

            # Create names
            ub_name = f"build{building_num}_clim{climate_id}_day{day_of_year}_zone{j}_UB"
            lb_name = f"build{building_num}_clim{climate_id}_day{day_of_year}_zone{j}_LB"

            # Store in dictionary
            daily_bounds[ub_name] = ub_chunk
            daily_bounds[lb_name] = lb_chunk

            # Optionally save to disk without overwriting
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

                ub_path = os.path.join(save_dir, f"{ub_name}.npy")
                lb_path = os.path.join(save_dir, f"{lb_name}.npy")

                if not os.path.exists(ub_path):
                    np.save(ub_path, ub_chunk)
                else:
                    print(f"⚠️  Skipped existing file: {ub_path}")

                if not os.path.exists(lb_path):
                    np.save(lb_path, lb_chunk)
                else:
                    print(f"⚠️  Skipped existing file: {lb_path}")

    print(f"Extracted {n_blocks * n_zones * 2} daily arrays "
          f"({n_blocks} days x {n_zones} zones x 2 bounds) "
          f"from {env.start_time.date()} to {env.end_time.date()}")
    
    return daily_bounds


# ---------- Core helpers ----------

def power_to_energy_bounds(p_upper: np.ndarray, p_lower: np.ndarray, dt_h: float):
    """
    Convert power bounds to energy bounds by discrete cumulative sum.
    Inputs: p_upper, p_lower: shape (n_horizon,)
    Returns: (E_upper, E_lower): shape (n_horizon,)
    """
    E_upper = np.cumsum(p_upper) * dt_h #multiply by dt in hours to get kWh
    E_lower = np.cumsum(p_lower) * dt_h #shape (n_horizon,)

    #ensure energy bounds are non-decreasing:
    #E_upper = np.maximum.accumulate(E_upper) #cumulative maximum
    #E_lower = np.maximum.accumulate(E_lower)
    #E_lower = np.minimum(E_lower, E_upper)  #ensure E_lower <= E_upper
    return E_upper, E_lower


def max_duration_for_constant_power(E_upper: np.ndarray, E_lower: np.ndarray,
                                    P: float, dt_h: float) -> float:
    """
    For a constant power level P (kW), compute the maximal sustained duration (hours)
    such that P*tau stays between [E_lower(tau), E_upper(tau)].
    """
    n = len(E_upper) #n_horizon
    #construct the energy trajectory for constant power level P:
    # cumulative energy of constant-P trajectory at each discrete step (1..n)
    tau_grid = (np.arange(1, n + 1)) * dt_h  # hours
    #tau_grid = [0.25, 0.5, 0.75, ..., 24.0]
    E_const = P * tau_grid                    # kWh #shape (n,) --> Energy trajectory, corresponds to a straight line

    # Feasibility at each discrete step (1..n)
    ok = (E_lower <= E_const) & (E_const <= E_upper) #shape (n,)
    #creates an array of booleans, True if feasible at that step
    #it tells us at which future instants the energy trajectory line stays within UB and LB bounds.

    if not np.any(ok): #if no feasible steps at all return duration = 0.0
        return 0.0

    #Longest sustained duration : find the first index where feasibility fails
    first_fail = np.argmax(~ok)  # 0 if ok[0] is False; otherwise index of first False
    #~ok is the negation of ok, so True where feasibility fails
    #np.argmax returns the index of the first occurrence of the maximum value (True=1, False=0)

    if ok.all(): #all steps are feasible
        k = n #duration is full horizon n_horizon
    elif first_fail == 0: #first step is infeasible
        k = 0 #duration is 0
    else:
        k = first_fail  # number of feasible steps

    return k * dt_h #return duration in hours (0.25, 0.5, ..., 24.0)


def slice_for_episode(p_upper_vec: np.ndarray, p_lower_vec: np.ndarray,
                      P_grid: np.ndarray, dt_h: float) -> np.ndarray:
    """
    One slice at a fixed lead time (episode): durations for all P in P_grid.
    Returns durations array shape (len(P_grid),) = (53,) for a given lead time
    """
    E_u, E_l = power_to_energy_bounds(p_upper_vec, p_lower_vec, dt_h)
    return np.array([
        max_duration_for_constant_power(E_u, E_l, P, dt_h)
        for P in P_grid])


# ---------- Envelope builders ----------

def envelope_for_zone_day(ub_day: np.ndarray, lb_day: np.ndarray,
                          dt_h: float = 0.25,
                          P_min: float = 0.0, P_max: float = 3.0, dP: float = 0.5):
    """
    Build the 3D envelope for a single zone over 96 lead times (episodes) in one day.

    Inputs:
      ub_day, lb_day: arrays shape (96, n_horizon) – upper/lower power bounds for each episode
      dt_h: timestep in hours (15 min -> 0.25)
      P grid: from P_min to P_max in steps of dP

    Returns:
      P_grid: (nP,) power levels
      durations: (nP, 96) matrix where column e is the slice at lead time e
    """
    n_episodes, n_hor = ub_day.shape #(96,96)
    assert n_episodes == 96, "Expecting 96 episodes per day."
    assert lb_day.shape == ub_day.shape

    P_grid = np.arange(P_min, P_max + 1e-9, dP) #P_grid.shape = (53,)
    durations = np.zeros((len(P_grid), n_episodes), dtype=float) # (53, 96)

    for e in range(n_episodes): #loop over each episode of the day, meaning each lead time 
        durations[:, e] = slice_for_episode(ub_day[e], lb_day[e], P_grid, dt_h)

    return P_grid, durations


def load_daily_bounds(save_dir):
    """
    Recreate the daily_bounds dictionary from a folder of saved .npy arrays.

    Expected file name pattern:
        build{building_num}_clim{climate_id}_day{day}_zone{zone}_{UB/LB}.npy

    Example:
        build1241_clim0_day3_zone2_UB.npy
        build1241_clim0_day3_zone2_LB.npy

    Parameters
    ----------
    save_dir : str
        Directory containing the .npy files.

    Returns
    -------
    daily_bounds : dict
        Dictionary with keys identical to your save names
        (without the ".npy" extension) and NumPy arrays as values.
    """

    daily_bounds = {}
    files = [f for f in os.listdir(save_dir) if f.endswith(".npy")]

    if not files:
        print(f"No .npy files found in {save_dir}")
        return daily_bounds

    for filename in files:
        filepath = os.path.join(save_dir, filename)
        key = os.path.splitext(filename)[0]   # remove '.npy'
        daily_bounds[key] = np.load(filepath)

    print(f"Loaded {len(daily_bounds)} arrays from {save_dir}")
    return daily_bounds

def process_power_bounds(power_bounds_dir, flex_env_dir, image_dir):
    os.makedirs(flex_env_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(power_bounds_dir) if f.endswith(".npy")])
    ub_files = [f for f in files if f.endswith("_UB.npy")]

    print(f"Found {len(ub_files)} upper/lower bound pair files")

    for ub_file in ub_files:
        base_name = ub_file.replace("_UB.npy", "")
        lb_file = base_name + "_LB.npy"

        ub_path = os.path.join(power_bounds_dir, ub_file)
        lb_path = os.path.join(power_bounds_dir, lb_file)

        if not os.path.exists(lb_path):
            print(f"⚠️  Missing lower bound file for {ub_file}, skipping.")
            continue

        # --- Load arrays ---
        ub_day = np.load(ub_path)
        lb_day = np.load(lb_path)

        # --- Compute flexibility envelope ---
        P_grid, durations = envelope_for_zone_day(
            ub_day, lb_day,
            dt_h=1/4,     # 15-min steps
            P_min=-12, P_max=14, dP=0.5
        )

        # --- Save CSV ---
        csv_path = os.path.join(flex_env_dir, f"{base_name}.csv")
        df = pd.DataFrame(
            durations,
            index=np.round(P_grid, 2),
            columns=[f"LeadTime_{i}" for i in range(durations.shape[1])]
        )
        df.index.name = "Power Level [kW]"
        df.to_csv(csv_path)
        print(f"Saved flexibility envelope CSV: {csv_path}")

        # --- Plot heatmap (fixed color scale 0–24 h) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        vmin, vmax = 0, 24
        im = ax.imshow(
            durations,
            aspect="auto",
            origin="lower",
            extent=[0, durations.shape[1]*0.25, P_grid[0], P_grid[-1]],  # convert x to hours
            cmap="viridis",     # purple→yellow gradient similar to paper
            vmin=vmin, vmax=vmax
        )

        # --- Colorbar ---
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Maximum sustained duration [hour]", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        # --- Labels & title ---
        ax.set_xlabel("Lead time [hour]")
        ax.set_ylabel("Power Level [kW]")
        ax.set_title(f"Flexibility Envelope — {base_name}")
        plt.tight_layout()

        # --- Save figure ---
        img_path = os.path.join(image_dir, f"{base_name}.png")
        plt.savefig(img_path, dpi=300)
        plt.close(fig)

    print("All flexibility envelopes processed successfully!")

def plot_energy_bounds_from_file(building_num, climate_id, day, month, year, ep_idx, power_bounds_dir, steps_per_hour):
    """
    Plot UB/LB energy bounds for a given episode from saved power bound files.
    
    Parameters:
    -----------
    building_num : int
        Building number (e.g., 1241)
    climate_id : int
        Climate zone ID (0-5)
    day : int
        Day of month (1-31)
    month : int
        Month (1-12)
    year : int
        Year (e.g., 2020)
    ep_idx : int, optional
        Episode index within the day (0-95 for 15-min intervals)
    power_bounds_dir : str, optional
        Directory containing saved UB/LB files
    steps_per_hour : int, optional
        Number of time steps per hour (default: 4)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    
    Raises:
    -------
    FileNotFoundError : If UB or LB file is missing
    ValueError : If files don't form a valid UB/LB pair
    """
    # Construct filenames
    ub_name = f"build{building_num}_clim{climate_id}_{day:02d}_{month:02d}_{year}_UB"
    lb_name = f"build{building_num}_clim{climate_id}_{day:02d}_{month:02d}_{year}_LB"
    
    ub_path = os.path.join(power_bounds_dir, f"{ub_name}.npy")
    lb_path = os.path.join(power_bounds_dir, f"{lb_name}.npy")
    
    # Check if both files exist
    if not os.path.exists(ub_path):
        raise FileNotFoundError(f"Upper bound file not found: {ub_path}")
    if not os.path.exists(lb_path):
        raise FileNotFoundError(f"Lower bound file not found: {lb_path}")
    
    # Load data
    try:
        ub = np.load(ub_path)  # shape: (96, horizon_length)
        lb = np.load(lb_path)
    except Exception as e:
        raise ValueError(f"Error loading UB/LB files: {e}")
    
    # Validate shapes match
    if ub.shape != lb.shape:
        raise ValueError(f"Shape mismatch: UB {ub.shape} vs LB {lb.shape}")
    
    # Validate episode index
    n_episodes, horizon_length = ub.shape
    if not (0 <= ep_idx < n_episodes):
        raise IndexError(f"ep_idx {ep_idx} out of range [0, {n_episodes-1}]")
    
    # Extract episode data
    ub_ep = ub[ep_idx, :]
    lb_ep = lb[ep_idx, :]
    
    # Convert power (kW) to cumulative energy (kWh)
    dt_h = 1.0 / steps_per_hour  # 15-min steps -> 0.25 h
    E_upper = np.cumsum(ub_ep) * dt_h
    E_lower = np.cumsum(lb_ep) * dt_h
    
    t_hours = np.arange(horizon_length) * dt_h
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_hours, E_upper, label="E_upper (kWh)", linewidth=2)
    ax.plot(t_hours, E_lower, label="E_lower (kWh)", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel("Cumulative energy [kWh]")
    ax.set_title(f"Energy bounds — episode {ep_idx} | Build {building_num} | Climate {climate_id} | {day:02d}/{month:02d}/{year}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax