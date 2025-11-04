import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.env import Env
from src.agents import RB, MPC
from src.flex import (plot_energy_bounds_from_file)

BASE_DIR = "/Users/edouardpaupe/Desktop/magnify-main_DATABASE"
POWER_BOUNDS_DIR = os.path.join(BASE_DIR, "data/power_bounds/")
FLEX_ENV_DIR = os.path.join(BASE_DIR, "data/flex_env/")
FLEX_IMG_DIR = os.path.join(BASE_DIR, "data/flex_env_images/")
STEPS_PER_HOUR = 4

try:
    fig, ax = plot_energy_bounds_from_file(
        building_num=1241,
        climate_id=0,
        day=2,
        month=1,
        year=2020,
        ep_idx=20,
        power_bounds_dir=POWER_BOUNDS_DIR,
        steps_per_hour=STEPS_PER_HOUR
    )
    plt.show()
except (FileNotFoundError, ValueError, IndexError) as e:
    print(f"Error: {e}")