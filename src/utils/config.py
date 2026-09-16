# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# load .env
load_dotenv()

# Test-Retest Database
DB_TEST_RETEST_PATH = os.getenv("DB_TEST_RETEST_PATH")

# Test-Retest with Gedai preprocessing Database
DB_TEST_RETEST_GEDAI_PATH = os.getenv("DB_TEST_RETEST_GEDAI_PATH")

# ANPHY-Sleep Database
DB_ANPHY_PATH = os.getenv("DB_ANPHY_PATH")

# Siena Scalp EEG Database
DB_SIENA_PATH = os.getenv("DB_SIENA_PATH")

# Ludovico 01 Database
DB_LUDOVICO_01_PATH = os.getenv("DB_LUDOVICO_01_PATH")
DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH = str(
    Path(DB_LUDOVICO_01_PATH) / "structural.csv"   # ajusta a la ruta real
)

# Cache and results
DATA_SIMULATIONS_CACHE_PATH = os.getenv("DATA_SIMULATIONS_CACHE_PATH")
BASE_CACHE_PATH = os.getenv("BASE_CACHE_PATH")
BASE_RESULTS_PATH = os.getenv("BASE_RESULTS_PATH")
BASE_PARAMS_FILE = os.getenv("BASE_PARAMS_FILE")

# Batch runs
DEFAULT_BATCH_RUNS_WORKERS = int(os.getenv("DEFAULT_BATCH_RUNS_WORKERS", 1))

os.makedirs(BASE_CACHE_PATH, exist_ok=True)
os.makedirs(BASE_RESULTS_PATH, exist_ok=True)
os.makedirs(BASE_PARAMS_FILE, exist_ok=True)

# Logging
LOGGING_BASE_PATH = os.getenv("LOGGING_BASE_PATH")
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")

os.makedirs(LOGGING_BASE_PATH, exist_ok=True)


# Configuration parameters for simulations
BURN_RATIO = 0.1
DT = 0.001
TIME=1800*2
SEED = 123
DEGREE = 2

CONFIGS = {
    'base_2D_model':{
        'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'bins': np.array([50, 50]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    },
    'ou': {
        'model': 'ou', 'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'theta': np.array([[1.0, 0.2], [0.2, 0.8]]), 'sigma': np.array([[0.5, 0.0], [0.0, 0.3]])},
        'bins': np.array([100, 100]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    },
    'single_well': {
        'model': 'single_well', 'D': 1, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'k': 1.0, 'sigma': 0.5},
        'bins': np.array([200]),
        'drift_components': [0], 'diff_components': [(0,0)],
        'degree': DEGREE,
    },
    'asymmetric_double_well': {
        'model': 'asymmetric_double_well', 'D': 1, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'a': 1.0, 'b': 0.5, 'c': -1.0, 'd': 0.2, 'sigma': 0.5},
        'bins': np.array([200]),
        'drift_components': [0], 'diff_components': [(0,0)],
        'degree': DEGREE,
    },
    'double_well': {
        'model': 'double_well', 'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'a': 1.0, 'b': 1.0, 'c': 0.1, 'sigma': 0.5},
        'bins': np.array([50,50]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    },
    'triple_well_3d': {
        'model': 'triple_well_3d', 'D': 3, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'a': 1.0, 'b': 1.0, 'c': 0.3, 'k_rest': 1.0, 'sigma': 0.4},
        'bins': np.array([40, 40, 40]),
        'drift_components': [0, 1, 2], 'diff_components': [(0,0), (1,1), (2,2), (0,1), (0,2),(1,2)],
        'degree': DEGREE,
    },
    'ring_attractor': {
        'model': 'ring_attractor', 'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'alpha': 1.0, 'r0': 1.0, 'omega': 2.0, 'sigma': 0.3},
        'bins': np.array([100, 100]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    },
    'multi_stable': {
        'model': 'multi_stable', 'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'a': 1.0, 'b': 1.0, 'c': 0.5, 'sigma': 0.5},
        'bins': np.array([100, 100]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    },
    'stochastic_oscillator': {
        'model': 'stochastic_oscillator', 'D': 2, 'dt': DT, 'T': TIME, 'burn_ratio': BURN_RATIO, 'seed': SEED,
        'params': {'lambda': 1.0, 'omega': 2.0, 'sigma': 0.3},
        'bins': np.array([60, 60]),
        'drift_components': [0, 1], 'diff_components': [(0,0), (1,1), (0,1)],
        'degree': DEGREE,
    }
}