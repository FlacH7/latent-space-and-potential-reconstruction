"""Configuración de pytest: variables de entorno y sys.path.

src.utils.config crea directorios en tiempo de importación a partir de
variables de entorno; para que la suite corra en cualquier máquina (CI,
sandbox, clones nuevos) se fijan rutas temporales ANTES de importar
cualquier módulo de src.*, y se inserta la raíz del repo en sys.path.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_tmp = tempfile.mkdtemp(prefix='lpr_test_')
_env_defaults = {
    'DB_TEST_RETEST_PATH': _tmp,
    'DB_TEST_RETEST_GEDAI_PATH': _tmp,
    'DB_ANPHY_PATH': _tmp,
    'DB_SIENA_PATH': _tmp,
    'DB_LUDOVICO_01_PATH': _tmp,
    'DATA_SIMULATIONS_CACHE_PATH': os.path.join(_tmp, 'sim_cache'),
    'BASE_CACHE_PATH': os.path.join(_tmp, 'cache'),
    'BASE_RESULTS_PATH': os.path.join(_tmp, 'results'),
    'BASE_PARAMS_FILE': os.path.join(_tmp, 'params'),
    'LOGGING_BASE_PATH': os.path.join(_tmp, 'logs'),
    'LOGGING_LEVEL': 'WARNING',
}
for _key, _value in _env_defaults.items():
    os.environ.setdefault(_key, _value)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
