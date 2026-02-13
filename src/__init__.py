"""
Machine Learning Project Package
Global School Electricity Access Analysis
Author: Oussama Achiban
Master ISI
"""

from .data_preprocessing import get_preprocessor
from .dimensionality_reduction import get_reducer
from .clustering import get_clusterer
from .classical_models import get_classical_models
from .evaluation import get_evaluator, get_mlflow_tracker, get_experiment_organizer

# Optional dependency: torch
try:
    from .neural_network_pytorch import get_neural_network_trainer
except ModuleNotFoundError:
    get_neural_network_trainer = None

__version__ = "1.0.0"
__author__ = "Oussama Achiban"

__all__ = [
    'get_preprocessor',
    'get_reducer',
    'get_clusterer',
    'get_classical_models',
    'get_evaluator',
    'get_mlflow_tracker',
    'get_experiment_organizer',
]

if get_neural_network_trainer is not None:
    __all__.append('get_neural_network_trainer')
