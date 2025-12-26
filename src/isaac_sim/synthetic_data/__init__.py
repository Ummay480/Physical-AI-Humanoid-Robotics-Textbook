"""Isaac Sim synthetic data module for AI-Robot Brain"""

from .generator import (
    SyntheticDataGenerator,
    DomainRandomizationEngine,
    get_default_synthetic_data_generator,
    generate_synthetic_dataset
)

from .data_generator import (
    SyntheticDataGenerator as PerceptionSyntheticDataGenerator,
    SyntheticDataSample,
    BoundingBox
)

__all__ = [
    'SyntheticDataGenerator',
    'DomainRandomizationEngine',
    'get_default_synthetic_data_generator',
    'generate_synthetic_dataset',
    'PerceptionSyntheticDataGenerator',
    'SyntheticDataSample',
    'BoundingBox'
]