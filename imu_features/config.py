import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _default_column_candidates():
    return {
        "timestamp": ["Timestamp", "timestamp", "time", "Time"],
        "accel_x": ["Accel_X", "accel_x", "acc_x"],
        "accel_y": ["Accel_Y", "accel_y", "acc_y"],
        "accel_z": ["Accel_Z", "accel_z", "acc_z"],
        "useracc_x": ["UserAccel_X", "useracc_x", "linacc_x", "linearacc_x"],
        "useracc_y": ["UserAccel_Y", "useracc_y", "linacc_y", "linearacc_y"],
        "useracc_z": ["UserAccel_Z", "useracc_z", "linacc_z", "linearacc_z"],
        "gyro_x": ["Gyro_X", "gyro_x", "gyr_x"],
        "gyro_y": ["Gyro_Y", "gyro_y", "gyr_y"],
        "gyro_z": ["Gyro_Z", "gyro_z", "gyr_z"],
    }


@dataclass
class FilterConfig:
    enabled: bool = True
    cutoff_hz: float = 5.0
    order: int = 4


@dataclass
class StepDetectionConfig:
    min_distance_s: float = 0.30
    prominence: float = 0.10
    height: float = None


@dataclass
class TurnPauseConfig:
    velocity_threshold: float = 0.20
    min_pause_duration_s: float = 0.20


@dataclass
class TransitionPeakConfig:
    min_distance_s: float = 0.15
    prominence: float = 0.10


@dataclass
class PipelineConfig:
    dataset_root: str = "Data"
    output_csv: str = "features_dataset.csv"
    walk_distance_m: float = 10.0
    min_rows_per_activity: int = 5
    prefer_useracc_for_motion: bool = True
    column_candidates: dict = field(default_factory=_default_column_candidates)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    step_detection: StepDetectionConfig = field(default_factory=StepDetectionConfig)
    turn_pause: TurnPauseConfig = field(default_factory=TurnPauseConfig)
    transition_peaks: TransitionPeakConfig = field(default_factory=TransitionPeakConfig)

    def to_dict(self):
        return asdict(self)


def _update_dataclass(instance, updates):
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current_value = getattr(instance, key)
        if hasattr(current_value, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current_value, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path=None):
    """Load pipeline config from JSON; returns defaults when path is omitted."""
    config = PipelineConfig()
    if not config_path:
        return config

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        updates = json.load(f)
    return _update_dataclass(config, updates)
