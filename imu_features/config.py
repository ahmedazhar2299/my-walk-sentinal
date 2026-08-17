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
class WaveletStepConfig:
    resample_fs_hz: int = 10
    walk_min_amp_threshold: float = 0.3  # fallback only if robust p2p threshold cannot be computed
    walk_threshold_k: float = 0.25
    min_active_windows: int = 3
    step_freq_min_hz: float = 0.8
    step_freq_max_hz: float = 2.3
    alpha: float = 0.6
    beta: float = 2.5
    delta: int = 20


@dataclass
class TurnPauseConfig:
    adaptive_beta: float = 0.6


@dataclass
class WindowGateConfig:
    window_sec: float = 1.0
    turn_window_sec: float = 0.4
    p2p_cap_scale: float | None = 0.3  # set None to disable median + K*MAD cap
    walk_min_amp_threshold: float = 0.3  # fallback only if robust p2p threshold cannot be computed
    walk_threshold_k: float = 0.25
    turn_min_amp_threshold: float = 0.1  # fallback only if robust p2p threshold cannot be computed
    turn_threshold_k: float = 5.0
    turn_step_peak_percentile: float = 55.0
    turn_step_min_interval_s: float = 0.50
    transition_min_amp_threshold: float = 0.1  # fallback only if robust p2p threshold cannot be computed
    transition_threshold_k: float = 3.0
    walk_min_duration_s: float = 1.0
    turn_min_duration_s: float = 0.8
    transition_min_duration_s: float = 0.5


@dataclass
class PipelineConfig:
    dataset_root: str = "Data"
    output_csv: str = "features_dataset.csv"
    walk_distance_m: float = 10.0
    min_rows_per_activity: int = 5
    prefer_useracc_for_motion: bool = True
    column_candidates: dict = field(default_factory=_default_column_candidates)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    wavelet_steps: WaveletStepConfig = field(default_factory=WaveletStepConfig)
    turn_pause: TurnPauseConfig = field(default_factory=TurnPauseConfig)
    window_gate: WindowGateConfig = field(default_factory=WindowGateConfig)

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
