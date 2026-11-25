"""Cross-part records. Body coordinates: x right, y forward, cm; time seconds.

Scores are not calibrated probabilities. Geometry and actuator units are explicit.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceSource(str, Enum):
    OBSERVED = "observed"
    MAP_PREDICTED = "map_predicted"
    TEMPORAL_PREDICTED = "temporal_predicted"


@dataclass(frozen=True)
class FrameStamp:
    sequence: int
    capture_s: float | None
    clock_id: str  # sensor_monotonic, video_pts, unavailable; never host processing time


@dataclass
class BoundaryFragment:
    id: str
    points_cm: list[list[float]]
    endpoint_reasons: list[str]
    side_hint: str  # left/right/unknown relative to ordered tangent, not screen x
    quality: dict[str, float]
    closed: bool = False
    source: EvidenceSource = EvidenceSource.OBSERVED


@dataclass
class ConnectionCandidate:
    fragment_a: str
    end_a: int
    fragment_b: str
    end_b: int
    gap_cm: float
    heading_mismatch_deg: float
    status: str = "unobserved_connection_candidate"


@dataclass
class LandmarkCandidate:
    id: str
    kind: str
    center_cm: list[float]
    contour_cm: list[list[float]]
    quality: dict[str, float]
    flags: list[str] = field(default_factory=list)
    map_id: str | None = None  # observation never assigns a persistent map ID
    source: EvidenceSource = EvidenceSource.OBSERVED


@dataclass
class ObservationFrame:
    stamp: FrameStamp
    calibration_id: str
    config_id: str
    fragments: list[BoundaryFragment]
    connections: list[ConnectionCandidate]
    landmarks: list[LandmarkCandidate]
    status: str
    diagnostics: dict[str, Any]

    def to_dict(self):
        return asdict(self)


@dataclass
class PoseHypothesis:
    id: str
    segment_id: str | None
    pose_xytheta: list[float]
    covariance: list[list[float]]
    associations: dict[str, str]
    relative_score: float
    stamp: FrameStamp


@dataclass
class PathDecision:
    points_body_cm: list[list[float]]
    speed_limit_cm_s: float
    valid_until_s: float
    clock_id: str
    reason: str
    reference_lateral_error_cm: float | None = None
    reference_heading_error_rad: float | None = None
    reference_curvature_cm_inv: float | None = None
    lookahead_cm: float | None = None


@dataclass
class AppliedCommand:
    applied_s: float
    clock_id: str
    pwm_by_channel: dict[str, float]
    limited_reason: str | None


@dataclass
class MissionEvent:
    id: str
    kind: str
    visit_id: str | None
    target_id: str | None
    stamp: FrameStamp
    payload: dict[str, Any]
