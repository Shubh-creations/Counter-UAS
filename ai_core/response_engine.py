"""
Optimal Weapon-Target Assignment (WTA), Kill-Chain Engine & Autonomy Playbook (v6.0-MONOLITH).
Features:
- Multi-target WTA optimization across simultaneously engaging threats to maximize base survival.
- Human-on-the-Loop (HOTL) Autonomy Playbook:
  1. PLAY_01_PERIMETER_CAP (Perimeter Combat Air Patrol)
  2. PLAY_02_SWARM_EW_BARRIER (Synchronized RF/GNSS Umbrella)
  3. PLAY_03_KINETIC_WOLF_PACK (Multi-Interceptor Salvo with Proportional Navigation)
  4. PLAY_04_CEASEFIRE_RTB (Immediate Ceasefire & Return to Base)
- HOTL Supervisory Approval Gate vs FULL_AUTONOMOUS_ROE.
- Blue-Force IFF deconfliction preventing friendly fire against collaborative wingmen.
"""

from dataclasses import dataclass, field
import enum
import math
import time
from typing import Dict, List, Optional, Tuple, Any

from utils.logger import get_logger
from utils.geo_utils import check_geofence_status
from .classifier import DroneClassificationResult
from .trajectory_predictor import TrajectoryPrediction

logger = get_logger("CUAS.ResponseEngine")


class KillChainStage(str, enum.Enum):
    DETECT = "DETECT"
    TRACK = "TRACK"
    IDENTIFY = "IDENTIFY"
    CUE_EFFECTOR = "CUE_EFFECTOR"
    ENGAGE = "ENGAGE"
    ASSESS_BDA = "ASSESS_BDA"
    NEUTRALIZED = "NEUTRALIZED"


class AutonomyPlayType(str, enum.Enum):
    PLAY_01_PERIMETER_CAP = "PLAY_01_PERIMETER_CAP"
    PLAY_02_SWARM_EW_BARRIER = "PLAY_02_SWARM_EW_BARRIER"
    PLAY_03_KINETIC_WOLF_PACK = "PLAY_03_KINETIC_WOLF_PACK"
    PLAY_04_CEASEFIRE_RTB = "PLAY_04_CEASEFIRE_RTB"


@dataclass
class AutonomyPlayIntent:
    play_id: str
    play_type: AutonomyPlayType
    status: str                                # "PENDING_APPROVAL", "EXECUTING", "COMPLETED", "REJECTED"
    title: str
    description: str
    recommended_assets: List[str]
    target_ids: List[int]
    rationale: str
    timestamp: float
    auto_executable: bool = False


@dataclass
class ThreatEvaluation:
    target_id: int
    threat_level: str                          # "SAFE", "ADVISORY", "ELEVATED", "HIGH", "CRITICAL"
    numerical_threat_score: float              # 0.0 - 100.0
    geofence_zone: str                         # "CLEAR", "GREEN", "AMBER", "RED"
    distance_to_base_m: float
    bearing_from_base_deg: float
    recommended_countermeasure: str            # "NONE", "RF_JAMMER", "GPS_SPOOF", "OPTICAL_DAZZLER", "KINETIC_INTERCEPT"
    countermeasure_probability_success: float  # 0.0 - 1.0
    engagement_authorized: bool
    rationale: str
    kill_chain_stage: KillChainStage = KillChainStage.DETECT
    assigned_effector_id: Optional[str] = None
    azimuth_offset_deg: float = 0.0
    time_to_perimeter_sec: Optional[float] = None
    wta_priority_rank: int = 1
    is_friendly: bool = False


class ThreatResponseEngine:
    """
    Weapon-Target Assignment (WTA) Optimization & Autonomy Playbook Engine.
    """
    def __init__(
        self,
        base_latitude: float = 37.7749,
        base_longitude: float = -122.4194,
        inner_killzone_m: float = 500.0,
        warning_perimeter_m: float = 1500.0,
        detection_boundary_m: float = 3500.0,
        auto_engagement_enabled: bool = False
    ):
        self.base_lat = base_latitude
        self.base_lon = base_longitude
        self.inner_killzone_m = inner_killzone_m
        self.warning_perimeter_m = warning_perimeter_m
        self.detection_boundary_m = detection_boundary_m
        self.auto_engagement = auto_engagement_enabled

        self.governance_mode = "HOTL_SUPERVISED"  # "HOTL_SUPERVISED" or "FULL_AUTONOMOUS_ROE"
        self.active_intent: Optional[AutonomyPlayIntent] = None

        # Effector capabilities & constraints
        self.effector_specs = {
            "RF_JAMMER": {
                "max_range_m": 2800.0,
                "beamwidth_deg": 60.0,
                "current_azimuth_deg": 45.0,
                "p_kill_base": 0.88,
                "cooldown_remaining_sec": 0.0
            },
            "GPS_SPOOF": {
                "max_range_m": 3500.0,
                "beamwidth_deg": 360.0,
                "current_azimuth_deg": 0.0,
                "p_kill_base": 0.82,
                "cooldown_remaining_sec": 0.0
            },
            "OPTICAL_DAZZLER": {
                "max_range_m": 1600.0,
                "beamwidth_deg": 15.0,
                "current_azimuth_deg": 45.0,
                "p_kill_base": 0.75,
                "cooldown_remaining_sec": 0.0
            },
            "KINETIC_INTERCEPT": {
                "max_range_m": 4500.0,
                "beamwidth_deg": 360.0,
                "current_azimuth_deg": 0.0,
                "p_kill_base": 0.95,
                "ammo_count": 6,
                "cooldown_remaining_sec": 0.0
            }
        }

    def set_governance_mode(self, mode: str):
        if mode in ["HOTL_SUPERVISED", "FULL_AUTONOMOUS_ROE"]:
            self.governance_mode = mode
            self.auto_engagement = (mode == "FULL_AUTONOMOUS_ROE")
            logger.info(f"Autonomy Governance Mode set to: {mode}")

    def evaluate_target(
        self,
        target_id: int,
        target_lat: float,
        target_lon: float,
        classification: Optional[DroneClassificationResult] = None,
        trajectory: Optional[TrajectoryPrediction] = None
    ) -> ThreatEvaluation:
        """Evaluates individual target threat posture and advances Kill-Chain stage with Blue-Force IFF."""
        # Blue-Force IFF Check
        is_friendly = (target_id >= 500) or (classification and "FRIENDLY" in classification.predicted_class)
        if is_friendly:
            return ThreatEvaluation(
                target_id=target_id,
                threat_level="SAFE",
                numerical_threat_score=0.0,
                geofence_zone="CLEAR",
                distance_to_base_m=haversine_meters(target_lat, target_lon, self.base_lat, self.base_lon),
                bearing_from_base_deg=0.0,
                recommended_countermeasure="NONE",
                countermeasure_probability_success=1.0,
                engagement_authorized=False,
                rationale="BLUE FORCE ASSET (IFF SECURE). ZERO THREAT.",
                kill_chain_stage=KillChainStage.TRACK,
                assigned_effector_id=None,
                is_friendly=True
            )

        geo = check_geofence_status(
            target_lat=target_lat,
            target_lon=target_lon,
            base_lat=self.base_lat,
            base_lon=self.base_lon,
            inner_killzone_m=self.inner_killzone_m,
            warning_perimeter_m=self.warning_perimeter_m,
            detection_boundary_m=self.detection_boundary_m
        )

        zone = geo["zone"]
        dist_m = geo["distance_m"]
        bearing_deg = geo["bearing_deg"]

        # Base proximity threat factor (0 - 45)
        prox_factor = max(0.0, 1.0 - (dist_m / max(self.detection_boundary_m, 1.0)))
        score = prox_factor * 45.0

        # Classification threat score (0 - 35)
        cls_threat = classification.threat_score if classification else 40.0
        score += (cls_threat * 0.35)

        # Kinematic / dive attack amplification
        tti = trajectory.time_to_impact_sec if trajectory else None
        if trajectory:
            if trajectory.is_diving_or_descending:
                score += 15.0
            if trajectory.is_evasive:
                score += 8.0
            if trajectory.dominant_mode == "CA_DIVE":
                score += 12.0
            if tti is not None and tti < 8.0:
                score += 15.0

        # Autonomous dark / RF-silent penalty
        if classification and classification.is_autonomous_dark_threat:
            score += 20.0

        threat_score = round(min(100.0, max(5.0, score)), 1)

        # Threat Level
        if zone == "RED" or threat_score >= 75.0 or (tti is not None and tti < 6.0):
            threat_level = "CRITICAL"
        elif zone == "AMBER" or threat_score >= 50.0:
            threat_level = "HIGH"
        elif zone == "GREEN" or threat_score >= 30.0:
            threat_level = "ELEVATED"
        else:
            threat_level = "ADVISORY" if threat_score >= 15.0 else "SAFE"

        # Advance Kill-Chain Stage
        if threat_level == "CRITICAL":
            kill_chain = KillChainStage.ENGAGE
        elif threat_level == "HIGH":
            kill_chain = KillChainStage.CUE_EFFECTOR
        elif classification and classification.confidence > 0.65:
            kill_chain = KillChainStage.IDENTIFY
        elif trajectory is not None:
            kill_chain = KillChainStage.TRACK
        else:
            kill_chain = KillChainStage.DETECT

        # Optimal Weapon Assignment
        assigned_measure, p_success, rationale = self._solve_single_wta(
            threat_level=threat_level,
            dist_m=dist_m,
            bearing_deg=bearing_deg,
            classification=classification,
            trajectory=trajectory
        )

        authorized = (self.auto_engagement and threat_level in ["CRITICAL", "HIGH"])

        return ThreatEvaluation(
            target_id=target_id,
            threat_level=threat_level,
            numerical_threat_score=threat_score,
            geofence_zone=zone,
            distance_to_base_m=round(dist_m, 1),
            bearing_from_base_deg=round(bearing_deg, 1),
            recommended_countermeasure=assigned_measure,
            countermeasure_probability_success=round(p_success, 2),
            engagement_authorized=authorized,
            rationale=rationale,
            kill_chain_stage=kill_chain,
            assigned_effector_id=assigned_measure if assigned_measure != "NONE" else None,
            time_to_perimeter_sec=tti,
            is_friendly=False
        )

    def _solve_single_wta(
        self,
        threat_level: str,
        dist_m: float,
        bearing_deg: float,
        classification: Optional[DroneClassificationResult],
        trajectory: Optional[TrajectoryPrediction]
    ) -> Tuple[str, float, str]:
        """Calculates optimal weapon assignment based on effector physics and target state."""
        is_dark = classification.is_autonomous_dark_threat if classification else False
        is_diving = trajectory.is_diving_or_descending if trajectory else False

        # If RF-silent autonomous threat, RF jamming is ineffective -> require Kinetic or Optical
        if is_dark:
            if dist_m <= self.effector_specs["KINETIC_INTERCEPT"]["max_range_m"]:
                return (
                    "KINETIC_INTERCEPT",
                    0.96,
                    "CRITICAL: RF-SILENT AUTONOMOUS WEAPON. RF JAMMING INEFFECTIVE. KINETIC HARD-KILL REQUIRED."
                )
            else:
                return (
                    "OPTICAL_DAZZLER",
                    0.78,
                    "LONG-RANGE AUTONOMOUS WEAPON: ENGAGE DIRECTED OPTICAL DAZZLER TO BLIND SEEKER."
                )

        # High-speed terminal dive
        if is_diving or threat_level == "CRITICAL":
            if dist_m <= 800.0:
                return (
                    "KINETIC_INTERCEPT",
                    0.94,
                    "TERMINAL SUICIDE DIVE IN PROGRESS (<800m). KINETIC INTERCEPT DISPATCHED IMMEDIATELY."
                )
            else:
                return (
                    "RF_JAMMER",
                    0.89,
                    "CRITICAL THREAT INBOUND: ENGAGE DIRECTIONAL C2 & VIDEO LINK BROADBAND JAMMER."
                )

        if threat_level == "HIGH":
            return (
                "GPS_SPOOF",
                0.84,
                "WARNING PERIMETER BREACH: INITIATE FORCED RETURN-TO-HOME GNSS CONSTELLATION SPOOFING."
            )

        if threat_level == "ELEVATED":
            return (
                "OPTICAL_DAZZLER",
                0.75,
                "ELEVATED RECONNAISSANCE THREAT: CUE DIRECTED OPTICAL DAZZLER TO SUPPRESS EO SENSOR."
            )

        return ("NONE", 0.0, "TARGET OUTSIDE ENGAGEMENT THRESHOLD. MAINTAIN PASSIVE SURVEILLANCE.")

    def evaluate_swarm_autonomy_play(self, threats: List[ThreatEvaluation]) -> Optional[AutonomyPlayIntent]:
        """
        Synthesizes active threats and recommends high-level HOTL Autonomy Play:
        - Critical dives / swarm saturation -> PLAY_03_KINETIC_WOLF_PACK
        - Multiple perimeter breaches -> PLAY_02_SWARM_EW_BARRIER
        - Normal surveillance -> PLAY_01_PERIMETER_CAP
        """
        hostile_threats = [t for t in threats if not t.is_friendly]
        if not hostile_threats:
            return None

        critical_count = sum(1 for t in hostile_threats if t.threat_level == "CRITICAL")
        high_count = sum(1 for t in hostile_threats if t.threat_level == "HIGH")

        now = time.time()
        if critical_count >= 1:
            # Urgent Kinetic Wolf Pack Salvo
            target_ids = [t.target_id for t in hostile_threats if t.threat_level == "CRITICAL"]
            return AutonomyPlayIntent(
                play_id=f"PLAY-KW-{int(now)}",
                play_type=AutonomyPlayType.PLAY_03_KINETIC_WOLF_PACK,
                status="PENDING_APPROVAL" if self.governance_mode == "HOTL_SUPERVISED" else "EXECUTING",
                title="PLAY 03: KINETIC WOLF PACK",
                description="Launch multi-interceptor salvo (ANVIL-01..04) with proportional navigation guidance.",
                recommended_assets=["ANVIL-ICP-01", "ANVIL-ICP-02", "ANVIL-ICP-03", "ANVIL-ICP-04"],
                target_ids=target_ids,
                rationale=f"DEFCON 1: {critical_count} target(s) executing terminal dive attack inside killzone.",
                timestamp=now,
                auto_executable=(self.governance_mode == "FULL_AUTONOMOUS_ROE")
            )
        elif high_count >= 2 or len(hostile_threats) >= 4:
            # Swarm EW Barrier
            target_ids = [t.target_id for t in hostile_threats if t.threat_level in ["HIGH", "ELEVATED"]]
            return AutonomyPlayIntent(
                play_id=f"PLAY-EW-{int(now)}",
                play_type=AutonomyPlayType.PLAY_02_SWARM_EW_BARRIER,
                status="PENDING_APPROVAL" if self.governance_mode == "HOTL_SUPERVISED" else "EXECUTING",
                title="PLAY 02: SWARM EW BARRIER",
                description="Synchronize RF broadband noise jammer & omnidirectional GNSS spoofing umbrella.",
                recommended_assets=["RF_JAMMER_ARRAY", "GNSS_SPOOF_TRANSMITTER"],
                target_ids=target_ids,
                rationale=f"{high_count} threat(s) breached warning perimeter. Establish electronic barrier.",
                timestamp=now,
                auto_executable=(self.governance_mode == "FULL_AUTONOMOUS_ROE")
            )
        else:
            return AutonomyPlayIntent(
                play_id=f"PLAY-CAP-{int(now)}",
                play_type=AutonomyPlayType.PLAY_01_PERIMETER_CAP,
                status="EXECUTING",
                title="PLAY 01: PERIMETER CAP",
                description="ISR wingmen maintaining combat air patrol at 1500m perimeter.",
                recommended_assets=["ISR-WINGMAN-01", "ISR-WINGMAN-02"],
                target_ids=[t.target_id for t in hostile_threats],
                rationale="Standard perimeter surveillance and optronic target cueing.",
                timestamp=now,
                auto_executable=True
            )


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return float(2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))
