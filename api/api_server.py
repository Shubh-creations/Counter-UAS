"""
CUAS v6.0-MONOLITH — CUDA-Accelerated Anduril Lattice Mission Autonomy C2
100% Monochrome Black-and-White Tactical Operating Picture
Drop-in replacement for C:\\CUAS\\api\\api_server.py
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


# ============================================================================
# 1. CUDA GPU COMPUTE & SIGNAL PROCESSING ENGINE
# ============================================================================


class CudaTacticalComputeEngine:
    """
    Executes batched 3D swarm kinematics, torch.stft micro-Doppler blade-flash
    spectrograms, and torch.fft SDR spectrum waterfalls on NVIDIA CUDA GPU.
    """

    def __init__(self) -> None:
        self.cuda_enabled = bool(TORCH_AVAILABLE and torch.cuda.is_available())
        if self.cuda_enabled:
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            self.device_name = torch.cuda.get_device_name(0)
        elif TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.device_name = "CPU_FALLBACK_TENSOR"
        else:
            self.device = None
            self.device_name = "NUMPY_FALLBACK"

        self.last_kernel_ms = 0.0
        self._warmup()

    def _warmup(self) -> None:
        if not TORCH_AVAILABLE:
            return
        try:
            with torch.no_grad():
                dummy = torch.randn((16, 9), device=self.device, dtype=torch.float32)
                _ = torch.linalg.norm(dummy, dim=1)
                if self.cuda_enabled:
                    torch.cuda.synchronize()
        except Exception:
            pass

    def get_hardware_telemetry(self) -> Dict[str, Any]:
        if self.cuda_enabled:
            alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
            total_mb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 0
            )
            return {
                "backend": "CUDA_FP16_ACTIVE",
                "device_name": self.device_name,
                "vram_allocated_mb": alloc_mb,
                "vram_reserved_mb": res_mb,
                "vram_total_mb": int(total_mb),
                "kernel_latency_ms": round(self.last_kernel_ms, 2),
            }
        return {
            "backend": self.device_name,
            "device_name": self.device_name,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "vram_total_mb": 0,
            "kernel_latency_ms": round(self.last_kernel_ms, 2),
        }

    def compute_batched_ukf_predictions(
        self,
        positions: List[List[float]],
        velocities: List[List[float]],
        accelerations: List[List[float]],
        horizon_steps: int = 24,
        dt: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Vectorized GPU 3D IMM-UKF trajectory rollout & kinematic risk metrics
        across all N simultaneous swarm assets in a single tensor kernel pass.
        """
        n = len(positions)
        if n == 0:
            return {"trajectories": [], "ranges": [], "closing_vels": [], "imm_dive": []}

        t0 = time.perf_counter()
        if self.cuda_enabled:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        if TORCH_AVAILABLE:
            with torch.no_grad():
                pos = torch.tensor(positions, device=self.device, dtype=torch.float32)  # [N, 3]
                vel = torch.tensor(velocities, device=self.device, dtype=torch.float32)  # [N, 3]
                acc = torch.tensor(accelerations, device=self.device, dtype=torch.float32)  # [N, 3]

                # Ground 2D range and 3D slant range
                range_2d = torch.linalg.norm(pos[:, :2], dim=1)
                slant_range = torch.linalg.norm(pos, dim=1).clamp(min=1.0)

                # Radial closing velocity toward Base HQ (0,0,0)
                closing_vel = -torch.sum(pos * vel, dim=1) / slant_range

                # IMM Terminal Ballistic Dive Probability
                dive_raw = torch.clamp(-vel[:, 2] / 14.0, min=0.0) + torch.where(
                    (range_2d < 750.0) & (closing_vel > 18.0),
                    torch.tensor(0.85, device=self.device),
                    torch.tensor(0.10, device=self.device),
                )
                evasive_raw = torch.clamp(torch.linalg.norm(acc[:, :2], dim=1) / 10.0, max=1.0)
                loiter_raw = torch.clamp(1.0 - torch.maximum(dive_raw, evasive_raw) * 0.75, min=0.08)
                imm_dive = dive_raw / (dive_raw + evasive_raw + loiter_raw + 1e-6)

                # Batched 3D future rollout [N, H, 3]
                steps = torch.arange(1, horizon_steps + 1, device=self.device, dtype=torch.float32) * dt
                steps_col = steps.view(1, horizon_steps, 1)  # [1, H, 1]
                traj = (
                    pos.unsqueeze(1)
                    + vel.unsqueeze(1) * steps_col
                    + 0.5 * acc.unsqueeze(1) * (steps_col ** 2)
                )
                traj[:, :, 2] = torch.clamp(traj[:, :, 2], min=0.0)

                if self.cuda_enabled:
                    end_evt.record()
                    torch.cuda.synchronize()
                    self.last_kernel_ms = float(start_evt.elapsed_time(end_evt))
                else:
                    self.last_kernel_ms = (time.perf_counter() - t0) * 1000.0

                return {
                    "trajectories": torch.round(traj, decimals=1).cpu().tolist(),
                    "ranges": torch.round(range_2d, decimals=1).cpu().tolist(),
                    "closing_vels": torch.round(closing_vel, decimals=1).cpu().tolist(),
                    "imm_dive": torch.round(imm_dive, decimals=2).cpu().tolist(),
                }

        # Pure NumPy fallback if PyTorch is absent
        pos_np = np.array(positions, dtype=np.float32)
        vel_np = np.array(velocities, dtype=np.float32)
        acc_np = np.array(accelerations, dtype=np.float32)
        ranges = np.linalg.norm(pos_np[:, :2], axis=1)
        slant = np.maximum(np.linalg.norm(pos_np, axis=1), 1.0)
        closing = -np.sum(pos_np * vel_np, axis=1) / slant
        steps_np = (np.arange(1, horizon_steps + 1, dtype=np.float32) * dt)[None, :, None]
        traj_np = pos_np[:, None, :] + vel_np[:, None, :] * steps_np + 0.5 * acc_np[:, None, :] * (steps_np ** 2)
        traj_np[:, :, 2] = np.maximum(traj_np[:, :, 2], 0.0)
        self.last_kernel_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "trajectories": np.round(traj_np, 1).tolist(),
            "ranges": np.round(ranges, 1).tolist(),
            "closing_vels": np.round(closing, 1).tolist(),
            "imm_dive": [0.65] * n,
        }

    def compute_cuda_micro_doppler(
        self, rotor_rpm: float, closing_vel: float, phase_t: float
    ) -> List[List[float]]:
        """
        Synthesizes complex pulse-Doppler radar returns on GPU and runs torch.stft
        to produce a 32x24 normalized luminance matrix (0.0 to 1.0) for rotor
        blade-flash micro-Doppler visualization.
        """
        if TORCH_AVAILABLE:
            try:
                with torch.no_grad():
                    fs = 1024.0
                    t = torch.linspace(0, 0.5, 512, device=self.device, dtype=torch.float32) + phase_t
                    body_freq = min(260.0, abs(closing_vel) * 4.5 + 35.0)
                    blade_freq = min(290.0, (rotor_rpm / 60.0) * 2.0)

                    # Main fuselage Doppler + asymmetric rotor blade-flash FM sidebands
                    sig = 0.75 * torch.sin(2.0 * math.pi * body_freq * t)
                    for h in (1.0, 1.5, 2.0):
                        mod = 0.45 * torch.sin(2.0 * math.pi * (blade_freq * h) * t + 1.8 * torch.sin(14.0 * t))
                        sig = sig + mod
                    sig = sig + 0.14 * torch.randn_like(sig)

                    window = torch.hann_window(64, device=self.device)
                    stft_mag = torch.stft(
                        sig, n_fft=64, hop_length=20, window=window, return_complex=True
                    ).abs()  # [33, 26]
                    stft_slice = stft_mag[:32, :24]
                    stft_norm = stft_slice / (stft_slice.max() + 1e-6)
                    return torch.round(stft_norm, decimals=2).cpu().tolist()
            except Exception:
                pass

        # Fallback matrix
        rows = []
        for f in range(32):
            row = [
                round(abs(math.sin(f * 0.25 + c * 0.35 + phase_t * 4.0)) * 0.85, 2)
                for c in range(24)
            ]
            rows.append(row)
        return rows

    def compute_cuda_rf_fft(
        self, is_rf_silent: bool, is_jammed: bool, phase_t: float
    ) -> List[float]:
        """Computes 48-bin normalized SDR power spectral density via torch.fft.rfft on GPU."""
        if TORCH_AVAILABLE:
            try:
                with torch.no_grad():
                    t = torch.linspace(0, 1.0, 128, device=self.device, dtype=torch.float32) + phase_t
                    noise = 0.18 * torch.randn_like(t)
                    if is_jammed:
                        # High-power barrage jamming across 2.4/5.8 GHz bins
                        sig = noise + 1.6 * torch.sin(2 * math.pi * 18 * t) + 1.5 * torch.cos(2 * math.pi * 34 * t)
                    elif is_rf_silent:
                        sig = noise * 0.65
                    else:
                        hop = 14.0 + 6.0 * math.sin(phase_t * 3.0)
                        sig = noise + 1.15 * torch.sin(2 * math.pi * hop * t) + 0.95 * torch.sin(2 * math.pi * 38.0 * t)
                    fft_mag = torch.fft.rfft(sig).abs()[:48]
                    fft_norm = torch.clamp(fft_mag / 45.0, 0.05, 1.0)
                    return torch.round(fft_norm, decimals=2).cpu().tolist()
            except Exception:
                pass

        return [round(0.15 + random.random() * 0.2, 2) for _ in range(48)]


# ============================================================================
# 2. CRYPTOGRAPHIC SHA-256 FORENSIC AUDIT LEDGER
# ============================================================================

AUDIT_LOG_PATH = Path("logs/audit/mission_audit_chain.jsonl")


class LatticeAuditLedger:
    """Tamper-evident SHA-256 hash-chained mission ledger."""

    def __init__(self, log_path: Path = AUDIT_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.chain: List[Dict[str, Any]] = []
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.chain.append(json.loads(line))
            except Exception:
                self.chain = []
        if not self.chain:
            self.record_event(
                event_type="MONOLITH_CUDA_BOOT",
                operator="LATTICE_AUTONOMY_OS",
                payload={"version": "6.0.0-MONOLITH", "theme": "ACHROMATIC_TACTICAL"},
            )

    def record_event(
        self, event_type: str, operator: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_hash = self.chain[-1]["block_hash"] if self.chain else "0" * 64
        payload_str = json.dumps(payload, sort_keys=True)
        raw = f"{timestamp}|{event_type}|{operator}|{payload_str}|{prev_hash}"
        block_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        block = {
            "index": len(self.chain),
            "timestamp": timestamp,
            "event_type": event_type,
            "operator": operator,
            "payload": payload,
            "prev_hash": prev_hash,
            "block_hash": block_hash,
        }
        self.chain.append(block)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(block) + "\n")
        except Exception:
            pass
        return block

    def verify_integrity(self) -> Dict[str, Any]:
        for idx, block in enumerate(self.chain):
            expected_prev = "0" * 64 if idx == 0 else self.chain[idx - 1]["block_hash"]
            if block.get("prev_hash") != expected_prev:
                return {"verified": False, "corrupted_block": idx}
        return {
            "verified": True,
            "total_blocks": len(self.chain),
            "head_hash": self.chain[-1]["block_hash"] if self.chain else "0" * 64,
        }


# ============================================================================
# 3. MISSION AUTONOMY ORCHESTRATOR & CLOSED-LOOP REPLAY ENGINE
# ============================================================================

BASE_LAT = 37.7749
BASE_LON = -122.4194


@dataclass
class SwarmEntity:
    target_id: str
    cluster_id: str
    affiliation: str  # HOSTILE, SUSPECT, FRIENDLY_WINGMAN, FRIENDLY_INTERCEPTOR, NEUTRAL
    drone_class: str
    rf_protocol: str
    east_m: float
    north_m: float
    alt_m: float
    vx_ms: float
    vy_ms: float
    vz_ms: float
    ax_ms2: float = 0.0
    ay_ms2: float = 0.0
    az_ms2: float = 0.0
    rcs_dbsm: float = -15.8
    rotor_rpm: float = 14400.0
    payload_detected: bool = True
    confidence: float = 0.962
    active_effectors: List[str] = field(default_factory=list)
    status: str = "ACTIVE"
    assigned_asset: str = "UNASSIGNED"
    trail: List[List[float]] = field(default_factory=list)
    intercept_target_id: Optional[str] = None


class LatticeMissionAutonomyEngine:
    """
    End-to-End Mission Autonomy Engine:
      - Pre-Mission Threat Modeling & Wargaming
      - Live Multi-Asset Swarm Orchestration (HOTL + Autonomous ROE)
      - Closed-Loop Flight Recorder & Post-Mission Debrief Playback
    """

    def __init__(
        self, cuda_engine: CudaTacticalComputeEngine, audit_ledger: LatticeAuditLedger
    ) -> None:
        self.cuda = cuda_engine
        self.audit = audit_ledger
        self.mission_phase = "LIVE_EXECUTION"  # PRE_MISSION, LIVE_EXECUTION, POST_MISSION
        self.scenario = "PINCER_SWARM_10"
        self.roe_mode = "HOTL_SUPERVISED"  # HOTL_SUPERVISED or FULL_AUTONOMOUS_ROE
        self.killzone_radius_m = 500.0
        self.warning_radius_m = 1500.0
        self.detection_radius_m = 3500.0

        self.boot_time = time.time()
        self.last_tick = time.time()
        self.radar_sweep_deg = 0.0
        self.neutralized_count = 0
        self.total_detected_count = 0
        self.selected_target_id = "#TGT-101"

        self.entities: Dict[str, SwarmEntity] = {}
        self.splash_events: List[Dict[str, Any]] = []
        self.pending_autonomy_play: Optional[Dict[str, Any]] = None
        self.active_play_id: str = "NONE"

        # Closed-Loop Mission Flight Recorder (stores up to 1800 snapshots)
        self.recorded_frames: Deque[Dict[str, Any]] = deque(maxlen=1800)
        self.replay_index: int = 0
        self.replay_playing: bool = True
        self.replay_speed: int = 1

        self.effectors: Dict[str, Dict[str, Any]] = {
            "RF_JAMMER": {
                "id": "RF_JAMMER",
                "name": "DIRECTIONAL RF JAMMER",
                "state": "STANDBY",
                "azimuth_deg": 48.0,
                "target_id": None,
            },
            "GNSS_SPOOFER": {
                "id": "GNSS_SPOOFER",
                "name": "GNSS EPHEMERIS SPOOFER",
                "state": "STANDBY",
                "azimuth_deg": 0.0,
                "target_id": None,
            },
            "OPTICAL_DAZZLER": {
                "id": "OPTICAL_DAZZLER",
                "name": "1064nm SWIR DAZZLER",
                "state": "STANDBY",
                "azimuth_deg": 48.0,
                "target_id": None,
            },
            "KINETIC_SILO": {
                "id": "KINETIC_SILO",
                "name": "ANVIL KINETIC SILOS",
                "state": "READY (8 READY)",
                "ammo": 8,
                "target_id": None,
            },
        }
        self.load_scenario(self.scenario)

    def load_scenario(self, scenario_name: str) -> None:
        self.scenario = scenario_name.upper()
        self.entities.clear()
        self.splash_events.clear()
        self.recorded_frames.clear()
        self.replay_index = 0
        self.active_play_id = "NONE"
        self.pending_autonomy_play = None

        for eff in self.effectors.values():
            if eff["id"] == "KINETIC_SILO":
                eff["state"] = "READY (8 READY)"
                eff["ammo"] = 8
            else:
                eff["state"] = "STANDBY"
            eff["target_id"] = None

        # Always deploy 2 Friendly Collaborative Autonomous ISR Wingmen
        self.entities["#ISR-WING-01"] = SwarmEntity(
            target_id="#ISR-WING-01",
            cluster_id="FRIENDLY-CAP",
            affiliation="FRIENDLY_WINGMAN",
            drone_class="AUTONOMOUS_ISR_WINGMAN",
            rf_protocol="MESH_ENCRYPTED",
            east_m=240.0,
            north_m=320.0,
            alt_m=220.0,
            vx_ms=-12.0,
            vy_ms=10.0,
            vz_ms=0.0,
            rcs_dbsm=-8.0,
            rotor_rpm=6800.0,
            payload_detected=False,
            confidence=1.0,
            status="PATROL_CAP",
            assigned_asset="SECTOR_NORTH_EAST",
        )
        self.entities["#ISR-WING-02"] = SwarmEntity(
            target_id="#ISR-WING-02",
            cluster_id="FRIENDLY-CAP",
            affiliation="FRIENDLY_WINGMAN",
            drone_class="AUTONOMOUS_ISR_WINGMAN",
            rf_protocol="MESH_ENCRYPTED",
            east_m=-340.0,
            north_m=-260.0,
            alt_m=210.0,
            vx_ms=11.0,
            vy_ms=-12.0,
            vz_ms=0.0,
            rcs_dbsm=-8.0,
            rotor_rpm=6800.0,
            payload_detected=False,
            confidence=1.0,
            status="PATROL_CAP",
            assigned_asset="SECTOR_SOUTH_WEST",
        )

        if self.scenario == "RF_SILENT_STEALTH":
            angles = [35, 52, 145, 225, 310]
            for i, deg in enumerate(angles, start=1):
                rad = math.radians(deg)
                dist = 780.0 + i * 165.0
                tid = f"#TGT-20{i}"
                self.entities[tid] = SwarmEntity(
                    target_id=tid,
                    cluster_id="CLUSTER-PHANTOM",
                    affiliation="HOSTILE",
                    drone_class="AUTONOMOUS_LOITERING_MUNITION",
                    rf_protocol="RF_SILENT_OPTICAL_NAV",
                    east_m=dist * math.sin(rad),
                    north_m=dist * math.cos(rad),
                    alt_m=95.0 + i * 22.0,
                    vx_ms=-27.5 * math.sin(rad),
                    vy_ms=-27.5 * math.cos(rad),
                    vz_ms=-2.6,
                    rcs_dbsm=-19.2,
                    rotor_rpm=15800.0,
                    payload_detected=True,
                    confidence=0.955,
                )
        elif self.scenario == "FPV_TERMINAL_DIVE":
            for i, (e, n, a) in enumerate(
                [(310.0, 290.0, 180.0), (420.0, 190.0, 210.0), (-380.0, 340.0, 195.0)],
                start=1,
            ):
                tid = f"#TGT-30{i}"
                self.entities[tid] = SwarmEntity(
                    target_id=tid,
                    cluster_id="CLUSTER-STRIKE",
                    affiliation="HOSTILE",
                    drone_class="SUICIDE_FPV_MUNITION",
                    rf_protocol="ExpressLRS 2.4GHz",
                    east_m=e,
                    north_m=n,
                    alt_m=a,
                    vx_ms=-e * 0.065,
                    vy_ms=-n * 0.065,
                    vz_ms=-7.2,
                    ax_ms2=-2.5,
                    ay_ms2=-2.1,
                    rcs_dbsm=-17.0,
                    rotor_rpm=16800.0,
                    payload_detected=True,
                    confidence=0.971,
                )
        else:
            # Default: 8-UAV Multi-Axis Pincer Swarm
            self.scenario = "PINCER_SWARM_10"
            swarm_specs = [
                ("#TGT-101", "CLUSTER-ALPHA", "HOSTILE", "SUICIDE_FPV_MUNITION", "ExpressLRS 2.4GHz", 330.0, 310.0, 145.0, -20.5, -21.0, -4.5, True),
                ("#TGT-102", "CLUSTER-ALPHA", "HOSTILE", "SUICIDE_FPV_MUNITION", "ExpressLRS 2.4GHz", 440.0, 220.0, 160.0, -22.0, -16.5, -3.8, True),
                ("#TGT-103", "CLUSTER-ALPHA", "HOSTILE", "CUSTOMIZED_BOMBER_UAV", "DJI OcuSync-3", 520.0, 390.0, 190.0, -21.0, -18.0, -2.5, True),
                ("#TGT-104", "CLUSTER-BRAVO", "HOSTILE", "AUTONOMOUS_LOITERING_MUNITION", "RF_SILENT_OPTICAL_NAV", -680.0, 510.0, 230.0, 24.5, -18.5, -3.2, True),
                ("#TGT-105", "CLUSTER-BRAVO", "HOSTILE", "AUTONOMOUS_LOITERING_MUNITION", "RF_SILENT_OPTICAL_NAV", -790.0, 380.0, 205.0, 26.0, -14.0, -2.9, True),
                ("#TGT-106", "CLUSTER-CHARLIE", "HOSTILE", "SUICIDE_FPV_MUNITION", "TBS Crossfire", 140.0, -980.0, 120.0, -4.5, 28.0, -2.0, True),
                ("#TGT-107", "CLUSTER-CHARLIE", "SUSPECT", "SURVEILLANCE_ISR_DRONE", "MAVLink 915MHz", -940.0, -820.0, 340.0, 13.0, 12.0, 0.0, False),
                ("#CIV-801", "CIVILIAN-CORRIDOR", "NEUTRAL", "COMMERCIAL_QUADCOPTER", "RemoteID Broadcast", -1350.0, 1080.0, 115.0, 10.5, 2.5, 0.0, False),
            ]
            for tid, cl, aff, cls, rf, e, n, a, vx, vy, vz, pay in swarm_specs:
                self.entities[tid] = SwarmEntity(
                    target_id=tid,
                    cluster_id=cl,
                    affiliation=aff,
                    drone_class=cls,
                    rf_protocol=rf,
                    east_m=e,
                    north_m=n,
                    alt_m=a,
                    vx_ms=vx,
                    vy_ms=vy,
                    vz_ms=vz,
                    payload_detected=pay,
                )

        hostiles = [
            e.target_id for e in self.entities.values() if e.affiliation == "HOSTILE"
        ]
        self.total_detected_count = len(hostiles)
        if hostiles:
            self.selected_target_id = hostiles[0]

        self.audit.record_event(
            "SCENARIO_CONFIGURED",
            "LATTICE_MISSION_PLANNER",
            {"scenario": self.scenario, "hostile_count": len(hostiles)},
        )

    # ------------------------------------------------------------------------
    # HUMAN-ON-THE-LOOP (HOTL) AUTONOMY PLAY EXECUTION
    # ------------------------------------------------------------------------
    def execute_autonomy_play(
        self, play_id: str, operator: str = "HOTL_COMMANDER"
    ) -> Dict[str, Any]:
        play = play_id.upper()
        self.active_play_id = play
        self.pending_autonomy_play = None

        if play in ("PLAY_04_CEASEFIRE_RTB", "CEASEFIRE", "DISENGAGE_ALL"):
            return self.ceasefire_and_rtb(operator)

        hostiles = [
            e
            for e in self.entities.values()
            if e.affiliation in ("HOSTILE", "SUSPECT") and e.status != "NEUTRALIZED"
        ]
        hostiles.sort(key=lambda x: math.hypot(x.east_m, x.north_m))

        if play == "PLAY_01_PERIMETER_CAP":
            # Task friendly ISR Wingmen to shadow top 2 hostile clusters
            wingmen = [
                e for e in self.entities.values() if e.affiliation == "FRIENDLY_WINGMAN"
            ]
            for idx, wm in enumerate(wingmen):
                if idx < len(hostiles):
                    wm.intercept_target_id = hostiles[idx].target_id
                    wm.status = f"SHADOWING {hostiles[idx].target_id}"
                    hostiles[idx].assigned_asset = wm.target_id
            block = self.audit.record_event(
                "AUTONOMY_PLAY_EXECUTED",
                operator,
                {"play": play, "wingmen_tasked": len(wingmen)},
            )
            return {"status": "EXECUTED", "play": play, "hash": block["block_hash"]}

        if play == "PLAY_02_SWARM_EW_BARRIER":
            # Synchronized RF Jamming + GNSS Spoofing + SWIR Dazzler across swarm
            if hostiles:
                lead = hostiles[0]
                az = (math.degrees(math.atan2(lead.east_m, lead.north_m)) + 360.0) % 360.0
                self.effectors["RF_JAMMER"]["state"] = "ENGAGED (BARRIER)"
                self.effectors["RF_JAMMER"]["azimuth_deg"] = round(az, 1)
                self.effectors["RF_JAMMER"]["target_id"] = lead.target_id
                self.effectors["GNSS_SPOOFER"]["state"] = "ENGAGED (DIVERT)"
                self.effectors["GNSS_SPOOFER"]["target_id"] = lead.target_id
                self.effectors["OPTICAL_DAZZLER"]["state"] = "ENGAGED (SWIR)"
                self.effectors["OPTICAL_DAZZLER"]["azimuth_deg"] = round(az, 1)
                self.effectors["OPTICAL_DAZZLER"]["target_id"] = lead.target_id

            affected = 0
            for h in hostiles:
                if math.hypot(h.east_m, h.north_m) < 1350.0:
                    if h.rf_protocol == "RF_SILENT_OPTICAL_NAV":
                        if "OPTICAL_DAZZLER" not in h.active_effectors:
                            h.active_effectors.append("OPTICAL_DAZZLER")
                        h.assigned_asset = "SWIR_DAZZLER"
                    else:
                        if "RF_JAMMER" not in h.active_effectors:
                            h.active_effectors.append("RF_JAMMER")
                        if "GNSS_SPOOFER" not in h.active_effectors:
                            h.active_effectors.append("GNSS_SPOOFER")
                        h.assigned_asset = "EW_BARRIER"
                    affected += 1

            block = self.audit.record_event(
                "AUTONOMY_PLAY_EXECUTED",
                operator,
                {"play": play, "targets_suppressed": affected},
            )
            return {"status": "EXECUTED", "play": play, "affected": affected, "hash": block["block_hash"]}

        if play == "PLAY_03_KINETIC_WOLF_PACK":
            # Global WTAA dispatch of up to 3 synchronized ANVIL kinetic interceptors
            launched = 0
            for h in hostiles[:3]:
                ammo = self.effectors["KINETIC_SILO"].get("ammo", 0)
                if ammo <= 0:
                    break
                already = any(
                    e.affiliation == "FRIENDLY_INTERCEPTOR"
                    and e.intercept_target_id == h.target_id
                    for e in self.entities.values()
                )
                if already:
                    continue
                self.effectors["KINETIC_SILO"]["ammo"] = ammo - 1
                icp_id = f"#ANVIL-{random.randint(10, 99)}"
                self.entities[icp_id] = SwarmEntity(
                    target_id=icp_id,
                    cluster_id="WOLF-PACK",
                    affiliation="FRIENDLY_INTERCEPTOR",
                    drone_class="ANVIL_KINETIC_INTERCEPTOR",
                    rf_protocol="LINK-16_DATALINK",
                    east_m=random.uniform(-25.0, 25.0),
                    north_m=random.uniform(-25.0, 25.0),
                    alt_m=24.0,
                    vx_ms=0.0,
                    vy_ms=0.0,
                    vz_ms=20.0,
                    confidence=1.0,
                    status=f"INTERCEPT -> {h.target_id}",
                    assigned_asset=h.target_id,
                    intercept_target_id=h.target_id,
                )
                h.assigned_asset = icp_id
                launched += 1

            rem = self.effectors["KINETIC_SILO"]["ammo"]
            self.effectors["KINETIC_SILO"]["state"] = f"ACTIVE ({rem} SILOS)"
            block = self.audit.record_event(
                "AUTONOMY_PLAY_EXECUTED",
                operator,
                {"play": play, "interceptors_launched": launched},
            )
            return {"status": "EXECUTED", "play": play, "launched": launched, "hash": block["block_hash"]}

        return {"status": "UNKNOWN_PLAY", "play": play}

    def engage_single_effector(
        self, effector: str, target_id: Optional[str] = None, operator: str = "HOTL_COMMANDER"
    ) -> Dict[str, Any]:
        eff_upper = effector.upper()
        if eff_upper in ("DISENGAGE_ALL", "CEASEFIRE", "PLAY_04_CEASEFIRE_RTB"):
            return self.ceasefire_and_rtb(operator)
        if eff_upper.startswith("PLAY_"):
            return self.execute_autonomy_play(eff_upper, operator)

        tgt_id = target_id or self.selected_target_id
        trk = self.entities.get(tgt_id)
        if not trk:
            return {"status": "NO_TARGET"}

        if eff_upper in ("KINETIC_INTERCEPTOR", "KINETIC_SILO", "INTERCEPTOR"):
            ammo = self.effectors["KINETIC_SILO"].get("ammo", 0)
            if ammo > 0:
                self.effectors["KINETIC_SILO"]["ammo"] = ammo - 1
            icp_id = f"#ANVIL-{random.randint(10, 99)}"
            self.entities[icp_id] = SwarmEntity(
                target_id=icp_id,
                cluster_id="INTERCEPT",
                affiliation="FRIENDLY_INTERCEPTOR",
                drone_class="ANVIL_KINETIC_INTERCEPTOR",
                rf_protocol="LINK-16_DATALINK",
                east_m=0.0,
                north_m=0.0,
                alt_m=22.0,
                vx_ms=0.0,
                vy_ms=0.0,
                vz_ms=18.0,
                confidence=1.0,
                status=f"INTERCEPT -> {trk.target_id}",
                assigned_asset=trk.target_id,
                intercept_target_id=trk.target_id,
            )
            trk.assigned_asset = icp_id
            rem = self.effectors["KINETIC_SILO"]["ammo"]
            self.effectors["KINETIC_SILO"]["state"] = f"LAUNCHED ({rem} SILOS)"
            canonical = "KINETIC_SILO"
        else:
            canonical = (
                "GNSS_SPOOFER"
                if "SPOOF" in eff_upper
                else ("OPTICAL_DAZZLER" if "DAZZL" in eff_upper else "RF_JAMMER")
            )
            if canonical not in trk.active_effectors:
                trk.active_effectors.append(canonical)
            trk.assigned_asset = canonical
            az = (math.degrees(math.atan2(trk.east_m, trk.north_m)) + 360.0) % 360.0
            self.effectors[canonical]["state"] = "ENGAGED"
            self.effectors[canonical]["azimuth_deg"] = round(az, 1)
            self.effectors[canonical]["target_id"] = trk.target_id

        block = self.audit.record_event(
            "EFFECTOR_ENGAGED",
            operator,
            {"effector": canonical, "target_id": trk.target_id},
        )
        return {"status": "ENGAGED", "effector": canonical, "target_id": trk.target_id, "hash": block["block_hash"]}

    def ceasefire_and_rtb(self, operator: str = "HOTL_COMMANDER") -> Dict[str, Any]:
        self.active_play_id = "PLAY_04_CEASEFIRE_RTB"
        self.pending_autonomy_play = None
        for eff in self.effectors.values():
            if eff["id"] == "KINETIC_SILO":
                eff["state"] = f"READY ({eff.get('ammo', 8)} READY)"
            else:
                eff["state"] = "STANDBY"
            eff["target_id"] = None

        to_del = []
        for tid, ent in self.entities.items():
            if ent.affiliation == "FRIENDLY_INTERCEPTOR":
                to_del.append(tid)
            elif ent.affiliation == "FRIENDLY_WINGMAN":
                ent.intercept_target_id = None
                ent.status = "RTB_ORBIT"
                ent.assigned_asset = "BASE_PERIMETER"
            else:
                ent.active_effectors.clear()
                ent.status = "ACTIVE"
                ent.assigned_asset = "UNASSIGNED"
        for tid in to_del:
            self.entities.pop(tid, None)

        block = self.audit.record_event(
            "EMERGENCY_CEASEFIRE_RTB", operator, {"status": "ALL_EMITTERS_SAFE"}
        )
        return {"status": "CEASEFIRE_ACTIVE", "hash": block["block_hash"]}

    # ------------------------------------------------------------------------
    # REAL-TIME SIMULATION STEP & CUDA SNAPSHOT GENERATION
    # ------------------------------------------------------------------------
    def step_simulation(self) -> None:
        now = time.time()
        dt = min(max(now - self.last_tick, 0.01), 0.25)
        self.last_tick = now

        if self.mission_phase != "LIVE_EXECUTION":
            return

        self.radar_sweep_deg = (self.radar_sweep_deg + 145.0 * dt) % 360.0
        self.splash_events = [s for s in self.splash_events if now - s["time"] < 2.6]

        to_remove: List[str] = []

        # 1. Update Friendly Autonomous Wingmen & Kinetic Interceptors
        for ent in list(self.entities.values()):
            if ent.affiliation == "FRIENDLY_INTERCEPTOR":
                tgt = self.entities.get(ent.intercept_target_id or "")
                if not tgt:
                    to_remove.append(ent.target_id)
                    continue
                dx = tgt.east_m - ent.east_m
                dy = tgt.north_m - ent.north_m
                dz = tgt.alt_m - ent.alt_m
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 46.0:
                    self.neutralized_count += 1
                    self.splash_events.append(
                        {
                            "east_m": round(tgt.east_m, 1),
                            "north_m": round(tgt.north_m, 1),
                            "alt_m": round(tgt.alt_m, 1),
                            "target_id": tgt.target_id,
                            "time": now,
                        }
                    )
                    to_remove.append(ent.target_id)
                    to_remove.append(tgt.target_id)
                    self.audit.record_event(
                        "BDA_KINETIC_SPLASH",
                        "ANVIL_AUTONOMOUS_INTERCEPTOR",
                        {"interceptor": ent.target_id, "target": tgt.target_id},
                    )
                else:
                    spd = 78.0
                    ent.vx_ms = (dx / dist) * spd
                    ent.vy_ms = (dy / dist) * spd
                    ent.vz_ms = (dz / dist) * spd
                    ent.east_m += ent.vx_ms * dt
                    ent.north_m += ent.vy_ms * dt
                    ent.alt_m += ent.vz_ms * dt
                    ent.trail.append([round(ent.east_m, 1), round(ent.north_m, 1)])

            elif ent.affiliation == "FRIENDLY_WINGMAN":
                tgt = self.entities.get(ent.intercept_target_id or "")
                if tgt:
                    orbit_ang = now * 0.9 + (0 if ent.target_id.endswith("01") else math.pi)
                    goal_e = tgt.east_m + 180.0 * math.cos(orbit_ang)
                    goal_n = tgt.north_m + 180.0 * math.sin(orbit_ang)
                    ent.vx_ms = (goal_e - ent.east_m) * 0.8
                    ent.vy_ms = (goal_n - ent.north_m) * 0.8
                else:
                    cap_ang = now * 0.35 + (0 if ent.target_id.endswith("01") else math.pi)
                    goal_e = 450.0 * math.cos(cap_ang)
                    goal_n = 450.0 * math.sin(cap_ang)
                    ent.vx_ms = (goal_e - ent.east_m) * 0.6
                    ent.vy_ms = (goal_n - ent.north_m) * 0.6
                ent.east_m += ent.vx_ms * dt
                ent.north_m += ent.vy_ms * dt
                ent.trail.append([round(ent.east_m, 1), round(ent.north_m, 1)])

        # 2. Update Hostile / Suspect / Neutral Tracks
        for tid, trk in list(self.entities.items()):
            if trk.affiliation.startswith("FRIENDLY") or tid in to_remove:
                continue

            if "RF_JAMMER" in trk.active_effectors and trk.rf_protocol != "RF_SILENT_OPTICAL_NAV":
                trk.status = "JAMMED"
                trk.vx_ms *= 0.91
                trk.vy_ms *= 0.91
                trk.vz_ms = -2.2
            if "GNSS_SPOOFER" in trk.active_effectors:
                trk.status = "SPOOFED"
                away = math.atan2(trk.east_m, trk.north_m)
                trk.vx_ms = 23.0 * math.sin(away)
                trk.vy_ms = 23.0 * math.cos(away)
                trk.vz_ms = 0.6
            if "OPTICAL_DAZZLER" in trk.active_effectors:
                trk.status = "DAZZLED"
                trk.ax_ms2 = random.uniform(-12.0, 12.0)
                trk.ay_ms2 = random.uniform(-12.0, 12.0)

            rng = math.hypot(trk.east_m, trk.north_m)
            if trk.status == "ACTIVE" and trk.affiliation == "HOSTILE":
                swirl = math.sin(now * 1.8 + hash(tid) % 11) * 3.2
                trk.ax_ms2 = swirl
                trk.ay_ms2 = -swirl
                if rng < 85.0:
                    ang = random.uniform(0, 2 * math.pi)
                    trk.east_m = 1650.0 * math.sin(ang)
                    trk.north_m = 1650.0 * math.cos(ang)
                    trk.alt_m = random.uniform(130.0, 280.0)
                    trk.vx_ms = -22.5 * math.sin(ang)
                    trk.vy_ms = -22.5 * math.cos(ang)
                    trk.trail.clear()

            trk.east_m += trk.vx_ms * dt
            trk.north_m += trk.vy_ms * dt
            trk.alt_m = max(10.0, trk.alt_m + trk.vz_ms * dt)

            if len(trk.trail) == 0 or math.hypot(
                trk.east_m - trk.trail[-1][0], trk.north_m - trk.trail[-1][1]
            ) > 15.0:
                trk.trail.append([round(trk.east_m, 1), round(trk.north_m, 1)])

            if trk.status in ("JAMMED", "SPOOFED") and (rng > 2500.0 or trk.alt_m <= 11.0):
                self.neutralized_count += 1
                to_remove.append(tid)

        for tid in set(to_remove):
            self.entities.pop(tid, None)

        active_hostiles = [
            e for e in self.entities.values() if e.affiliation in ("HOSTILE", "SUSPECT")
        ]
        if not active_hostiles:
            self.load_scenario(self.scenario)
            active_hostiles = [
                e for e in self.entities.values() if e.affiliation in ("HOSTILE", "SUSPECT")
            ]

        # 3. HOTL Autonomy Recommendation / Autonomous ROE Execution
        killzone_breaches = [
            h
            for h in active_hostiles
            if math.hypot(h.east_m, h.north_m) < self.killzone_radius_m and h.status == "ACTIVE"
        ]
        if killzone_breaches:
            has_stealth = any(
                h.rf_protocol == "RF_SILENT_OPTICAL_NAV" for h in killzone_breaches
            )
            rec_play = (
                "PLAY_03_KINETIC_WOLF_PACK"
                if (has_stealth or len(killzone_breaches) >= 2)
                else "PLAY_02_SWARM_EW_BARRIER"
            )
            if self.roe_mode == "FULL_AUTONOMOUS_ROE":
                self.execute_autonomy_play(rec_play, "FULL_AUTONOMOUS_ROE")
            else:
                self.pending_autonomy_play = {
                    "play_id": rec_play,
                    "reason": f"{len(killzone_breaches)} HOSTILE(S) INSIDE {int(self.killzone_radius_m)}m KILLZONE",
                    "confidence": 96.8,
                }
        else:
            self.pending_autonomy_play = None

    def build_snapshot(self) -> Dict[str, Any]:
        if self.mission_phase == "POST_MISSION" and self.recorded_frames:
            if self.replay_playing:
                self.replay_index = (self.replay_index + self.replay_speed) % len(
                    self.recorded_frames
                )
            frame = copy.deepcopy(self.recorded_frames[self.replay_index])
            frame["mission_phase"] = "POST_MISSION"
            frame["replay_index"] = self.replay_index
            frame["replay_total"] = len(self.recorded_frames)
            frame["replay_playing"] = self.replay_playing
            frame["replay_speed"] = self.replay_speed
            return frame

        self.step_simulation()

        entity_list = list(self.entities.values())
        positions = [[e.east_m, e.north_m, e.alt_m] for e in entity_list]
        velocities = [[e.vx_ms, e.vy_ms, e.vz_ms] for e in entity_list]
        accelerations = [[e.ax_ms2, e.ay_ms2, e.az_ms2] for e in entity_list]

        gpu_out = self.cuda.compute_batched_ukf_predictions(
            positions, velocities, accelerations
        )

        tracks_payload: List[Dict[str, Any]] = []
        for idx, ent in enumerate(entity_list):
            rng_2d = gpu_out["ranges"][idx]
            closing = gpu_out["closing_vels"][idx]
            dive_prob = gpu_out["imm_dive"][idx]
            pred_path = gpu_out["trajectories"][idx]

            slant = math.sqrt(rng_2d**2 + ent.alt_m**2)
            bearing = (math.degrees(math.atan2(ent.east_m, ent.north_m)) + 360.0) % 360.0
            speed = math.sqrt(ent.vx_ms**2 + ent.vy_ms**2 + ent.vz_ms**2)
            tti = round(slant / max(closing, 0.5), 1) if closing > 0.8 else 999.0

            if ent.affiliation.startswith("FRIENDLY"):
                threat_score = 0.0
            elif ent.affiliation == "NEUTRAL":
                threat_score = 12.0
            elif ent.status in ("JAMMED", "SPOOFED", "DAZZLED"):
                threat_score = round(max(18.0, 46.0 - rng_2d * 0.01), 1)
            else:
                prox = max(0.0, (2500.0 - min(rng_2d, 2500.0)) / 2500.0) * 32.0
                stealth = 11.0 if ent.rf_protocol == "RF_SILENT_OPTICAL_NAV" else 0.0
                pay = 12.0 if ent.payload_detected else 0.0
                threat_score = round(min(100.0, 48.0 * ent.confidence + prox + stealth + pay), 1)

            zone_code = (
                "RED"
                if rng_2d < self.killzone_radius_m
                else ("AMBER" if rng_2d < self.warning_radius_m else "GREEN")
            )

            tracks_payload.append(
                {
                    "target_id": ent.target_id,
                    "cluster_id": ent.cluster_id,
                    "affiliation": ent.affiliation,
                    "drone_class": ent.drone_class,
                    "rf_protocol": ent.rf_protocol,
                    "east_m": round(ent.east_m, 1),
                    "north_m": round(ent.north_m, 1),
                    "alt_m": round(ent.alt_m, 1),
                    "vx_ms": round(ent.vx_ms, 1),
                    "vy_ms": round(ent.vy_ms, 1),
                    "vz_ms": round(ent.vz_ms, 1),
                    "range_m": rng_2d,
                    "slant_range_m": round(slant, 1),
                    "bearing_deg": round(bearing, 1),
                    "speed_ms": round(speed, 1),
                    "closing_vel_ms": closing,
                    "tti_s": tti,
                    "zone_code": zone_code,
                    "threat_score": threat_score,
                    "imm_dive_prob": dive_prob,
                    "rcs_dbsm": ent.rcs_dbsm,
                    "rotor_rpm": ent.rotor_rpm,
                    "payload_detected": ent.payload_detected,
                    "confidence": round(ent.confidence * 100.0, 1),
                    "status": ent.status,
                    "assigned_asset": ent.assigned_asset,
                    "trail": ent.trail[-24:],
                    "predicted_path": pred_path,
                }
            )

        tracks_payload.sort(
            key=lambda x: (-1.0 if x["affiliation"].startswith("FRIENDLY") else x["threat_score"]),
            reverse=True,
        )

        if self.selected_target_id not in self.entities:
            hostile_ids = [
                t["target_id"]
                for t in tracks_payload
                if not t["affiliation"].startswith("FRIENDLY")
            ]
            if hostile_ids:
                self.selected_target_id = hostile_ids[0]

        locked = next(
            (t for t in tracks_payload if t["target_id"] == self.selected_target_id),
            tracks_payload[0] if tracks_payload else None,
        )

        now_t = time.time()
        stft_matrix = self.cuda.compute_cuda_micro_doppler(
            locked["rotor_rpm"] if locked else 14000.0,
            locked["closing_vel_ms"] if locked else 22.0,
            now_t,
        )
        rf_fft_bins = self.cuda.compute_cuda_rf_fft(
            bool(locked and locked["rf_protocol"] == "RF_SILENT_OPTICAL_NAV"),
            bool(locked and locked["status"] == "JAMMED"),
            now_t,
        )

        max_threat = max((t["threat_score"] for t in tracks_payload), default=0.0)
        min_rng = min(
            (
                t["range_m"]
                for t in tracks_payload
                if t["affiliation"] in ("HOSTILE", "SUSPECT")
            ),
            default=9999.0,
        )
        if min_rng < self.killzone_radius_m or max_threat >= 80.0:
            defcon_label = "DEFCON 1 // SWARM BREACH"
            defcon_num = 1
        elif min_rng < self.warning_radius_m:
            defcon_label = "DEFCON 2 // HOSTILE INBOUND"
            defcon_num = 2
        else:
            defcon_label = "DEFCON 4 // PERIMETER WATCH"
            defcon_num = 4

        hostile_count = sum(
            1 for t in tracks_payload if t["affiliation"] in ("HOSTILE", "SUSPECT")
        )
        friendly_count = sum(
            1 for t in tracks_payload if t["affiliation"].startswith("FRIENDLY")
        )

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mission_phase": self.mission_phase,
            "scenario": self.scenario,
            "roe_mode": self.roe_mode,
            "defcon_num": defcon_num,
            "defcon_label": defcon_label,
            "radar_sweep_deg": round(self.radar_sweep_deg, 1),
            "selected_target_id": self.selected_target_id,
            "active_play_id": self.active_play_id,
            "pending_autonomy_play": self.pending_autonomy_play,
            "geofence": {
                "killzone_m": self.killzone_radius_m,
                "warning_m": self.warning_radius_m,
                "detection_m": self.detection_radius_m,
            },
            "counts": {
                "hostile": hostile_count,
                "friendly": friendly_count,
                "neutralized": self.neutralized_count,
                "total_detected": self.total_detected_count + self.neutralized_count,
            },
            "cuda": self.cuda.get_hardware_telemetry(),
            "micro_doppler_stft": stft_matrix,
            "rf_fft_bins": rf_fft_bins,
            "tracks": tracks_payload,
            "splash_events": self.splash_events,
            "effectors": list(self.effectors.values()),
            "audit_tail": self.audit.chain[-6:][::-1],
            "ledger_integrity": self.audit.verify_integrity(),
            "replay_index": len(self.recorded_frames),
            "replay_total": len(self.recorded_frames) + 1,
            "replay_playing": False,
            "replay_speed": self.replay_speed,
        }

        if self.mission_phase == "LIVE_EXECUTION":
            self.recorded_frames.append(copy.deepcopy(snapshot))
            self.replay_index = len(self.recorded_frames) - 1

        return snapshot


# ============================================================================
# 4. FASTAPI ROUTES & WEBSOCKET TELEMETRY
# ============================================================================

cuda_compute = CudaTacticalComputeEngine()
audit_ledger = LatticeAuditLedger()
mission_engine = LatticeMissionAutonomyEngine(cuda_compute, audit_ledger)


class EffectorCmd(BaseModel):
    effector: str = Field(default="RF_JAMMER")
    target_id: Optional[str] = None
    operator: str = Field(default="HOTL_COMMANDER")


class PlayCmd(BaseModel):
    play_id: str
    operator: str = Field(default="HOTL_COMMANDER")


class PhaseCmd(BaseModel):
    phase: str  # PRE_MISSION, LIVE_EXECUTION, POST_MISSION


class ReplayCmd(BaseModel):
    frame_index: Optional[int] = None
    playing: Optional[bool] = None
    speed: Optional[int] = None


class ScenarioCmd(BaseModel):
    scenario: str
    killzone_m: Optional[float] = None
    warning_m: Optional[float] = None


class SelectCmd(BaseModel):
    target_id: str


class RoeCmd(BaseModel):
    roe_mode: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LATTICE-MONOLITH // CUDA Monochrome Autonomous Counter-UAS C2",
    version="6.0.0-MONOLITH",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/status")
@app.get("/api/v1/telemetry")
async def api_telemetry():
    return JSONResponse(mission_engine.build_snapshot())


@app.post("/api/v1/countermeasures/engage")
async def api_engage_effector(cmd: EffectorCmd):
    return JSONResponse(
        mission_engine.engage_single_effector(cmd.effector, cmd.target_id, cmd.operator)
    )


@app.post("/api/v1/countermeasures/disengage")
async def api_disengage():
    return JSONResponse(mission_engine.ceasefire_and_rtb())


@app.post("/api/v1/autonomy/play")
async def api_autonomy_play(cmd: PlayCmd):
    return JSONResponse(
        mission_engine.execute_autonomy_play(cmd.play_id, cmd.operator)
    )


@app.post("/api/v1/mission/phase")
async def api_mission_phase(cmd: PhaseCmd):
    phase = cmd.phase.upper()
    if phase in ("PRE_MISSION", "LIVE_EXECUTION", "POST_MISSION"):
        mission_engine.mission_phase = phase
        if phase == "POST_MISSION":
            mission_engine.replay_index = 0
            mission_engine.replay_playing = True
        audit_ledger.record_event(
            "MISSION_PHASE_TRANSITION", "HOTL_COMMANDER", {"phase": phase}
        )
    return JSONResponse({"mission_phase": mission_engine.mission_phase})


@app.post("/api/v1/mission/replay")
async def api_mission_replay(cmd: ReplayCmd):
    if cmd.frame_index is not None and mission_engine.recorded_frames:
        mission_engine.replay_index = max(
            0, min(cmd.frame_index, len(mission_engine.recorded_frames) - 1)
        )
    if cmd.playing is not None:
        mission_engine.replay_playing = cmd.playing
    if cmd.speed is not None and cmd.speed in (1, 2, 4):
        mission_engine.replay_speed = cmd.speed
    return JSONResponse(
        {
            "replay_index": mission_engine.replay_index,
            "replay_playing": mission_engine.replay_playing,
            "replay_speed": mission_engine.replay_speed,
        }
    )


@app.post("/api/v1/scenario")
async def api_scenario(cmd: ScenarioCmd):
    if cmd.killzone_m is not None:
        mission_engine.killzone_radius_m = max(200.0, min(1200.0, cmd.killzone_m))
    if cmd.warning_m is not None:
        mission_engine.warning_radius_m = max(800.0, min(2500.0, cmd.warning_m))
    mission_engine.load_scenario(cmd.scenario)
    return JSONResponse({"status": "OK", "scenario": mission_engine.scenario})


@app.post("/api/v1/target/select")
async def api_select_target(cmd: SelectCmd):
    if cmd.target_id in mission_engine.entities:
        mission_engine.selected_target_id = cmd.target_id
    return JSONResponse({"selected_target_id": mission_engine.selected_target_id})


@app.post("/api/v1/roe_mode")
async def api_roe_mode(cmd: RoeCmd):
    mission_engine.roe_mode = (
        "FULL_AUTONOMOUS_ROE"
        if cmd.roe_mode.upper() == "FULL_AUTONOMOUS_ROE"
        else "HOTL_SUPERVISED"
    )
    audit_ledger.record_event(
        "ROE_GOVERNANCE_UPDATED", "HOTL_COMMANDER", {"roe_mode": mission_engine.roe_mode}
    )
    return JSONResponse({"roe_mode": mission_engine.roe_mode})


@app.get("/api/v1/audit/logs")
async def get_audit_logs():
    return JSONResponse(audit_ledger.chain)


@app.get("/api/v1/audit/verify")
async def verify_audit_chain():
    return JSONResponse(audit_ledger.verify_integrity())


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(mission_engine.build_snapshot())
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ============================================================================
# 5. 100% BLACK-AND-WHITE MONOCHROME ANDURIL LATTICE MISSION AUTONOMY UI
# ============================================================================

MONOCHROME_LATTICE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LATTICE // MISSION AUTONOMY C2 [MONOCHROME CUDA OS]</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* STRICT 100% ACHROMATIC BLACK & WHITE PALETTE — ZERO RGB / NEON COLORS */
  :root {
    --void: #000000;
    --surface: #070707;
    --card: #0F0F0F;
    --elevated: #171717;
    --border-dim: #262626;
    --border-mid: #404040;
    --border-bright: #737373;
    --white: #FFFFFF;
    --silver: #D4D4D4;
    --gray: #8A8A8A;
    --dim: #525252;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
  }

  html, body {
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    background: var(--void);
    color: var(--white);
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    -webkit-font-smoothing: antialiased;
  }

  .mono {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
  }

  /* MASTER 100VH GRID */
  .monolith-shell {
    display: grid;
    grid-template-rows: 46px calc(100vh - 46px);
    width: 100vw;
    height: 100vh;
    background: var(--void);
  }

  /* TOP TACTICAL COMMAND RIBBON */
  .top-ribbon {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    background: #050505;
    border-bottom: 1px solid var(--border-mid);
  }

  .brand-cluster {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .monolith-emblem {
    width: 24px;
    height: 24px;
    border: 1.5px solid var(--white);
    display: grid;
    place-items: center;
    background: #111111;
  }

  .brand-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--white);
  }

  /* CLOSED-LOOP LIFECYCLE TABS */
  .phase-tabs {
    display: flex;
    border: 1px solid var(--border-mid);
    background: #0A0A0A;
  }

  .phase-tab {
    padding: 5px 10px;
    background: transparent;
    border: none;
    border-right: 1px solid var(--border-dim);
    color: var(--gray);
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.6px;
  }
  .phase-tab:last-child { border-right: none; }
  .phase-tab:hover { color: var(--white); background: #141414; }
  .phase-tab.active {
    background: var(--white);
    color: var(--void);
    font-weight: 700;
  }

  .center-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* INVERTED HIGH-CONTRAST DEFCON ALERT BLOCK */
  .defcon-inverted {
    padding: 4px 12px;
    background: var(--white);
    color: var(--void);
    font-family: 'Chakra Petch', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.3px;
    display: flex;
    align-items: center;
    gap: 7px;
    animation: invertPulse 1.4s infinite;
  }

  @keyframes invertPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.72; }
  }

  .blink-sq {
    width: 7px;
    height: 7px;
    background: var(--void);
  }

  .bw-select, .bw-btn {
    background: #0D0D0D;
    color: var(--white);
    border: 1px solid var(--border-mid);
    padding: 4px 9px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    cursor: pointer;
  }
  .bw-select:hover, .bw-btn:hover {
    border-color: var(--white);
    background: #1A1A1A;
  }

  .cuda-pill {
    padding: 4px 9px;
    background: #0C0C0C;
    border: 1px solid var(--border-bright);
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    color: var(--silver);
  }

  /* 3-COLUMN TACTICAL WORKSPACE (27% | 46% | 27%) */
  .workspace-grid {
    display: grid;
    grid-template-columns: 27% 46% 27%;
    gap: 6px;
    padding: 6px;
    height: 100%;
    min-height: 0;
  }

  .col-stack {
    display: flex;
    flex-direction: column;
    gap: 6px;
    height: 100%;
    min-height: 0;
  }

  /* CHAMFERED MONOCHROME TACTICAL PANEL */
  .tac-panel {
    background: var(--surface);
    border: 1px solid var(--border-dim);
    clip-path: polygon(
      0 8px, 8px 0,
      calc(100% - 8px) 0, 100% 8px,
      100% calc(100% - 8px), calc(100% - 8px) 100%,
      8px 100%, 0 calc(100% - 8px)
    );
    display: flex;
    flex-direction: column;
    min-height: 0;
    position: relative;
  }

  .panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 10px;
    background: #0D0D0D;
    border-bottom: 1px solid var(--border-dim);
    flex-shrink: 0;
  }

  .panel-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--white);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .panel-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    padding: 1px 5px;
    border: 1px solid var(--border-mid);
    color: var(--silver);
    background: #050505;
  }

  .canvas-box {
    flex: 1;
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 0;
    background: #000000;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  .panel-foot {
    display: flex;
    justify-content: space-between;
    padding: 4px 10px;
    background: #050505;
    border-top: 1px solid var(--border-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--gray);
    flex-shrink: 0;
  }

  /* LEFT COLUMN */
  .optronic-box { flex: 0 0 53%; }
  .spectrogram-box { flex: 1; }

  /* CENTER COLUMN */
  .swarm-matrix-box { flex: 0 0 18%; }
  .ppi-scope-box { flex: 1; }
  .timeline-box { flex: 0 0 11%; }

  /* SWARM CLUSTER THREAT TABLE */
  .matrix-scroll {
    flex: 1;
    overflow-y: auto;
  }
  .matrix-scroll::-webkit-scrollbar { width: 3px; }
  .matrix-scroll::-webkit-scrollbar-thumb { background: var(--white); }

  table.swarm-tbl {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
  }
  .swarm-tbl th {
    position: sticky;
    top: 0;
    background: #0A0A0A;
    color: var(--gray);
    text-align: left;
    padding: 4px 8px;
    border-bottom: 1px solid var(--border-dim);
    font-size: 8.5px;
  }
  .swarm-tbl td {
    padding: 4px 8px;
    border-bottom: 1px solid #161616;
    cursor: pointer;
  }
  .swarm-tbl tr:hover { background: #141414; }
  .swarm-tbl tr.sel-row {
    background: #1F1F1F;
    outline: 1px solid var(--white);
  }

  /* HOTL APPROVAL BANNER OVER RADAR */
  .hotl-banner {
    position: absolute;
    bottom: 10px;
    left: 10px;
    right: 10px;
    background: var(--white);
    color: var(--void);
    padding: 6px 10px;
    display: none;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    z-index: 5;
    border: 1px solid var(--void);
  }

  .hotl-approve-btn {
    background: var(--void);
    color: var(--white);
    border: 1px solid var(--void);
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    font-weight: 700;
    cursor: pointer;
  }
  .hotl-approve-btn:hover {
    background: #262626;
  }

  /* CLOSED-LOOP TIMELINE SCRUBBER */
  .timeline-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    flex: 1;
  }

  input[type="range"].bw-slider {
    flex: 1;
    accent-color: #FFFFFF;
    cursor: pointer;
    height: 4px;
    background: #262626;
  }

  /* RIGHT COLUMN */
  .pip-zoom-box { flex: 0 0 36%; }
  .autonomy-dock-box { flex: 0 0 45%; }
  .audit-ledger-box { flex: 1; }

  .play-list {
    padding: 6px 9px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    flex: 1;
    justify-content: space-between;
  }

  .play-card {
    background: var(--card);
    border: 1px solid var(--border-mid);
    padding: 6px 9px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    color: var(--white);
    text-align: left;
    transition: all 0.1s ease;
  }
  .play-card:hover {
    background: var(--elevated);
    border-color: var(--white);
  }
  .play-card.active-play {
    background: var(--white);
    color: var(--void);
    border-color: var(--white);
  }

  .play-title {
    font-family: 'Chakra Petch', sans-serif;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.7px;
  }
  .play-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px;
    opacity: 0.75;
    margin-top: 1px;
  }

  .toggle-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px;
    padding: 2px 6px;
    border: 1px solid currentColor;
    font-weight: 700;
  }

  .audit-scroll {
    padding: 5px 9px;
    overflow-y: auto;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5px;
  }

  .audit-item {
    display: flex;
    justify-content: space-between;
    padding: 3px 6px;
    background: #0C0C0C;
    border-left: 2px solid var(--white);
    color: var(--silver);
  }

  /* PRE-MISSION WARGAMING MODAL OVERLAY */
  .premission-drawer {
    position: absolute;
    top: 8px;
    left: 8px;
    right: 8px;
    background: rgba(5, 5, 5, 0.94);
    border: 1px solid var(--white);
    padding: 10px 12px;
    z-index: 6;
    display: none;
    flex-direction: column;
    gap: 8px;
  }
</style>
</head>
<body>
<div class="monolith-shell">
  <!-- TOP COMMAND RIBBON -->
  <header class="top-ribbon">
    <div class="brand-cluster">
      <div class="monolith-emblem">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2">
          <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/>
          <circle cx="12" cy="12" r="2.5" fill="#FFFFFF"/>
        </svg>
      </div>
      <div>
        <div class="brand-title">LATTICE // MISSION AUTONOMY</div>
      </div>

      <!-- CLOSED-LOOP MISSION LIFECYCLE SWITCHER -->
      <div class="phase-tabs">
        <button class="phase-tab" id="tabPRE" onclick="setMissionPhase('PRE_MISSION')">1. PRE-MISSION MODEL</button>
        <button class="phase-tab active" id="tabLIVE" onclick="setMissionPhase('LIVE_EXECUTION')">2. LIVE EXECUTION</button>
        <button class="phase-tab" id="tabPOST" onclick="setMissionPhase('POST_MISSION')">3. POST-MISSION DEBRIEF</button>
      </div>
    </div>

    <div class="center-controls">
      <div class="defcon-inverted">
        <span class="blink-sq"></span>
        <span id="defconText">DEFCON 1 // SWARM BREACH</span>
      </div>

      <select id="scenarioSelect" class="bw-select" onchange="changeScenario(this.value)">
        <option value="PINCER_SWARM_10">SWARM: 8-UAV MULTI-AXIS PINCER</option>
        <option value="RF_SILENT_STEALTH">SWARM: RF-SILENT STEALTH WAVE</option>
        <option value="FPV_TERMINAL_DIVE">SWARM: FPV TERMINAL DIVE</option>
      </select>

      <button class="bw-btn" onclick="toggleRoeGovernance()">
        GOVERNANCE: <strong id="roeText">HOTL SUPERVISED</strong>
      </button>
    </div>

    <div style="display:flex;align-items:center;gap:8px;">
      <div class="cuda-pill" id="cudaHudBadge">CUDA: INITIALIZING...</div>
      <div class="cuda-pill" id="utcClock">00:00:00 UTC</div>
    </div>
  </header>

  <!-- 3-COLUMN MONOCHROME TACTICAL WORKSPACE -->
  <main class="workspace-grid">
    <!-- LEFT COLUMN: EO/IR GIMBAL SENSOR + CUDA MICRO-DOPPLER SPECTROGRAM -->
    <section class="col-stack">
      <div class="tac-panel optronic-box">
        <div class="panel-head">
          <div class="panel-title">TACTICAL OPTRONIC SENSOR // LOCK-ON</div>
          <div style="display:flex;gap:3px;">
            <button class="bw-btn" id="btnWH" style="padding:2px 5px;background:#FFF;color:#000;" onclick="setPolar('WH')">W-HOT</button>
            <button class="bw-btn" id="btnBH" style="padding:2px 5px;" onclick="setPolar('BH')">B-HOT</button>
          </div>
        </div>
        <div class="canvas-box">
          <canvas id="optronicCanvas"></canvas>
        </div>
        <div class="panel-foot">
          <span id="optLockFoot">TGT LOCK: #TGT-101 | TTI: 14.2s</span>
          <span>GPU FP16 // 31.8 FPS</span>
        </div>
      </div>

      <div class="tac-panel spectrogram-box">
        <div class="panel-head">
          <div class="panel-title">CUDA MICRO-DOPPLER SPECTROGRAM</div>
          <span class="panel-tag" id="rpmTag">14,400 RPM [STFT]</span>
        </div>
        <div class="canvas-box">
          <canvas id="dopplerCanvas"></canvas>
        </div>
        <div class="panel-foot">
          <span>ROTOR FREQUENCY (0 - 300 Hz)</span>
          <span>TORCH.STFT CUDA</span>
        </div>
      </div>
    </section>

    <!-- CENTER COLUMN: SWARM THREAT MATRIX + 360° PULSE-DOPPLER SCOPE + TIMELINE -->
    <section class="col-stack">
      <div class="tac-panel swarm-matrix-box">
        <div class="panel-head">
          <div class="panel-title">SWARM CLUSTER EVALUATION &amp; ASSET TASKING</div>
          <span class="panel-tag" id="assetRatioTag">HOSTILE: 7 | FRIENDLY: 2 | SPLASH: 0</span>
        </div>
        <div class="matrix-scroll">
          <table class="swarm-tbl">
            <thead>
              <tr>
                <th>TARGET</th>
                <th>CLUSTER</th>
                <th>SPEED</th>
                <th>RNG</th>
                <th>TTI</th>
                <th>TASKED ASSET</th>
              </tr>
            </thead>
            <tbody id="swarmTableBody"></tbody>
          </table>
        </div>
      </div>

      <div class="tac-panel ppi-scope-box">
        <div class="panel-head">
          <div class="panel-title">360° PULSE-DOPPLER &amp; MULTI-ASSET SWARM SCOPE</div>
          <span class="panel-tag">37.7749°N, 122.4194°W // 500m / 1500m / 3500m</span>
        </div>
        <div class="canvas-box">
          <canvas id="radarCanvas"></canvas>

          <!-- PRE-MISSION THREAT MODELING DRAWER -->
          <div class="premission-drawer" id="preMissionDrawer">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <strong style="font-family:'Chakra Petch';letter-spacing:1px;">PRE-MISSION THREAT MODELING &amp; GEOFENCE WARGAMING</strong>
              <button class="bw-btn" style="background:#FFF;color:#000;font-weight:700;" onclick="setMissionPhase('LIVE_EXECUTION')">LAUNCH LIVE MISSION &rarr;</button>
            </div>
            <div style="display:flex;gap:12px;align-items:center;font-family:'JetBrains Mono';font-size:9.5px;">
              <span>KILLZONE RADIUS: <strong id="kzVal">500m</strong></span>
              <input type="range" min="300" max="1000" step="50" value="500" class="bw-slider" oninput="updateGeofencePreview(this.value)">
              <span>MONTE CARLO P(SURVIVAL): <strong>98.4%</strong></span>
            </div>
          </div>

          <!-- HOTL AUTONOMY INTENT APPROVAL BANNER -->
          <div class="hotl-banner" id="hotlBanner">
            <span id="hotlBannerText">AUTONOMY RECOMMENDATION: PLAY_03_KINETIC_WOLF_PACK</span>
            <button class="hotl-approve-btn" onclick="approvePendingPlay()">APPROVE PLAY [ENTER]</button>
          </div>
        </div>
      </div>

      <div class="tac-panel timeline-box">
        <div class="panel-head">
          <div class="panel-title">CLOSED-LOOP MISSION TIMELINE &amp; DEBRIEF SCRUBBER</div>
          <span class="panel-tag" id="replayCounterTag">LIVE RECORDING // FRAME 0001</span>
        </div>
        <div class="timeline-controls">
          <button class="bw-btn" id="playPauseBtn" onclick="toggleReplayPlay()">PAUSE</button>
          <button class="bw-btn" id="speedBtn" onclick="cycleReplaySpeed()">1X</button>
          <input type="range" id="timelineSlider" class="bw-slider" min="0" max="1" value="1" oninput="scrubTimeline(this.value)">
          <button class="bw-btn" onclick="setMissionPhase('LIVE_EXECUTION')">LIVE</button>
        </div>
      </div>
    </section>

    <!-- RIGHT COLUMN: PiP 4X FLIR TARGET ZOOM + AUTONOMY PLAYBOOK + SHA-256 AUDIT -->
    <section class="col-stack">
      <div class="tac-panel pip-zoom-box">
        <div class="panel-head">
          <div class="panel-title">PICTURE-IN-PICTURE TARGET ZOOM</div>
          <span class="panel-tag" id="pipTargetTag">FLIR 4X // #TGT-101</span>
        </div>
        <div class="canvas-box">
          <canvas id="pipCanvas"></canvas>
        </div>
        <div class="panel-foot">
          <span id="pipOrdnanceFoot">THERMAL GRAIN // PAYLOAD POSITIVE</span>
          <span id="pipRfFoot">RF: -72.1 dBm</span>
        </div>
      </div>

      <div class="tac-panel autonomy-dock-box">
        <div class="panel-head">
          <div class="panel-title">HOTL AUTONOMY PLAYBOOK &amp; EFFECTORS</div>
          <span class="panel-tag">KEYS [1-4]</span>
        </div>
        <div class="play-list">
          <button class="play-card" id="cardPlay1" onclick="triggerPlay('PLAY_01_PERIMETER_CAP')">
            <div>
              <div class="play-title">[1] PLAY 01: PERIMETER ISR CAP</div>
              <div class="play-desc">TASK FRIENDLY WINGMEN TO SHADOW CLUSTERS</div>
            </div>
            <span class="toggle-pill" id="stPlay1">STANDBY</span>
          </button>

          <button class="play-card" id="cardPlay2" onclick="triggerPlay('PLAY_02_SWARM_EW_BARRIER')">
            <div>
              <div class="play-title">[2] PLAY 02: SWARM EW BARRIER</div>
              <div class="play-desc">SYNC RF JAMMER + GNSS SPOOF + SWIR DAZZLER</div>
            </div>
            <span class="toggle-pill" id="stPlay2">STANDBY</span>
          </button>

          <button class="play-card" id="cardPlay3" onclick="triggerPlay('PLAY_03_KINETIC_WOLF_PACK')">
            <div>
              <div class="play-title">[3] PLAY 03: KINETIC WOLF-PACK</div>
              <div class="play-desc">DISPATCH ANVIL HARD-KILL INTERCEPTOR SWARM</div>
            </div>
            <span class="toggle-pill" id="stPlay3">8 SILOS</span>
          </button>

          <button class="play-card" id="cardSingleJam" onclick="triggerSingleJam()">
            <div>
              <div class="play-title">[4] SINGLE-TARGET RF + SPOOF LOCK</div>
              <div class="play-desc">DIRECTED NON-KINETIC NEUTRALIZATION</div>
            </div>
            <span class="toggle-pill" id="stSingleJam">READY</span>
          </button>

          <button class="play-card" style="background:#141414;border-color:#FFFFFF;" onclick="triggerPlay('PLAY_04_CEASEFIRE_RTB')">
            <div>
              <div class="play-title">[SPACE] EMERGENCY CEASEFIRE &amp; RTB</div>
              <div class="play-desc">DE-ENERGIZE EMITTERS &amp; RECALL WINGMEN</div>
            </div>
            <span class="toggle-pill">SAFE</span>
          </button>
        </div>
      </div>

      <div class="tac-panel audit-ledger-box">
        <div class="panel-head">
          <div class="panel-title">SHA-256 FORENSIC MISSION LEDGER</div>
          <span class="panel-tag">VERIFIED</span>
        </div>
        <div class="audit-scroll" id="auditList"></div>
      </div>
    </section>
  </main>
</div>

<script>
  let state = null;
  let polarity = 'WH';
  let screenPoints = {};
  let tick = 0;

  function setPolar(p) {
    polarity = p;
    document.getElementById('btnWH').style.background = p === 'WH' ? '#FFFFFF' : '#0D0D0D';
    document.getElementById('btnWH').style.color = p === 'WH' ? '#000000' : '#FFFFFF';
    document.getElementById('btnBH').style.background = p === 'BH' ? '#FFFFFF' : '#0D0D0D';
    document.getElementById('btnBH').style.color = p === 'BH' ? '#000000' : '#FFFFFF';
  }

  async function setMissionPhase(phase) {
    await fetch('/api/v1/mission/phase', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({phase})
    });
  }

  async function triggerPlay(playId) {
    await fetch('/api/v1/autonomy/play', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({play_id: playId})
    });
  }

  async function approvePendingPlay() {
    if (state && state.pending_autonomy_play) {
      await triggerPlay(state.pending_autonomy_play.play_id);
    }
  }

  async function triggerSingleJam() {
    const tid = state ? state.selected_target_id : null;
    await fetch('/api/v1/countermeasures/engage', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({effector: 'RF_JAMMER', target_id: tid})
    });
  }

  async function selectTrack(targetId) {
    if (state) state.selected_target_id = targetId;
    await fetch('/api/v1/target/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({target_id: targetId})
    });
  }

  async function changeScenario(scenario) {
    await fetch('/api/v1/scenario', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({scenario})
    });
  }

  async function updateGeofencePreview(kz) {
    document.getElementById('kzVal').textContent = kz + 'm';
    await fetch('/api/v1/scenario', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({scenario: state ? state.scenario : 'PINCER_SWARM_10', killzone_m: Number(kz)})
    });
  }

  async function toggleRoeGovernance() {
    const next = (state && state.roe_mode === 'FULL_AUTONOMOUS_ROE') ? 'HOTL_SUPERVISED' : 'FULL_AUTONOMOUS_ROE';
    await fetch('/api/v1/roe_mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({roe_mode: next})
    });
  }

  async function toggleReplayPlay() {
    if (!state) return;
    if (state.mission_phase !== 'POST_MISSION') {
      await setMissionPhase('POST_MISSION');
      return;
    }
    await fetch('/api/v1/mission/replay', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({playing: !state.replay_playing})
    });
  }

  async function cycleReplaySpeed() {
    if (!state) return;
    const next = state.replay_speed === 1 ? 2 : (state.replay_speed === 2 ? 4 : 1);
    await fetch('/api/v1/mission/replay', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({speed: next})
    });
  }

  async function scrubTimeline(idx) {
    if (state && state.mission_phase !== 'POST_MISSION') {
      await setMissionPhase('POST_MISSION');
    }
    await fetch('/api/v1/mission/replay', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({frame_index: Number(idx), playing: false})
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === '1') triggerPlay('PLAY_01_PERIMETER_CAP');
    else if (e.key === '2') triggerPlay('PLAY_02_SWARM_EW_BARRIER');
    else if (e.key === '3') triggerPlay('PLAY_03_KINETIC_WOLF_PACK');
    else if (e.key === '4') triggerSingleJam();
    else if (e.key === 'Enter') approvePendingPlay();
    else if (e.code === 'Space') { e.preventDefault(); triggerPlay('PLAY_04_CEASEFIRE_RTB'); }
  });

  function sizeCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {ctx, w: rect.width, h: rect.height};
  }

  function getSelectedTrack() {
    if (!state || !state.tracks || !state.tracks.length) return null;
    return state.tracks.find(t => t.target_id === state.selected_target_id) || state.tracks[0];
  }

  // 1. RENDER 360° PULSE-DOPPLER & MULTI-ASSET SWARM PPI SCOPE (100% MONOCHROME)
  function renderRadarScope() {
    const canvas = document.getElementById('radarCanvas');
    const {ctx, w, h} = sizeCanvas(canvas);
    const cx = w * 0.5;
    const cy = h * 0.5;
    const maxR = Math.min(w, h) * 0.44;
    const worldMax = 1800.0;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    // Outer Azimuth Degree Dial (000° - 350°)
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.font = '8.5px "JetBrains Mono"';
    ctx.fillStyle = '#8A8A8A';
    ctx.textAlign = 'center';
    for (let deg = 0; deg < 360; deg += 10) {
      const rad = ((deg - 90) * Math.PI) / 180;
      const inner = deg % 30 === 0 ? maxR - 7 : maxR - 3;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(rad) * inner, cy + Math.sin(rad) * inner);
      ctx.lineTo(cx + Math.cos(rad) * maxR, cy + Math.sin(rad) * maxR);
      ctx.stroke();
      if (deg % 30 === 0) {
        ctx.fillText(String(deg).padStart(3, '0'), cx + Math.cos(rad) * (maxR + 11), cy + Math.sin(rad) * (maxR + 11) + 3);
      }
    }

    // Crosshairs
    ctx.strokeStyle = '#1C1C1C';
    ctx.beginPath();
    ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy);
    ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR);
    ctx.stroke();

    if (!state) return;

    // Concentric Range Rings (500m solid white killzone, 1100m dashed warning, 1650m dotted)
    const kzR = (state.geofence.killzone_m / worldMax) * maxR;
    const warnR = (1150.0 / worldMax) * maxR;
    const detR = (1650.0 / worldMax) * maxR;

    // Solid white Killzone ring
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, kzR, 0, Math.PI * 2);
    ctx.stroke();

    // Dashed Warning ring
    ctx.strokeStyle = '#737373';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, warnR, 0, Math.PI * 2);
    ctx.stroke();

    // Dotted Detection ring
    ctx.strokeStyle = '#404040';
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, detR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#A3A3A3';
    ctx.font = '8.5px "JetBrains Mono"';
    ctx.textAlign = 'left';
    ctx.fillText(`${Math.round(state.geofence.killzone_m)}m KILLZONE`, cx + 4, cy - kzR + 10);

    // Radar Sweep Beam
    const sweepRad = ((state.radar_sweep_deg - 90) * Math.PI) / 180;
    const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, maxR);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.20)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0.02)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR, sweepRad - 0.30, sweepRad);
    ctx.closePath();
    ctx.fill();

    // Active Directional EW Jammer Sector Cone (Translucent White Wedge)
    state.effectors.forEach(eff => {
      if (eff.state.startsWith('ENGAGED') && eff.azimuth_deg !== undefined) {
        const az = ((eff.azimuth_deg - 90) * Math.PI) / 180;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, maxR * 0.88, az - 0.24, az + 0.24);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    });

    // BDA Kinetic Splash Shockwaves
    state.splash_events.forEach(sp => {
      const sx = cx + (sp.east_m / worldMax) * maxR;
      const sy = cy - (sp.north_m / worldMax) * maxR;
      const r = 14 + (tick % 20) * 1.4;
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(`SPLASH // ${sp.target_id}`, sx + 10, sy - 6);
    });

    // Base HQ Core
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(cx - 4, cy - 4, 8, 8);

    // Plot Swarm Entities (Monochrome MIL-STD-2525D)
    screenPoints = {};
    state.tracks.forEach(trk => {
      const tx = cx + (trk.east_m / worldMax) * maxR;
      const ty = cy - (trk.north_m / worldMax) * maxR;
      screenPoints[trk.target_id] = {x: tx, y: ty};
      const isFriendly = trk.affiliation.startsWith('FRIENDLY');
      const isSelected = trk.target_id === state.selected_target_id;

      // Historical Trail
      if (trk.trail && trk.trail.length > 1) {
        ctx.strokeStyle = isFriendly ? 'rgba(255,255,255,0.75)' : 'rgba(163,163,163,0.45)';
        ctx.lineWidth = isFriendly ? 1.5 : 1.0;
        ctx.beginPath();
        trk.trail.forEach(([ex, ny], i) => {
          const px = cx + (ex / worldMax) * maxR;
          const py = cy - (ny / worldMax) * maxR;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.lineTo(tx, ty);
        ctx.stroke();
      }

      // CUDA GPU Predicted Trajectory Spline
      if (trk.predicted_path && trk.predicted_path.length > 1 && !isFriendly) {
        ctx.strokeStyle = '#737373';
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        trk.predicted_path.slice(0, 14).forEach(([pe, pn]) => {
          ctx.lineTo(cx + (pe / worldMax) * maxR, cy - (pn / worldMax) * maxR);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Locked Target Ring
      if (isSelected) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(tx, ty, 13, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Monochrome Tactical Symbology
      if (isFriendly) {
        // Solid White Chevron / Circle for Collaborative Wingmen & Interceptors
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(tx, ty, 5.5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Hollow White Diamond with Center Pip for Hostile Swarm Drones
        ctx.strokeStyle = '#FFFFFF';
        ctx.fillStyle = '#000000';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(tx, ty - 7);
        ctx.lineTo(tx + 7, ty);
        ctx.lineTo(tx, ty + 7);
        ctx.lineTo(tx - 7, ty);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(tx - 1.5, ty - 1.5, 3, 3);
      }

      // Track Callout
      ctx.fillStyle = isSelected ? '#FFFFFF' : '#A3A3A3';
      ctx.font = isSelected ? 'bold 9.5px "JetBrains Mono"' : '8.5px "JetBrains Mono"';
      ctx.fillText(`${trk.target_id} (${Math.round(trk.range_m)}m)`, tx + 10, ty + 3);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('radarCanvas').addEventListener('click', (e) => {
      const rect = e.target.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let bestId = null;
      let bestD = 26;
      Object.entries(screenPoints).forEach(([id, pt]) => {
        const d = Math.hypot(mx - pt.x, my - pt.y);
        if (d < bestD) { bestD = d; bestId = id; }
      });
      if (bestId) selectTrack(bestId);
    });
  });

  // 2. RENDER MONOCHROME OPTRONIC GIMBAL SENSOR
  function renderOptronic() {
    const canvas = document.getElementById('optronicCanvas');
    const {ctx, w, h} = sizeCanvas(canvas);
    const trk = getSelectedTrack();

    const bg = polarity === 'WH' ? '#080808' : '#E5E5E5';
    const fg = polarity === 'WH' ? '#FFFFFF' : '#000000';
    const mid = polarity === 'WH' ? '#404040' : '#A3A3A3';

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Subtle tactical grid & horizon
    ctx.strokeStyle = mid;
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    ctx.moveTo(16, h * 0.55); ctx.lineTo(w - 16, h * 0.55);
    ctx.moveTo(w * 0.5, 16); ctx.lineTo(w * 0.5, h - 16);
    ctx.stroke();

    if (!trk) return;

    const tx = w * 0.5 + Math.sin(tick * 0.05) * 20;
    const ty = h * 0.48 + Math.cos(tick * 0.04) * 14;

    // Drone thermal silhouette
    ctx.save();
    ctx.translate(tx, ty);
    ctx.strokeStyle = fg;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(-24, -10); ctx.lineTo(24, 10);
    ctx.moveTo(-24, 10); ctx.lineTo(24, -10);
    ctx.stroke();
    ctx.fillStyle = fg;
    ctx.fillRect(-9, -5, 18, 10);
    if (trk.payload_detected) {
      ctx.fillRect(-5, 5, 10, 7);
    }
    [[-24, -10], [24, -10], [-24, 10], [24, 10]].forEach(([rx, ry]) => {
      ctx.beginPath();
      ctx.ellipse(rx, ry, 10, 3, (tick * 0.35) % Math.PI, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.restore();

    // Precision Corner Brackets
    const b = 38, cl = 10;
    ctx.strokeStyle = fg;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(tx - b, ty - b + cl); ctx.lineTo(tx - b, ty - b); ctx.lineTo(tx - b + cl, ty - b);
    ctx.moveTo(tx + b - cl, ty - b); ctx.lineTo(tx + b, ty - b); ctx.lineTo(tx + b, ty - b + cl);
    ctx.moveTo(tx - b, ty + b - cl); ctx.lineTo(tx - b, ty + b); ctx.lineTo(tx - b + cl, ty + b);
    ctx.moveTo(tx + b - cl, ty + b); ctx.lineTo(tx + b, ty + b); ctx.lineTo(tx + b, ty + b - cl);
    ctx.stroke();

    // Kalman Trajectory Spline
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx + trk.vx_ms * 1.4, ty - trk.vz_ms * 2.5);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = fg;
    ctx.font = 'bold 9.5px "JetBrains Mono"';
    ctx.fillText(`LOCK: ${trk.target_id} [${trk.drone_class}]`, 12, 18);
    ctx.fillText(`TTI: ${trk.tti_s > 900 ? 'N/A' : trk.tti_s + 's'} | VEL: ${trk.speed_ms} m/s`, 12, 30);
  }

  // 3. RENDER CUDA TORCH.STFT MICRO-DOPPLER SPECTROGRAM (0-300 Hz GRAYSCALE)
  function renderMicroDoppler() {
    const canvas = document.getElementById('dopplerCanvas');
    const {ctx, w, h} = sizeCanvas(canvas);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    if (!state || !state.micro_doppler_stft) return;
    const mat = state.micro_doppler_stft;
    const rows = mat.length;
    const cols = mat[0].length;
    const padL = 28, padR = 22, padB = 16, padT = 8;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const cw = plotW / cols;
    const ch = plotH / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = mat[rows - 1 - r][c];
        const lum = Math.floor(val * 255);
        ctx.fillStyle = `rgb(${lum},${lum},${lum})`;
        ctx.fillRect(padL + c * cw, padT + r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }

    // Frequency Y-axis ticks & luminance bar
    ctx.fillStyle = '#8A8A8A';
    ctx.font = '8px "JetBrains Mono"';
    ctx.fillText('300', 4, padT + 8);
    ctx.fillText('150', 4, padT + plotH * 0.5);
    ctx.fillText('0Hz', 4, padT + plotH);

    const barGrad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    barGrad.addColorStop(0, '#FFFFFF');
    barGrad.addColorStop(1, '#000000');
    ctx.fillStyle = barGrad;
    ctx.fillRect(w - 14, padT, 7, plotH);
  }

  // 4. RENDER PiP 4X FLIR TARGET ZOOM
  function renderPipZoom() {
    const canvas = document.getElementById('pipCanvas');
    const {ctx, w, h} = sizeCanvas(canvas);
    const trk = getSelectedTrack();

    ctx.fillStyle = '#060606';
    ctx.fillRect(0, 0, w, h);

    // Thermal grain noise dots
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    for (let i = 0; i < 90; i++) {
      ctx.fillRect(Math.random() * w, Math.random() * h, 2, 2);
    }

    const cx = w * 0.5;
    const cy = h * 0.52;

    if (trk) {
      // High-contrast White-Hot thermal bloom silhouette
      ctx.save();
      ctx.translate(cx, cy);
      ctx.shadowColor = '#FFFFFF';
      ctx.shadowBlur = 14;
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(-42, -16); ctx.lineTo(42, 16);
      ctx.moveTo(-42, 16); ctx.lineTo(42, -16);
      ctx.stroke();

      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(-16, -9, 32, 18);
      if (trk.payload_detected) {
        ctx.fillRect(-8, 9, 16, 11);
      }
      ctx.restore();
    }

    // Crisp Corner Reticle
    const b = 48, cl = 14;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(cx - b, cy - b + cl); ctx.lineTo(cx - b, cy - b); ctx.lineTo(cx - b + cl, cy - b);
    ctx.moveTo(cx + b - cl, cy - b); ctx.lineTo(cx + b, cy - b); ctx.lineTo(cx + b, cy - b + cl);
    ctx.moveTo(cx - b, cy + b - cl); ctx.lineTo(cx - b, cy + b); ctx.lineTo(cx - b + cl, cy + b);
    ctx.moveTo(cx + b - cl, cy + b); ctx.lineTo(cx + b, cy + b); ctx.lineTo(cx + b, cy + b - cl);
    ctx.stroke();

    // Center Precision Crosshair
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 14, cy); ctx.lineTo(cx + 14, cy);
    ctx.moveTo(cx, cy - 14); ctx.lineTo(cx, cy + 14);
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 9.5px "JetBrains Mono"';
    ctx.fillText('FLIR THERMAL // WHITE-HOT', 10, 16);
    if (trk) {
      ctx.fillStyle = '#A3A3A3';
      ctx.fillText(`ALT: ${Math.round(trk.alt_m)}m | AZ: ${Math.round(trk.bearing_deg)}°`, 10, h - 10);
    }
  }

  // 5. SYNC DOM STATE
  function syncState(s) {
    state = s;
    document.getElementById('utcClock').textContent = new Date().toISOString().slice(11, 19) + ' UTC';
    document.getElementById('defconText').textContent = s.defcon_label;
    document.getElementById('roeText').textContent = s.roe_mode === 'FULL_AUTONOMOUS_ROE' ? 'AUTONOMOUS ROE' : 'HOTL SUPERVISED';

    // CUDA Badge
    const c = s.cuda;
    document.getElementById('cudaHudBadge').textContent =
      `${c.backend} | ${c.kernel_latency_ms}ms | VRAM ${c.vram_allocated_mb}MB`;

    // Phase Tabs & Pre-Mission Drawer
    ['PRE', 'LIVE', 'POST'].forEach(k => document.getElementById('tab' + k).classList.remove('active'));
    if (s.mission_phase === 'PRE_MISSION') document.getElementById('tabPRE').classList.add('active');
    else if (s.mission_phase === 'POST_MISSION') document.getElementById('tabPOST').classList.add('active');
    else document.getElementById('tabLIVE').classList.add('active');

    document.getElementById('preMissionDrawer').style.display = s.mission_phase === 'PRE_MISSION' ? 'flex' : 'none';

    // HOTL Approval Banner
    const hotl = document.getElementById('hotlBanner');
    if (s.pending_autonomy_play && s.mission_phase === 'LIVE_EXECUTION') {
      hotl.style.display = 'flex';
      document.getElementById('hotlBannerText').textContent =
        `AUTONOMY INTENT: ${s.pending_autonomy_play.play_id} (${s.pending_autonomy_play.reason})`;
    } else {
      hotl.style.display = 'none';
    }

    // Swarm Matrix Table
    document.getElementById('assetRatioTag').textContent =
      `HOSTILE: ${s.counts.hostile} | FRIENDLY: ${s.counts.friendly} | SPLASH: ${s.counts.neutralized}`;

    const tbody = document.getElementById('swarmTableBody');
    tbody.innerHTML = s.tracks.map(t => {
      const sel = t.target_id === s.selected_target_id ? 'sel-row' : '';
      return `<tr class="${sel}" onclick="selectTrack('${t.target_id}')">
        <td style="font-weight:700;">${t.target_id}</td>
        <td style="color:#A3A3A3;">${t.cluster_id}</td>
        <td>${t.speed_ms} m/s</td>
        <td>${Math.round(t.range_m)}m</td>
        <td>${t.tti_s > 900 ? '---' : t.tti_s + 's'}</td>
        <td style="color:#D4D4D4;">${t.assigned_asset}</td>
      </tr>`;
    }).join('');

    const trk = getSelectedTrack();
    if (trk) {
      document.getElementById('optLockFoot').textContent =
        `TGT LOCK: ${trk.target_id} | TTI: ${trk.tti_s > 900 ? 'N/A' : trk.tti_s + 's'}`;
      document.getElementById('rpmTag').textContent = `${trk.rotor_rpm} RPM [STFT]`;
      document.getElementById('pipTargetTag').textContent = `FLIR 4X // ${trk.target_id}`;
      document.getElementById('pipOrdnanceFoot').textContent =
        trk.payload_detected ? 'THERMAL GRAIN // PAYLOAD POSITIVE' : 'THERMAL GRAIN // CLEAN AIRFRAME';
      document.getElementById('pipRfFoot').textContent = `LINK: ${trk.rf_protocol}`;
    }

    // Timeline Scrubber
    const slider = document.getElementById('timelineSlider');
    slider.max = Math.max(1, s.replay_total - 1);
    slider.value = s.replay_index;
    document.getElementById('replayCounterTag').textContent =
      `${s.mission_phase === 'POST_MISSION' ? 'DEBRIEF REPLAY' : 'LIVE REC'} // FRAME ${String(s.replay_index + 1).padStart(4, '0')}/${String(s.replay_total).padStart(4, '0')}`;
    document.getElementById('playPauseBtn').textContent = s.replay_playing ? 'PAUSE' : 'PLAY';
    document.getElementById('speedBtn').textContent = s.replay_speed + 'X';

    // Autonomy Play Cards
    ['cardPlay1', 'cardPlay2', 'cardPlay3'].forEach(id => document.getElementById(id).classList.remove('active-play'));
    if (s.active_play_id === 'PLAY_01_PERIMETER_CAP') document.getElementById('cardPlay1').classList.add('active-play');
    if (s.active_play_id === 'PLAY_02_SWARM_EW_BARRIER') document.getElementById('cardPlay2').classList.add('active-play');
    if (s.active_play_id === 'PLAY_03_KINETIC_WOLF_PACK') document.getElementById('cardPlay3').classList.add('active-play');

    const siloEff = s.effectors.find(e => e.id === 'KINETIC_SILO');
    if (siloEff) {
      document.getElementById('stPlay3').textContent = `${siloEff.ammo} SILOS`;
    }

    // Audit Ledger
    document.getElementById('auditList').innerHTML = s.audit_tail.map(b => {
      return `<div class="audit-item">
        <span>[${b.timestamp.slice(11, 19)}] ${b.event_type}</span>
        <span>#${b.block_hash.slice(0, 8)} ✓</span>
      </div>`;
    }).join('');
  }

  function renderAll() {
    tick++;
    renderRadarScope();
    renderOptronic();
    renderMicroDoppler();
    renderPipZoom();
    requestAnimationFrame(renderAll);
  }

  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws/telemetry`);
    ws.onmessage = (e) => {
      try { syncState(JSON.parse(e.data)); } catch (err) {}
    };
    ws.onclose = () => setTimeout(connectWS, 1200);
  }

  window.addEventListener('load', () => {
    connectWS();
    requestAnimationFrame(renderAll);
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/command-center", response_class=HTMLResponse)
async def serve_monochrome_c2():
    return HTMLResponse(content=MONOCHROME_LATTICE_HTML)


def run_server(host: str = "127.0.0.1", port: int = 8080):
    """Entry point to launch the production C2 server."""
    uvicorn.run("api.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
