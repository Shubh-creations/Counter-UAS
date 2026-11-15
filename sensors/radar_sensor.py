"""
3D Phased-Array Pulse-Doppler Radar & Micro-Doppler Blade Flash Simulator (v6.0-MONOLITH).
Features:
- Electronic-scan active electronically steered array (AESA) operating at 20 Hz track-while-scan.
- CUDA-accelerated torch.stft Micro-Doppler rotor blade harmonic spectrogram processing.
- Symmetric harmonic separation for drone identification vs low-frequency biological clutter.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math
import random
import time
import numpy as np
import torch

from utils.logger import get_logger
from utils.cuda_runtime import get_cuda_engine

logger = get_logger("CUAS.Radar")


@dataclass
class RadarContact:
    contact_id: int
    azimuth_deg: float                         # 0 to 360 degrees
    elevation_deg: float                       # -5 to +60 degrees
    range_m: float                             # Radial 3D distance
    radial_velocity_mps: float                 # + approaching, - departing
    rcs_dbsm: float                            # Radar Cross Section in dBsm (-20 to -10 typical for small UAV)
    rotor_blade_detected: bool                 # Micro-Doppler modulation detected
    snr_db: float
    timestamp: float
    rotor_rpm_estimate: float = 18000.0        # Estimated rotor speed
    blade_count: int = 4                       # Quadrotor (4x2=8 blades)
    micro_doppler_spectrum_64: List[float] = field(default_factory=list) # 64-bin Doppler spectrum
    elevation_m: float = 120.0                 # Target altitude above radar plane
    track_quality: float = 0.95                # 0.0 to 1.0 confidence


class RadarSensorSimulator:
    """
    3D Phased-Array Pulse-Doppler Tactical Radar Simulator.
    Features 20 Hz track-while-scan update rate and CUDA-accelerated micro-Doppler STFT spectral physics.
    """
    def __init__(self, sweep_rate_hz: float = 2.0, max_range_m: float = 5000.0):
        self.sweep_rate = sweep_rate_hz
        self.max_range_m = max_range_m
        self.current_azimuth = 0.0
        self.last_update = time.time()
        self.next_contact_id = 201
        self.cuda_engine = get_cuda_engine()

    def update_sweep(self) -> float:
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        self.current_azimuth = (self.current_azimuth + (self.sweep_rate * 360.0 * dt)) % 360.0
        return self.current_azimuth

    def generate_micro_doppler_slice(
        self,
        base_velocity_mps: float = 18.0,
        rotor_rpm: float = 18000.0,
        is_bird: bool = False
    ) -> List[float]:
        """
        Synthesizes a 64-bin micro-Doppler velocity spectrogram slice.
        Uses CUDA PyTorch STFT when available:
        s(t) = A_body * exp(j 2*pi*f_d*t) + sum(A_blade * cos(2*pi*f_r*t)*exp(j 2*pi*f_d*t)) + noise
        Quadrotors produce distinct symmetric sideband spikes (flashes) around the central body return.
        Birds produce asymmetric low-frequency bulk motion.
        """
        if self.cuda_engine.is_cuda:
            try:
                device = self.cuda_engine.device
                n_samples = 512
                t = torch.linspace(0, 0.05, n_samples, device=device)  # 50 ms sample at 10 kHz
                carrier_doppler = base_velocity_mps * 66.7  # X-band Doppler constant ~66.7 Hz per m/s

                if is_bird:
                    # Low frequency flapping (2-8 Hz)
                    flap_freq = 6.0
                    phase = 2.0 * math.pi * carrier_doppler * t + 0.8 * torch.sin(2.0 * math.pi * flap_freq * t)
                    sig_real = torch.cos(phase) + torch.randn(n_samples, device=device) * 0.35
                    sig_imag = torch.sin(phase) + torch.randn(n_samples, device=device) * 0.35
                else:
                    # Rotor blade tips high-frequency modulation
                    f_blade = (rotor_rpm / 60.0) * 4.0  # Rotor blade passing frequency
                    flash_mod = 0.65 * torch.cos(2.0 * math.pi * f_blade * t)
                    phase = 2.0 * math.pi * carrier_doppler * t + flash_mod
                    sig_real = torch.cos(phase) + torch.randn(n_samples, device=device) * 0.15
                    sig_imag = torch.sin(phase) + torch.randn(n_samples, device=device) * 0.15

                iq_signal = torch.complex(sig_real, sig_imag)
                # Compute STFT
                window = torch.hann_window(128, device=device)
                stft_res = torch.stft(
                    iq_signal,
                    n_fft=128,
                    hop_length=32,
                    win_length=128,
                    window=window,
                    return_complex=True,
                    center=True
                )
                spec_power = stft_res.abs().mean(dim=-1)  # Mean power across time frames
                # Downsample/interpolate to 64 bins
                spec_64 = torch.nn.functional.interpolate(
                    spec_power.unsqueeze(0).unsqueeze(0),
                    size=64,
                    mode="linear",
                    align_corners=False
                ).squeeze().cpu().numpy()

                # Convert to dB scale relative to peak
                peak = max(float(spec_64.max()), 1e-6)
                db_bins = 20.0 * np.log10(np.maximum(spec_64 / peak, 1e-3))
                # Scale between -45 dBm and +15 dBm
                scaled_bins = np.clip(db_bins + 15.0, -45.0, 18.0)
                return [round(float(v), 1) for v in scaled_bins]
            except Exception as e:
                logger.debug(f"CUDA STFT failed, using fallback: {e}")

        # NumPy CPU Fallback
        bins = np.random.normal(-45.0, 2.5, 64)
        center_bin = 32

        # Central fuselage/body radar return peak
        v_offset = int(max(-10, min(10, base_velocity_mps * 0.5)))
        body_bin = center_bin + v_offset
        if 0 <= body_bin < 64:
            bins[body_bin] = max(bins[body_bin], 15.0 + random.uniform(-2.0, 2.0))
            if body_bin - 1 >= 0: bins[body_bin - 1] = max(bins[body_bin - 1], 6.0)
            if body_bin + 1 < 64: bins[body_bin + 1] = max(bins[body_bin + 1], 6.0)

        if is_bird:
            # Low-frequency broad flapping modulation (< 10 Hz)
            for i in range(body_bin - 3, body_bin + 4):
                if 0 <= i < 64:
                    bins[i] = max(bins[i], -10.0 + random.uniform(-3.0, 3.0))
        else:
            # High-RPM rotor blade tip flashes (symmetric harmonics extending out to ±20 bins)
            tip_speed_factor = int(min(22, (rotor_rpm / 1000.0) * 0.9))
            for arm in [-tip_speed_factor, tip_speed_factor]:
                flash_bin = body_bin + arm
                if 0 <= flash_bin < 64:
                    bins[flash_bin] = max(bins[flash_bin], 4.0 + random.uniform(-1.5, 1.5))
                # First harmonic sideband
                arm_half = arm // 2
                half_bin = body_bin + arm_half
                if 0 <= half_bin < 64:
                    bins[half_bin] = max(bins[half_bin], -8.0 + random.uniform(-2.0, 2.0))

        return [round(float(v), 1) for v in bins]

    def scan_for_contacts(
        self,
        target_range_m: Optional[float] = None,
        target_azimuth_deg: Optional[float] = None,
        target_velocity_mps: Optional[float] = None,
        target_altitude_m: Optional[float] = None,
        rotor_rpm: float = 18000.0,
        is_bird: bool = False
    ) -> List[RadarContact]:
        """Scans radar surveillance sector and returns tracked 3D radar returns."""
        self.update_sweep()
        contacts = []

        if target_range_m is not None and target_azimuth_deg is not None:
            if target_range_m <= self.max_range_m:
                az_err = random.gauss(0.0, 0.4)
                range_err = random.gauss(0.0, 2.5)
                vel_mps = target_velocity_mps if target_velocity_mps is not None else random.uniform(14.0, 28.0)
                alt_m = target_altitude_m if target_altitude_m is not None else random.uniform(80.0, 240.0)

                elev_rad = math.atan2(alt_m, max(1.0, target_range_m))
                elev_deg = math.degrees(elev_rad)

                # Radar Cross Section in dBsm (small carbon-frame drone ~ -16 dBsm)
                rcs = -24.0 if is_bird else round(random.uniform(-18.0, -12.0), 1)

                # Radar equation SNR ~ RCS / R^4
                snr = max(8.0, min(36.0, 32.0 - 20.0 * math.log10(max(100.0, target_range_m) / 1000.0)))

                micro_doppler = self.generate_micro_doppler_slice(
                    base_velocity_mps=vel_mps,
                    rotor_rpm=rotor_rpm,
                    is_bird=is_bird
                )

                contact = RadarContact(
                    contact_id=self.next_contact_id,
                    azimuth_deg=round((target_azimuth_deg + az_err) % 360.0, 1),
                    elevation_deg=round(elev_deg, 1),
                    range_m=round(target_range_m + range_err, 1),
                    radial_velocity_mps=round(vel_mps, 1),
                    rcs_dbsm=rcs,
                    rotor_blade_detected=(not is_bird),
                    snr_db=round(snr, 1),
                    timestamp=time.time(),
                    rotor_rpm_estimate=rotor_rpm if not is_bird else 0.0,
                    blade_count=4 if not is_bird else 0,
                    micro_doppler_spectrum_64=micro_doppler,
                    elevation_m=round(alt_m, 1),
                    track_quality=round(min(1.0, snr / 25.0), 2)
                )
                contacts.append(contact)

        return contacts
