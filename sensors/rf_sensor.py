"""
Radio Frequency (RF) Signal Detection, High-Resolution SDR Waterfall & Direction Finding (v6.0-MONOLITH).
Features:
- Multi-channel software-defined radio (SDR) capturing 128-bin FFT Power Spectral Density (PSD) waterfalls.
- CUDA-accelerated torch.fft.rfft baseband signal processing.
- Multi-band coverage across 2.4 GHz, 5.8 GHz, and 900 MHz / 433 MHz bands.
- Angle of Arrival (AoA) bearing lobing and frequency-hopping (FHSS) telemetry anomaly detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math
import random
import time
import numpy as np
import torch

from utils.logger import get_logger
from utils.cuda_runtime import get_cuda_engine

logger = get_logger("CUAS.RFSensor")


@dataclass
class RFEmission:
    frequency_mhz: float
    bandwidth_mhz: float
    rssi_dbm: float                         # Received Signal Strength (-105 to -20 dBm)
    snr_db: float                           # Signal-to-Noise Ratio
    protocol: str                           # "OcuSync-3", "ExpressLRS", "Crossfire", "WiFi-Telemetry", "RF-SILENT"
    mac_address_oui: str                    # Manufacturer ID or MAC prefix
    bearing_deg: float                      # Angle of Arrival (AoA)
    estimated_distance_m: float
    detected_timestamp: float
    is_frequency_hopping: bool = True
    hop_rate_hops_per_sec: float = 250.0    # FHSS hop rate
    waterfall_psd_bin_128: List[float] = field(default_factory=list) # 128-bin normalized dBm spectrum


class RFSensorSimulator:
    """
    High-Fidelity Multi-Channel SDR Receiver & Direction Finding (DF) Array.
    Generates 128-bin Power Spectral Density (PSD) waterfalls via CUDA torch.fft.rfft.
    """
    def __init__(
        self,
        scan_frequencies_mhz: Optional[List[float]] = None,
        noise_floor_dbm: float = -105.0
    ):
        self.scan_frequencies = scan_frequencies_mhz or [2412.0, 2437.0, 2462.0, 5745.0, 5805.0, 915.0]
        self.noise_floor_dbm = noise_floor_dbm
        self.last_scan_time = time.time()
        self.active_jammer_bands: List[str] = []
        self.cuda_engine = get_cuda_engine()

    def set_jammer_active(self, active_bands: List[str]):
        """Simulates jammer saturation of RF waterfall when countermeasures are engaged."""
        self.active_jammer_bands = active_bands

    def generate_waterfall_slice(self, center_freq_mhz: float = 2437.0, target_present: bool = True) -> List[float]:
        """
        Synthesizes a 128-bin FFT Power Spectral Density (PSD) array.
        Uses CUDA PyTorch torch.fft.rfft when available to process simulated I/Q baseband samples.
        """
        if self.cuda_engine.is_cuda:
            try:
                device = self.cuda_engine.device
                n_samples = 256
                # Base Gaussian thermal noise on GPU
                noise_real = torch.randn(n_samples, device=device) * 0.05
                noise_imag = torch.randn(n_samples, device=device) * 0.05

                # Check jammer saturation
                is_jammed = any("2.4" in b for b in self.active_jammer_bands) or any("JAM" in b for b in self.active_jammer_bands)

                if is_jammed:
                    # Broadband high-power jamming noise
                    jam_real = torch.randn(n_samples, device=device) * 8.0
                    jam_imag = torch.randn(n_samples, device=device) * 8.0
                    iq_samples = torch.complex(noise_real + jam_real, noise_imag + jam_imag)
                elif target_present:
                    t = torch.linspace(0, 1, n_samples, device=device)
                    # OFDM subcarriers (sinc-like pulse envelope)
                    f_carrier = 0.25  # Normalized frequency
                    carrier = torch.exp(2j * math.pi * f_carrier * t * 64) * 0.8
                    # Fast FHSS hopping spike
                    hop_freq = float(random.uniform(0.1, 0.45))
                    hop_tone = torch.exp(2j * math.pi * hop_freq * t * 128) * 1.5
                    iq_samples = torch.complex(noise_real, noise_imag) + carrier + hop_tone
                else:
                    iq_samples = torch.complex(noise_real, noise_imag)

                # Compute CUDA FFT
                fft_vals = torch.fft.rfft(iq_samples)
                power = torch.clamp(fft_vals.abs() ** 2, min=1e-12)
                psd_raw = 10.0 * torch.log10(power)

                # Interpolate to 128 bins
                psd_128 = torch.nn.functional.interpolate(
                    psd_raw.unsqueeze(0).unsqueeze(0),
                    size=128,
                    mode="linear",
                    align_corners=False
                ).squeeze().cpu().numpy()

                if is_jammed:
                    calibrated = np.clip(psd_128 - 25.0, -45.0, -25.0)
                elif target_present:
                    calibrated = np.clip(psd_128 - 50.0, self.noise_floor_dbm, -38.0)
                else:
                    calibrated = np.clip(psd_128 - 70.0, self.noise_floor_dbm, -92.0)

                return [round(float(v), 1) for v in calibrated]
            except Exception as e:
                logger.debug(f"CUDA FFT failed, using fallback: {e}")

        # NumPy CPU Fallback
        bins = np.random.normal(self.noise_floor_dbm, 2.0, 128)

        # If jammer active on this band, flood the spectrum with white noise
        if any("2.4" in b for b in self.active_jammer_bands) or any("JAM" in b for b in self.active_jammer_bands):
            jammer_noise = np.random.normal(-35.0, 4.0, 128)
            bins = np.maximum(bins, jammer_noise)
            return [round(float(v), 1) for v in bins]

        if target_present:
            # Inject OFDM video carrier (wide bump ~20 bins wide)
            carrier_center = 64
            for i in range(carrier_center - 15, carrier_center + 15):
                dist_from_c = abs(i - carrier_center)
                peak_db = -65.0 - (dist_from_c * 1.5) + random.uniform(-1.5, 1.5)
                bins[i] = max(bins[i], peak_db)

            # Inject fast FHSS telemetry spikes
            hop_bin_1 = random.randint(10, 45)
            hop_bin_2 = random.randint(85, 120)
            bins[hop_bin_1] = max(bins[hop_bin_1], -52.0 + random.uniform(-2.0, 2.0))
            bins[hop_bin_2] = max(bins[hop_bin_2], -48.0 + random.uniform(-2.0, 2.0))

        return [round(float(v), 1) for v in bins]

    def scan_spectrum(
        self,
        simulated_target_bearing: Optional[float] = None,
        distance_m: Optional[float] = None,
        target_protocol: str = "OcuSync-3",
        is_rf_silent: bool = False
    ) -> List[RFEmission]:
        """Executes multi-channel spectrum sweep and returns active detected emissions."""
        now = time.time()
        emissions = []

        if is_rf_silent or simulated_target_bearing is None or distance_m is None or distance_m > 4200.0:
            return emissions

        # Free-Space Path Loss (FSPL) model: FSPL(dB) = 20*log10(d) + 20*log10(f_MHz) - 27.55
        freq_mhz = 2437.0 if "2.4" in target_protocol or "Ocu" in target_protocol else 5805.0
        fspl = 20.0 * math.log10(max(10.0, distance_m)) + 20.0 * math.log10(freq_mhz) - 27.55
        tx_power_dbm = 26.0  # +26 dBm EIRP (~400 mW typical drone transmitter)
        rx_rssi = round(tx_power_dbm - fspl + random.uniform(-2.5, 2.5), 1)

        if rx_rssi > self.noise_floor_dbm:
            snr = round(rx_rssi - self.noise_floor_dbm, 1)
            bearing_error = random.gauss(0.0, 1.8)
            measured_bearing = round((simulated_target_bearing + bearing_error) % 360.0, 1)

            waterfall = self.generate_waterfall_slice(center_freq_mhz=freq_mhz, target_present=True)

            emission = RFEmission(
                frequency_mhz=freq_mhz,
                bandwidth_mhz=20.0 if "Ocu" in target_protocol else 10.0,
                rssi_dbm=rx_rssi,
                snr_db=snr,
                protocol=target_protocol,
                mac_address_oui="60:60:1F (DJI)" if "Ocu" in target_protocol else "00:1B:44 (ELRS)",
                bearing_deg=measured_bearing,
                estimated_distance_m=round(distance_m * random.uniform(0.92, 1.08), 1),
                detected_timestamp=now,
                is_frequency_hopping=True,
                hop_rate_hops_per_sec=250.0,
                waterfall_psd_bin_128=waterfall
            )
            emissions.append(emission)

        return emissions
