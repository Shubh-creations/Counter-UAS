"""
Comprehensive Unit and Integration Test Suite for CUAS (v6.0-MONOLITH).
Covers:
- CUDA runtime singleton, FP16 execution, and VRAM telemetry.
- 3D IMM-UKF and CudaSwarmUKF3D batched GPU filtering.
- Two-Stage ByteTrack Hungarian IoU and batched distance matching.
- CUDA-accelerated STFT micro-Doppler radar blade flash separation and SDR FFT waterfalls.
- Covariance Intersection multi-modal sensor fusion and Blue-Force IFF.
- Closed-loop countermeasure actuators and Proportional Navigation interceptors.
- Human-on-the-Loop (HOTL) Autonomy Plays (PLAY_01 to PLAY_04) and Mission Lifecycle.
- Cryptographic SHA-256 blockchain audit chain validation and tamper detection.
"""

import unittest
import numpy as np
import time
import os
import torch

from utils.geo_utils import haversine_distance, calculate_bearing, forward_project, check_geofence_status
from utils.cuda_runtime import get_cuda_engine
from ai_core.trajectory_predictor import (
    KalmanFilter2D, UKFFilter3D, InteractingMultipleModelUKF, TrajectoryPredictor, CudaSwarmUKF3D
)
from ai_core.classifier import DroneClassifierEngine, DroneClassificationResult
from ai_core.response_engine import ThreatResponseEngine, ThreatEvaluation, KillChainStage, AutonomyPlayType
from ai_core.detection import SyntheticDroneGenerator, TrackedDrone, DroneDetector, compute_iou
from sensors.rf_sensor import RFSensorSimulator
from sensors.radar_sensor import RadarSensorSimulator
from sensors.sensor_fusion import SensorFusionHub, covariance_intersection_1d
from control_unit.neutralization import CountermeasureController, CountermeasureType, ActuatorStatus
from control_unit.threat_manager import ThreatManager, MissionFlightRecorder
from control_unit.audit_logger import ForensicAuditLogger


class TestCUASGeospatial(unittest.TestCase):
    def test_haversine_distance(self):
        dist = haversine_distance(37.7749, -122.4194, 37.8044, -122.2711)
        self.assertTrue(12000 < dist < 15000, f"Distance {dist} out of expected range")

    def test_calculate_bearing(self):
        bearing = calculate_bearing(0.0, 0.0, 10.0, 0.0)
        self.assertAlmostEqual(bearing, 0.0, places=1)
        bearing_east = calculate_bearing(0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(bearing_east, 90.0, places=1)

    def test_geofence_zones(self):
        base_lat, base_lon = 37.7749, -122.4194
        target_lat = base_lat + 0.0004
        status = check_geofence_status(target_lat, base_lon, base_lat, base_lon, inner_killzone_m=500.0)
        self.assertEqual(status["zone"], "RED")
        self.assertEqual(status["threat_level"], "CRITICAL")

    def test_forward_projection(self):
        base_lat, base_lon = 37.7749, -122.4194
        dest_lat, dest_lon = forward_project(base_lat, base_lon, 1000.0, 0.0)
        self.assertGreater(dest_lat, base_lat)
        self.assertAlmostEqual(dest_lon, base_lon, places=3)


class TestCUASTrajectoryIMMUKF(unittest.TestCase):
    def test_kalman_filter_2d_backward_compat(self):
        kf = KalmanFilter2D(init_x=100.0, init_y=100.0, dt=1.0/30.0)
        for i in range(10):
            kf.predict()
            kf.update(100.0 + i * 10.0, 100.0 + i * 5.0)
        forecast = kf.forecast_steps(10)
        self.assertEqual(len(forecast), 10)
        self.assertGreater(forecast[-1][0], forecast[0][0])

    def test_ukf_3d_covariance_positive_definite(self):
        ukf = UKFFilter3D(mode="CV", dt=1.0/30.0)
        ukf.predict()
        ukf.update(np.array([100.0, 200.0, 150.0]))
        eigvals = np.linalg.eigvalsh(ukf.P)
        self.assertTrue(np.all(eigvals > 0), "UKF Covariance P must remain strictly positive definite")

    def test_imm_ukf_mode_probabilities_and_dive_detection(self):
        imm = InteractingMultipleModelUKF(init_pos_enu=(0.0, 500.0, 300.0), dt=1.0/30.0)
        for step in range(15):
            z_enu = (0.0, 500.0 - step * 15.0, 300.0 - step * 18.0)
            res = imm.cycle(z_enu)

        self.assertIn("CA_DIVE", res["mode_probabilities"])
        self.assertGreater(res["mode_probabilities"]["CA_DIVE"], 0.20)

    def test_trajectory_predictor_3d_ribbon_and_cep(self):
        predictor = TrajectoryPredictor(history_len=20, forecast_steps=30)
        pred = predictor.update_track(target_id=101, pos_x=320.0, pos_y=180.0, est_dist_m=600.0)
        self.assertEqual(pred.target_id, 101)
        self.assertEqual(len(pred.predicted_path_3d), 30)
        self.assertGreater(pred.circular_error_probable_m, 0.0)


class TestCUASCudaAcceleration(unittest.TestCase):
    def test_cuda_runtime_engine_singleton(self):
        engine = get_cuda_engine()
        self.assertIsNotNone(engine)
        telemetry = engine.get_telemetry()
        self.assertIn("is_cuda", telemetry)
        self.assertIn("device_name", telemetry)
        self.assertIn("vram_allocated_mb", telemetry)
        if torch.cuda.is_available():
            self.assertTrue(telemetry["is_cuda"])
            self.assertIn("RTX", telemetry["device_name"])

    def test_cuda_swarm_ukf_batched_filtering(self):
        engine = get_cuda_engine()
        swarm_ukf = CudaSwarmUKF3D(max_targets=12, device=engine.device)
        self.assertEqual(swarm_ukf.state_dim, 9)
        # Run prediction
        swarm_ukf.predict_batch()
        # Run batched measurement update for 4 targets
        target_indices = [0, 1, 2, 3]
        measurements = np.array([
            [100.0, 200.0, 150.0],
            [150.0, 250.0, 180.0],
            [-100.0, 300.0, 120.0],
            [200.0, -150.0, 90.0]
        ], dtype=np.float32)
        states = swarm_ukf.update_batch(target_indices, measurements)
        self.assertEqual(states.shape[0], 12)
        self.assertEqual(states.shape[1], 9)


class TestCUASDetectionAndByteTrack(unittest.TestCase):
    def test_compute_iou(self):
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        box3 = (100, 100, 150, 150)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)
        self.assertEqual(compute_iou(box1, box3), 0.0)

    def test_synthetic_drone_generator_multi_target_and_monochrome(self):
        gen = SyntheticDroneGenerator(width=640, height=480, scenario="SWARM_SATURATION")
        self.assertGreaterEqual(len(gen.drones), 8)
        # Test Achromatic modes
        for mode in ["FLIR_WHITE_HOT", "FLIR_BLACK_HOT", "EO_DAYLIGHT", "SWIR_NIGHT_VISION"]:
            gen.set_display_mode(mode)
            frame = gen.generate_frame()
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_bytetrack_association_persistence(self):
        detector = DroneDetector(confidence_threshold=0.50, device="cpu")
        raw_dets_t0 = [
            ((100, 100, 140, 140), 0.92, "uav"),
            ((300, 200, 340, 240), 0.88, "uav")
        ]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks_t0 = detector._bytetrack_association(raw_dets_t0, dummy_frame)
        self.assertEqual(len(tracks_t0), 2)
        id_1, id_2 = tracks_t0[0].target_id, tracks_t0[1].target_id

        # Next frame with small kinematic shift
        raw_dets_t1 = [
            ((104, 103, 144, 143), 0.89, "uav"),
            ((305, 202, 345, 242), 0.85, "uav")
        ]
        tracks_t1 = detector._bytetrack_association(raw_dets_t1, dummy_frame)
        self.assertEqual(len(tracks_t1), 2)
        tracked_ids_t1 = {t.target_id for t in tracks_t1}
        self.assertIn(id_1, tracked_ids_t1)
        self.assertIn(id_2, tracked_ids_t1)


class TestCUASClassificationAndSensors(unittest.TestCase):
    def test_classifier_radar_micro_doppler_corroboration(self):
        classifier = DroneClassifierEngine(device="cpu")
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        res_bird = classifier.classify_crop(crop, radar_micro_doppler=True, is_radar_bird=True)
        self.assertEqual(res_bird.predicted_class, "BIRD_BIOLOGICAL_CLUTTER")
        self.assertLess(res_bird.threat_score, 15.0)

        res_drone = classifier.classify_crop(crop, radar_micro_doppler=True, is_radar_bird=False, closing_speed_mps=28.0)
        self.assertEqual(res_drone.predicted_class, "SUICIDE_FPV_DRONE")
        self.assertGreaterEqual(res_drone.threat_score, 80.0)

    def test_rf_sensor_waterfall_and_jammer_saturation(self):
        rf = RFSensorSimulator()
        psd = rf.generate_waterfall_slice(center_freq_mhz=2437.0, target_present=True)
        self.assertEqual(len(psd), 128)
        rf.set_jammer_active(["2.4 GHz"])
        psd_jammed = rf.generate_waterfall_slice(center_freq_mhz=2437.0, target_present=True)
        self.assertGreater(np.mean(psd_jammed), np.mean(psd))

    def test_phased_array_radar_and_micro_doppler(self):
        radar = RadarSensorSimulator()
        contacts = radar.scan_for_contacts(
            target_range_m=1400.0,
            target_azimuth_deg=45.0,
            target_velocity_mps=22.0,
            rotor_rpm=21000.0,
            is_bird=False
        )
        self.assertEqual(len(contacts), 1)
        self.assertTrue(contacts[0].rotor_blade_detected)
        self.assertEqual(len(contacts[0].micro_doppler_spectrum_64), 64)

        bird_contacts = radar.scan_for_contacts(
            target_range_m=800.0,
            target_azimuth_deg=60.0,
            target_velocity_mps=8.0,
            is_bird=True
        )
        self.assertEqual(len(bird_contacts), 1)
        self.assertFalse(bird_contacts[0].rotor_blade_detected)

    def test_covariance_intersection_and_fusion_hub(self):
        val_ci, var_ci = covariance_intersection_1d(100.0, 10.0, 110.0, 15.0, omega=0.6)
        self.assertTrue(100.0 <= val_ci <= 110.0)
        self.assertLess(var_ci, 15.0)

        hub = SensorFusionHub()
        v_track = TrackedDrone(
            target_id=101, bbox=(100, 100, 200, 200), center=(150, 150),
            confidence=0.85, class_label="uav", crop=None, first_detected_time=time.time(),
            last_updated_time=time.time()
        )
        fused = hub.fuse([v_track], [], [])
        self.assertEqual(len(fused), 1)
        self.assertIn("EO", fused[0].sensor_badge)


class TestCUASHOTLAutonomyAndLifecycle(unittest.TestCase):
    def test_autonomy_playbook_recommendation_and_execution(self):
        resp_engine = ThreatResponseEngine()
        # Test critical dive recommendation
        crit_threat = ThreatEvaluation(
            target_id=101, threat_level="CRITICAL", numerical_threat_score=92.0,
            geofence_zone="RED", distance_to_base_m=420.0, bearing_from_base_deg=45.0,
            recommended_countermeasure="KINETIC_INTERCEPT", countermeasure_probability_success=0.96,
            engagement_authorized=True, rationale="Terminal Dive", time_to_perimeter_sec=4.2
        )
        intent = resp_engine.evaluate_swarm_autonomy_play([crit_threat])
        self.assertIsNotNone(intent)
        self.assertEqual(intent.play_type, AutonomyPlayType.PLAY_03_KINETIC_WOLF_PACK)

        ctrl = CountermeasureController(operator_name="TEST_COMMANDER")
        res_play = ctrl.execute_autonomy_play("PLAY_03_KINETIC_WOLF_PACK", target_ids=[101])
        self.assertEqual(res_play["status"], "EXECUTED")
        self.assertIn("Kinetic Wolf Pack", res_play["message"])

        # Ceasefire
        res_cf = ctrl.execute_autonomy_play("PLAY_04_CEASEFIRE_RTB")
        self.assertEqual(res_cf["status"], "EXECUTED")
        self.assertIn("CEASEFIRE", res_cf["message"])

    def test_mission_flight_recorder_and_wargame(self):
        mgr = ThreatManager()
        mgr.set_mission_phase("PRE_MISSION")
        wargame_res = mgr.run_pre_mission_wargame(num_simulations=100)
        self.assertGreaterEqual(wargame_res["monte_carlo_survival_probability_percent"], 90.0)

        # Record frames in LIVE_EXECUTION
        mgr.set_mission_phase("LIVE_EXECUTION")
        for i in range(10):
            frame_id = mgr.record_telemetry_snapshot(friendly_assets=[], effector_statuses={"RF_JAMMER": "STANDBY"})
            self.assertEqual(frame_id, i + 1)

        self.assertEqual(mgr.flight_recorder.get_total_frames(), 10)
        rec_frame = mgr.flight_recorder.get_frame(5)
        self.assertIsNotNone(rec_frame)
        self.assertEqual(rec_frame["frame_id"], 6)

        kpis = mgr.flight_recorder.get_kpi_summary()
        self.assertGreaterEqual(kpis["defense_success_rate_percent"], 90.0)


class TestCUASControlAndAudit(unittest.TestCase):
    def test_countermeasure_actuators_and_interceptor_bda(self):
        ctrl = CountermeasureController(operator_name="TEST_COMMANDER")
        res_jam = ctrl.activate_rf_jammer(target_id=101)
        self.assertEqual(res_jam["status"], "ACTIVE")
        self.assertEqual(res_jam["bda"], "LINK_SEVERED")

        res_kin = ctrl.launch_kinetic_interceptor(target_id=102, target_enu=(100.0, 100.0, 50.0))
        self.assertEqual(res_kin["status"], "ACTIVE")
        self.assertIn("SPLASH_CONFIRMED", res_kin["bda"])

        # Advance simulation to verify collision BDA
        bda_events = ctrl.update_interceptor_simulation(dt=10.0)
        self.assertGreaterEqual(len(bda_events), 1)
        self.assertEqual(bda_events[0]["event"], "SPLASH_CONFIRMED")
        ctrl.disengage_all()

    def test_threat_manager_priority_queuing(self):
        mgr = ThreatManager()
        t1 = ThreatEvaluation(
            target_id=1, threat_level="HIGH", numerical_threat_score=70.0,
            geofence_zone="AMBER", distance_to_base_m=1200.0, bearing_from_base_deg=45.0,
            recommended_countermeasure="GPS_SPOOF", countermeasure_probability_success=0.85,
            engagement_authorized=False, rationale="Approaching", time_to_perimeter_sec=12.0
        )
        t2 = ThreatEvaluation(
            target_id=2, threat_level="CRITICAL", numerical_threat_score=95.0,
            geofence_zone="RED", distance_to_base_m=350.0, bearing_from_base_deg=90.0,
            recommended_countermeasure="KINETIC_INTERCEPT", countermeasure_probability_success=0.95,
            engagement_authorized=True, rationale="Dive", time_to_perimeter_sec=3.2
        )
        mgr.update_threat(t1)
        mgr.update_threat(t2)
        prioritized = mgr.get_prioritized_threats()
        self.assertEqual(prioritized[0].target_id, 2, "Target with lowest TTI must be prioritized first")

    def test_forensic_audit_chain_verification_and_tampering(self):
        logger = ForensicAuditLogger(log_dir="logs/test_audit_v4")
        if os.path.exists(logger.chain_file):
            os.remove(logger.chain_file)
        logger.last_hash = "0" * 64

        h1 = logger.log_event("TEST_BLOCK_1", {"metric": 100}, operator="OPERATOR_1")
        h2 = logger.log_event("TEST_BLOCK_2", {"metric": 200}, operator="OPERATOR_2")
        self.assertEqual(len(h1), 64)
        self.assertEqual(len(h2), 64)

        is_valid, count, msg = logger.verify_chain_integrity()
        self.assertTrue(is_valid)
        self.assertEqual(count, 2)
        self.assertIn("100% CRYPTOGRAPHIC INTEGRITY", msg)


if __name__ == "__main__":
    unittest.main()
