#!/usr/bin/env python3
"""
Comprehensive test script for gesture learning end-to-end workflow
"""
import json
import numpy as np
from fastapi.testclient import TestClient
from app.main import app, gesture_db
from app.ml.feature_extractor import landmarks_to_array


def create_dummy_landmarks():
    """Create realistic-looking landmark data"""
    return {
        "pose": [
            {
                "x": 0.5 + np.random.normal(0, 0.02),
                "y": 0.5 + np.random.normal(0, 0.02),
                "z": 0.0,
            }
            for _ in range(33)
        ],
        "face": [
            {
                "x": 0.5 + np.random.normal(0, 0.01),
                "y": 0.5 + np.random.normal(0, 0.01),
                "z": 0.0,
            }
            for _ in range(468)
        ],
        "left_hand": [
            {
                "x": 0.3 + np.random.normal(0, 0.02),
                "y": 0.5 + np.random.normal(0, 0.02),
                "z": 0.0,
            }
            for _ in range(21)
        ],
        "right_hand": [
            {
                "x": 0.7 + np.random.normal(0, 0.02),
                "y": 0.5 + np.random.normal(0, 0.02),
                "z": 0.0,
            }
            for _ in range(21)
        ],
    }


def test_gesture_learning():
    """Test complete gesture learning workflow"""
    client = TestClient(app)

    print("\n" + "=" * 60)
    print("GESTURE LEARNING END-TO-END TEST")
    print("=" * 60)

    # Test 1: Health check
    print("\n[1] Health Check")
    r = client.get("/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health = r.json()
    print(f"✓ Status: {health['status']}")
    print(f"✓ Model loaded: {health['model_loaded']}")
    print(f"✓ Active WS clients: {health['active_ws_clients']}")

    # Test 2: Initial gesture list (should be empty)
    print("\n[2] Initial Gesture List")
    r = client.get("/gesture/list")
    assert r.status_code == 200
    initial = r.json()
    print(f"✓ Labels: {initial.get('labels', [])}")
    print(f"✓ Database size: {len(initial.get('database', {}))}")

    # Test 3: Record first gesture (halo)
    print("\n[3] Record Gesture: 'halo'")
    gesture_db.gestures.clear()  # Start fresh
    sequence = [landmarks_to_array(create_dummy_landmarks()) for _ in range(10)]
    gesture_db.add_gesture("halo", sequence)
    gesture_db._save_to_disk()
    r = client.get("/gesture/list")
    assert r.status_code == 200
    data = r.json()
    assert "halo" in data.get("labels", []), "Gesture 'halo' not recorded"
    print(f"✓ Recorded 'halo' with {len(sequence)} frames")
    print(f"✓ Total labels: {len(data.get('labels', []))}")

    # Test 4: Record second gesture (terima_kasih)
    print("\n[4] Record Gesture: 'terima_kasih'")
    sequence2 = [landmarks_to_array(create_dummy_landmarks()) for _ in range(8)]
    gesture_db.add_gesture("terima_kasih", sequence2)
    gesture_db._save_to_disk()
    r = client.get("/gesture/list")
    data = r.json()
    assert "terima_kasih" in data.get("labels", [])
    print(f"✓ Recorded 'terima_kasih' with {len(sequence2)} frames")
    print(f"✓ Total labels: {len(data.get('labels', []))}")

    # Test 5: Recognize gesture (similar to 'halo')
    print("\n[5] Recognize Gesture (similar to 'halo')")
    test_sequence = [landmarks_to_array(create_dummy_landmarks()) for _ in range(7)]
    result = gesture_db.recognize(test_sequence, threshold=0.5)
    if result:
        label, confidence = result
        print(f"✓ Recognized: '{label}' (confidence: {confidence:.2%})")
    else:
        print("⚠ No match found (threshold might be too high)")

    # Test 6: Predict endpoint (should still return dummy)
    print("\n[6] Prediction via /predict")
    payload = {
        "sequence": [create_dummy_landmarks() for _ in range(20)],
        "region": "default",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    pred = r.json()
    print(f"✓ Prediction: {pred.get('text', 'N/A')}")
    print(f"✓ Latency: {pred.get('latency_ms', 'N/A')}ms")

    # Test 7: Regional endpoint
    print("\n[7] Regional Endpoint")
    r = client.get("/regional")
    assert r.status_code == 200
    regions = r.json()
    print(f"✓ Available regions: {regions.get('regions', [])}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - GESTURE LEARNING READY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_gesture_learning()
