#!/usr/bin/env python3
"""Test delete gesture feature"""
import numpy as np
from fastapi.testclient import TestClient
from app.main import app, gesture_db
from app.ml.feature_extractor import landmarks_to_array

client = TestClient(app)
gesture_db.gestures.clear()


def create_landmarks():
    """Create dummy landmarks"""
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
        "left_hand": [{"x": 0.3, "y": 0.5, "z": 0.0} for _ in range(21)],
        "right_hand": [{"x": 0.7, "y": 0.5, "z": 0.0} for _ in range(21)],
    }


print("\n" + "=" * 60)
print("TEST DELETE GESTURE FEATURE")
print("=" * 60)

# 1. Record gesture 'halo'
print("\n[1] Record gesture: 'halo'")
sequence = [create_landmarks() for _ in range(10)]
gesture_db.add_gesture("halo", [landmarks_to_array(l) for l in sequence])
gesture_db._save_to_disk()
r = client.get("/gesture/list")
labels = r.json()["labels"]
print(f"✓ Recorded. Labels: {labels}")

# 2. Record gesture 'terima_kasih'
print("\n[2] Record gesture: 'terima_kasih'")
sequence2 = [create_landmarks() for _ in range(8)]
gesture_db.add_gesture("terima_kasih", [landmarks_to_array(l) for l in sequence2])
gesture_db._save_to_disk()
r = client.get("/gesture/list")
labels = r.json()["labels"]
print(f"✓ Recorded. Total labels: {len(labels)} ({labels})")

# 3. Delete first gesture
print("\n[3] Delete gesture: 'halo'")
r = client.delete("/gesture/halo")
if r.status_code == 200:
    print(f"✓ Response: {r.json()}")
else:
    print(f"✗ Error {r.status_code}: {r.json()}")

# 4. Verify deletion
print("\n[4] Verify deletion")
r = client.get("/gesture/list")
labels = r.json()["labels"]
print(f"✓ Remaining labels: {labels}")
assert "halo" not in labels, "Gesture 'halo' should be deleted"
assert "terima_kasih" in labels, "Gesture 'terima_kasih' should still exist"

# 5. Try delete non-existent
print("\n[5] Try delete non-existent gesture")
r = client.delete("/gesture/nonexistent")
print(f"✓ Status code: {r.status_code} (expected 404)")
assert r.status_code == 404, "Should return 404 for non-existent gesture"

print("\n" + "=" * 60)
print("✓ ALL DELETE TESTS PASSED!")
print("=" * 60 + "\n")
