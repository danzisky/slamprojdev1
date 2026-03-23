#!/usr/bin/env python3
"""Test script to verify association with simplified interface."""

import sys
from pathlib import Path

# Add slam_rover to path
slam_rover = Path(__file__).parent / "slam_rover"
sys.path.insert(0, str(slam_rover))

from localization.types import ChairObservation, ChairLandmark, PredictedLandmarkMeasurement
from localization.association import JointCompatibilityAssociator
import math


def test_basic_association():
    """Test basic association functionality."""
    print("Test 1: Basic Hungarian association")
    
    associator = JointCompatibilityAssociator(matching_method="hungarian")
    
    # Create matching observation and prediction
    landmark = ChairLandmark(landmark_id=0, x_m=5.0, y_m=5.0, label="chair")
    
    observation = ChairObservation(
        range_m=7.071,
        bearing_rad=math.radians(45.0),
        confidence=0.85,
        range_std_m=0.5,
        bearing_std_rad=math.radians(5.0)
    )
    
    prediction = PredictedLandmarkMeasurement(
        landmark=landmark,
        range_m=7.071,
        bearing_rad=math.radians(45.0)
    )
    
    result = associator.associate([observation], [prediction])
    assert len(result.matches) == 1, "Should match observation to prediction"
    assert result.matches[0] == (0, 0), "Should match obs 0 to pred 0"
    assert abs(result.likelihood - 1.0) < 1e-9, "Perfect match should have likelihood 1.0"
    print(f"  ✓ Matched successfully, likelihood: {result.likelihood:.6f}")


def test_icp_association():
    """Test ICP-style greedy association."""
    print("Test 2: ICP-style greedy association")
    
    associator = JointCompatibilityAssociator(matching_method="icp")
    
    landmark_a = ChairLandmark(landmark_id=0, x_m=5.0, y_m=5.0, label="chair")
    landmark_b = ChairLandmark(landmark_id=1, x_m=10.0, y_m=0.0, label="chair")
    
    obs1 = ChairObservation(
        range_m=5.0,
        bearing_rad=math.radians(0.0),
        confidence=0.9,
        range_std_m=0.3,
        bearing_std_rad=math.radians(3.0)
    )
    
    obs2 = ChairObservation(
        range_m=10.0,
        bearing_rad=math.radians(90.0),
        confidence=0.85,
        range_std_m=0.3,
        bearing_std_rad=math.radians(3.0)
    )
    
    pred1 = PredictedLandmarkMeasurement(
        landmark=landmark_a,
        range_m=5.0,
        bearing_rad=math.radians(0.0)
    )
    
    pred2 = PredictedLandmarkMeasurement(
        landmark=landmark_b,
        range_m=10.0,
        bearing_rad=math.radians(90.0)
    )
    
    result = associator.associate([obs1, obs2], [pred1, pred2])
    assert len(result.matches) == 2, "Should match both observations"
    assert abs(result.likelihood - 1.0) < 1e-9, "Perfect ICP match should have likelihood 1.0"
    print(f"  ✓ Matched 2 pairs, likelihood: {result.likelihood:.6f}")


def test_occlusion_penalty():
    """Test occlusion penalty for close landmarks in same direction."""
    print("Test 3: Occlusion penalty")
    
    associator = JointCompatibilityAssociator(matching_method="hungarian")
    
    # Create scenario: match to far landmark while leaving close unmatched
    landmark_far = ChairLandmark(landmark_id=0, x_m=10.0, y_m=0.0, label="chair")
    landmark_close = ChairLandmark(landmark_id=1, x_m=2.0, y_m=0.1, label="chair")
    
    # Observation that could match far landmark
    obs = ChairObservation(
        range_m=10.0,
        bearing_rad=math.radians(0.0),        confidence=0.9,
        range_std_m=0.3,
        bearing_std_rad=math.radians(3.0)
    )
    
    # Predictions: far and close
    pred_far = PredictedLandmarkMeasurement(
        landmark=landmark_far,
        range_m=10.0,
        bearing_rad=math.radians(0.0)
    )
    
    pred_close = PredictedLandmarkMeasurement(
        landmark=landmark_close,
        range_m=2.0,
        bearing_rad=math.radians(2.8)  # Close in bearing to far
    )
    
    result = associator.associate([obs], [pred_far, pred_close])
    
    # Should match obs to far but apply occlusion penalty (matched far while close unmatched)
    assert len(result.matches) == 1, "Should have 1 match"
    assert result.matches[0] == (0, 0), "Should match obs to far prediction"
    assert abs(result.likelihood - 0.5) < 1e-9, "Single occlusion violation should apply 0.5x penalty"
    
    # Likelihood should be reduced by occlusion penalty (0.5)
    print(f"  ✓ Match with occlusion penalty applied, likelihood: {result.likelihood:.6f}")


def test_count_mismatch_penalty():
    """Test penalty when observed count exceeds predicted visible count."""
    print("Test 4: Count mismatch penalty")
    
    associator = JointCompatibilityAssociator()
    
    landmark1 = ChairLandmark(landmark_id=0, x_m=5.0, y_m=0.0, label="chair")
    landmark2 = ChairLandmark(landmark_id=1, x_m=10.0, y_m=0.0, label="chair")
    
    obs1 = ChairObservation(
        range_m=5.0,
        bearing_rad=0.0,
        confidence=0.9,
        range_std_m=0.3,
        bearing_std_rad=math.radians(3.0)
    )

    obs2 = ChairObservation(
        range_m=5.2,
        bearing_rad=math.radians(1.0),
        confidence=0.9,
        range_std_m=0.3,
        bearing_std_rad=math.radians(3.0)
    )
    
    pred1 = PredictedLandmarkMeasurement(
        landmark=landmark1,
        range_m=5.0,
        bearing_rad=0.0
    )
    
    pred2 = PredictedLandmarkMeasurement(
        landmark=landmark2,
        range_m=10.0,
        bearing_rad=0.0
    )
    
    # 2 observations, 1 prediction should apply old-style 0.1^1 penalty.
    result = associator.associate([obs1, obs2], [pred1])
    assert abs(result.likelihood - 0.1) < 1e-9, "Count mismatch should apply 0.1^1 penalty"
    print(f"  ✓ Count mismatch penalty applied (0.1^1), likelihood: {result.likelihood:.6f}")


def test_empty_observations():
    """Test handling of empty observations or predictions."""
    print("Test 5: Empty observations/predictions handling")
    
    associator = JointCompatibilityAssociator()
    landmark = ChairLandmark(landmark_id=0, x_m=5.0, y_m=0.0, label="chair")
    pred = PredictedLandmarkMeasurement(landmark=landmark, range_m=5.0, bearing_rad=0.0)
    
    # Empty observations
    result = associator.associate([], [pred])
    assert result.likelihood == 1e-300, "Empty obs should return minimal likelihood"
    print(f"  ✓ Empty observations handled correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Association Simplification Tests")
    print("=" * 60)
    print()
    
    test_basic_association()
    print()
    
    test_icp_association()
    print()
    
    test_occlusion_penalty()
    print()
    
    test_count_mismatch_penalty()
    print()
    
    test_empty_observations()
    print()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
