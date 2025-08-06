#!/usr/bin/env python3
"""
Test the video chat functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_ui import process_video_question

# Mock video results for testing
mock_results = {
    'video_info': {
        'duration': 30.5,
        'fps': 25.0,
        'frame_count': 763
    },
    'events': [
        {'timestamp': 5.2, 'description': 'Car detected', 'vehicles_detected': 3},
        {'timestamp': 12.8, 'description': 'Traffic light interaction', 'vehicles_detected': 2},
        {'timestamp': 20.1, 'description': 'Lane change detected', 'vehicles_detected': 4}
    ],
    'traffic_statistics': {
        'total_violations': 2,
        'total_vehicles': 15,
        'avg_vehicles_per_frame': 2.5,
        'safety_score': 7.2,
        'violation_types': ['speeding', 'following_too_close'],
        'frames_with_violations': 2
    },
    'qa_analysis': {
        'video_summary': 'Traffic video showing urban driving with multiple vehicles',
        'most_significant_event': {
            'description': 'Multiple vehicles detected during lane change maneuver',
            'timestamp': 20.1,
            'significance_score': 0.87
        },
        'camera_vehicle': {
            'type': 'car',
            'behavior': 'forward_driving',
            'confidence': 0.82
        },
        'temporal_sequence': [
            'Initial traffic detection at 5.2s',
            'Traffic light interaction at 12.8s', 
            'Lane change event at 20.1s'
        ]
    }
}

# Test different types of questions
test_questions = [
    "How many cars did you count?",
    "count cars",
    "number of vehicles",
    "What violations were detected?",
    "Is this safe driving?",
    "What type of vehicle is the camera on?",
    "What's the most important event?",
    "What happens at 15 seconds?",
    "Describe what you see",
    "Hello, tell me about this video"
]

print("🧪 Testing Video Chat Functionality")
print("=" * 50)

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. **Question:** {question}")
    answer = process_video_question(question, mock_results)
    print(f"   **Answer:** {answer[:200]}..." if len(answer) > 200 else f"   **Answer:** {answer}")
    print("-" * 30)

print("\n✅ Chat functionality test completed!")