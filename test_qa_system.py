#!/usr/bin/env python3
"""
Test script to validate the Q&A functionality of the enhanced VideoLLM system
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Test Q&A questions
test_questions = [
    "What is the most significant, unexpected event that occurs partway through the video?",
    "Immediately after the event identified in Question 1, how does the camera vehicle react?", 
    "Which of the following statements best summarizes the content of the video?",
    "Throughout the video, what type of vehicle is the camera mounted on?"
]

def create_mock_video_analysis():
    """Create mock video analysis data for testing"""
    return {
        'video_info': {
            'duration': 65.5,
            'fps': 30.0,
            'width': 1920,
            'height': 1080
        },
        'events': [
            {
                'timestamp': 15.2,
                'description': 'Traffic flowing normally with several cars in view. White sedan visible in left lane.',
                'frame_number': 0,
            },
            {
                'timestamp': 32.8,
                'description': 'A pedestrian suddenly runs across the road in front of the camera, crossing from right to left between vehicles.',
                'frame_number': 1,
            },
            {
                'timestamp': 34.1,
                'description': 'Camera vehicle continues forward in its lane after pedestrian crosses, maintaining steady speed and direction.',
                'frame_number': 2,
            }
        ],
        'traffic_statistics': {
            'total_violations': 1,
            'total_vehicles': 8,
            'safety_score': 6.5,
            'violation_types': ['pedestrian jaywalking']
        },
        'qa_analysis': {
            'video_summary': 'routine drive with unexpected pedestrian crossing',
            'most_significant_event': {
                'description': 'A pedestrian suddenly runs across the road in front of the camera, crossing from right to left between vehicles.',
                'timestamp': 32.8,
                'significance_score': 0.92
            },
            'camera_vehicle': {
                'type': 'car',
                'behavior': 'continues to drive forward in its lane',
                'confidence': 0.85
            },
            'temporal_sequence': [
                {
                    'order': 1,
                    'timestamp': 15.2,
                    'event_type': 'normal_traffic',
                    'description': 'Traffic flowing normally',
                    'is_key_event': False
                },
                {
                    'order': 2,
                    'timestamp': 32.8,
                    'event_type': 'pedestrian_crossing',
                    'description': 'Pedestrian runs across road',
                    'is_key_event': True
                },
                {
                    'order': 3,
                    'timestamp': 34.1,
                    'event_type': 'vehicle_reaction',
                    'description': 'Camera vehicle continues forward',
                    'is_key_event': True
                }
            ]
        }
    }

def test_chat_manager_qa():
    """Test the ChatManager Q&A functionality"""
    print("=== Testing ChatManager Q&A System ===\n")
    
    try:
        from src.models.chat_manager import ChatManager
        
        # Create mock analysis
        mock_analysis = create_mock_video_analysis()
        
        # Initialize chat manager
        print("Initializing ChatManager...")
        chat_manager = ChatManager()
        
        print(f"Chat manager initialized: {chat_manager.is_healthy}")
        print(f"Advanced analyzer available: {chat_manager.advanced_analyzer is not None}\n")
        
        # Test each question type
        for i, question in enumerate(test_questions, 1):
            print(f"--- Question {i} ---")
            print(f"Q: {question}")
            
            # Check if it's detected as Q&A style
            is_qa = chat_manager._is_qa_style_question(question.lower())
            print(f"Detected as Q&A style: {is_qa}")
            
            if is_qa:
                # Get Q&A response
                response = chat_manager._handle_qa_question(question, mock_analysis)
                if response:
                    print(f"A: {response}")
                else:
                    print("A: No specific Q&A response generated")
            else:
                print("A: Not recognized as Q&A question")
            
            print("\n" + "="*80 + "\n")
            
    except Exception as e:
        print(f"ChatManager test failed: {e}")
        import traceback
        traceback.print_exc()

def test_advanced_analyzer():
    """Test the AdvancedVideoAnalyzer directly"""
    print("=== Testing AdvancedVideoAnalyzer ===\n")
    
    try:
        from src.utils.advanced_video_analyzer import AdvancedVideoAnalyzer
        
        analyzer = AdvancedVideoAnalyzer()
        print("AdvancedVideoAnalyzer initialized successfully")
        
        # Create mock events for analysis
        mock_events = [
            {
                'timestamp': 15.2,
                'description': 'Traffic flowing normally with several cars visible in multiple lanes.',
                'violations_detected': [],
                'safety_score': 8
            },
            {
                'timestamp': 32.8, 
                'description': 'A pedestrian suddenly runs across the road in front of the camera, appearing unexpectedly between vehicles.',
                'violations_detected': ['jaywalking'],
                'safety_score': 3
            },
            {
                'timestamp': 34.1,
                'description': 'Camera vehicle maintains course and continues driving forward in its lane after the pedestrian incident.',
                'violations_detected': [],
                'safety_score': 7
            }
        ]
        
        mock_video_info = {'duration': 65.5, 'fps': 30.0}
        
        # Analyze video for Q&A
        print("\nPerforming comprehensive video analysis...")
        summary = analyzer.analyze_video_for_qa(mock_events, mock_video_info)
        
        print(f"Analysis completed:")
        print(f"- Main theme: {summary.main_theme}")
        print(f"- Key events detected: {len(summary.key_events)}")
        print(f"- Camera vehicle type: {summary.vehicle_analysis.camera_vehicle_type.value}")
        print(f"- Camera behavior: {summary.vehicle_analysis.camera_behavior}")
        print(f"- Most significant event: {summary.significance_ranking[0][0].description[:100]}..." if summary.significance_ranking else "None")
        
        # Test specific question answering
        print(f"\n--- Testing Question Answering ---")
        for i, question in enumerate(test_questions[:2], 1):  # Test first 2 questions
            print(f"\nQ{i}: {question}")
            response = analyzer.answer_specific_question(question, summary)
            print(f"A{i}: {response.get('answer', 'No answer')}")
            print(f"Confidence: {response.get('confidence', 0):.2f}")
            
    except Exception as e:
        print(f"AdvancedVideoAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()

def test_qa_detection_patterns():
    """Test Q&A pattern detection"""
    print("=== Testing Q&A Pattern Detection ===\n")
    
    test_messages = [
        "What is the most significant, unexpected event?",
        "How does the camera vehicle react?", 
        "Which statement best summarizes the video?",
        "What type of vehicle is the camera mounted on?",
        "Tell me about traffic violations",  # Should not be Q&A
        "How many cars are there?",  # Should not be Q&A
        "Immediately after the event, what happens?",
        "Throughout the video, what occurs?"
    ]
    
    try:
        from src.models.chat_manager import ChatManager
        
        chat_manager = ChatManager()
        
        for message in test_messages:
            is_qa = chat_manager._is_qa_style_question(message.lower())
            print(f"'{message}' -> Q&A Style: {is_qa}")
            
    except Exception as e:
        print(f"Pattern detection test failed: {e}")

def main():
    """Run all Q&A system tests"""
    print("🔍 VideoLLM Q&A System Testing\n")
    
    # Test pattern detection first
    test_qa_detection_patterns()
    print("\n" + "="*100 + "\n")
    
    # Test advanced analyzer
    test_advanced_analyzer()
    print("\n" + "="*100 + "\n")
    
    # Test chat manager integration
    test_chat_manager_qa()
    
    print("🎯 All Q&A tests completed!")

if __name__ == "__main__":
    main()