"""Test Google AI integration with the face-body-swap pipeline"""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set API key in environment
os.environ['GOOGLE_AI_API_KEY'] = "AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU"

def test_google_ai_client():
    """Test Google AI client directly"""
    print("=" * 60)
    print("Testing Google AI Client Integration")
    print("=" * 60)
    
    try:
        from src.models.google_ai_client import GoogleAIClient, create_google_ai_client
        
        # Test client creation
        print("\n1. Creating Google AI client...")
        client = create_google_ai_client()
        
        if client is None:
            print("[FAIL] Could not create Google AI client")
            return False
        
        print("[OK] Google AI client created successfully")
        
        # Test image quality analysis
        print("\n2. Testing image quality analysis...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        analysis = client.analyze_image_quality(test_image)
        
        print(f"[OK] Quality analysis completed")
        print(f"    Overall score: {analysis.get('overall_score', 0):.2f}")
        print(f"    Face score: {analysis.get('face_score', 0):.2f}")
        print(f"    Body score: {analysis.get('body_score', 0):.2f}")
        
        # Test refinement suggestions
        print("\n3. Testing refinement suggestions...")
        suggestions = client.get_refinement_suggestions(test_image, ["face blending", "edge seams"])
        print(f"[OK] Refinement suggestions generated")
        print(f"    Priority areas: {suggestions.get('priority_areas', [])}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Google AI client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_control_integration():
    """Test Google AI integration in quality control"""
    print("\n" + "=" * 60)
    print("Testing Quality Control Integration")
    print("=" * 60)
    
    try:
        from src.pipeline.quality_control import QualityControl
        
        print("\n1. Creating QualityControl instance...")
        qc = QualityControl()
        
        if qc.google_ai_client is None:
            print("[WARNING] Google AI client not initialized in QualityControl")
            print("          This is OK if API key is not set or package is not installed")
        else:
            print("[OK] QualityControl created with Google AI client")
        
        # Test quality assessment
        print("\n2. Testing quality assessment...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        quality_metrics = qc.assess_quality(
            result_image=test_image,
            customer_faces=[],
            template_faces=[],
            template_analysis={}
        )
        
        print(f"[OK] Quality assessment completed")
        print(f"    Overall score: {quality_metrics.get('overall_score', 0):.2f}")
        print(f"    Face similarity: {quality_metrics.get('face_similarity', 0):.2f}")
        
        # Check if AI analysis was included
        if 'ai_analysis' in quality_metrics:
            print(f"[OK] Google AI analysis included in quality metrics")
            ai_analysis = quality_metrics['ai_analysis']
            print(f"    AI overall score: {ai_analysis.get('overall_score', 0):.2f}")
        else:
            print("[INFO] Google AI analysis not included (may be disabled or unavailable)")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Quality control integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("GOOGLE AI INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = {
        "Google AI Client": test_google_ai_client(),
        "Quality Control Integration": test_quality_control_integration()
    }
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("Google AI is successfully integrated into your system.")
    else:
        print("[WARNING] SOME TESTS FAILED - Check errors above")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

