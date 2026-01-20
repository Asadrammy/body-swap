"""Full integration test: Test Google AI with actual pipeline"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up environment
os.environ['GOOGLE_AI_API_KEY'] = "AIzaSyCioMnUARXoWlLmQZDKS-wrUZhoulS6hPU"

def test_full_pipeline_integration():
    """Test Google AI integration in the full pipeline"""
    print("=" * 60)
    print("Full Pipeline Integration Test")
    print("=" * 60)
    
    try:
        from src.pipeline.quality_control import QualityControl
        from src.models.google_ai_client import create_google_ai_client
        
        print("\n1. Initializing QualityControl...")
        qc = QualityControl()
        
        if qc.google_ai_client:
            print("   [OK] Google AI client initialized")
        else:
            print("   [WARNING] Google AI client not available")
            return False
        
        print("\n2. Creating test image...")
        # Create a realistic test image (not random noise)
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        # Add some structure
        test_image[100:400, 100:400] = [200, 180, 160]  # Face-like region
        test_image[400:500, 150:350] = [150, 120, 100]  # Body-like region
        
        print("   [OK] Test image created")
        
        print("\n3. Running quality assessment with Google AI...")
        quality_metrics = qc.assess_quality(
            result_image=test_image,
            customer_faces=[],
            template_faces=[],
            template_analysis={}
        )
        
        print("   [OK] Quality assessment completed")
        print(f"   Overall score: {quality_metrics.get('overall_score', 0):.2f}")
        print(f"   Face similarity: {quality_metrics.get('face_similarity', 0):.2f}")
        
        # Check AI integration
        if 'ai_analysis' in quality_metrics:
            print("\n4. Google AI Analysis Results:")
            ai_analysis = quality_metrics['ai_analysis']
            print(f"   AI Overall Score: {ai_analysis.get('overall_score', 0):.2f}")
            print(f"   AI Face Score: {ai_analysis.get('face_score', 0):.2f}")
            print(f"   AI Body Score: {ai_analysis.get('body_score', 0):.2f}")
            
            if ai_analysis.get('recommendations'):
                print(f"\n   AI Recommendations ({len(ai_analysis['recommendations'])}):")
                for i, rec in enumerate(ai_analysis['recommendations'][:3], 1):
                    print(f"   {i}. {rec[:80]}...")
            
            print("\n   [OK] Google AI analysis successfully integrated!")
        else:
            print("\n   [WARNING] AI analysis not found in quality metrics")
            return False
        
        print("\n5. Testing refinement suggestions...")
        client = create_google_ai_client()
        if client:
            suggestions = client.get_refinement_suggestions(
                test_image,
                issues=["face blending", "edge seams"]
            )
            print("   [OK] Refinement suggestions generated")
            print(f"   Priority areas: {suggestions.get('priority_areas', [])}")
        else:
            print("   [WARNING] Could not create Google AI client")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_key_from_env():
    """Test that API key is read from environment correctly"""
    print("\n" + "=" * 60)
    print("Testing Environment Variable Loading")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if api_key:
            print(f"[OK] API key loaded from environment: {api_key[:20]}...")
            
            # Test it works
            from src.models.google_ai_client import create_google_ai_client
            client = create_google_ai_client()
            
            if client:
                print("[OK] Google AI client created from environment variable")
                return True
            else:
                print("[FAIL] Could not create client")
                return False
        else:
            print("[FAIL] API key not found in environment")
            return False
            
    except Exception as e:
        print(f"[FAIL] Environment test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("GOOGLE AI FULL INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = {
        "Environment Variable Loading": test_api_key_from_env(),
        "Full Pipeline Integration": test_full_pipeline_integration()
    }
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("\nGoogle AI is fully integrated and working correctly.")
        print("The system will automatically use Google AI for quality assessment")
        print("when processing images through the pipeline.")
    else:
        print("[WARNING] SOME TESTS FAILED")
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

