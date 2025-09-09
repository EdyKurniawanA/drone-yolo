#!/usr/bin/env python3
"""
Debug script to identify why inference isn't working
"""

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

def debug_inference_issue():
    """Debug the inference issue step by step"""
    
    print("🔍 Debugging Roboflow Local Inference Issue")
    print("=" * 50)
    
    # Configuration
    INFERENCE_URL = "http://localhost:9001"
    API_KEY = "Pwr60R16IPozPzElpd1Q"
    WORKSPACE = "edys-flow"
    WORKFLOW_ID = "custom-workflow-2"
    
    # Step 1: Test server connection
    print("\n1️⃣ Testing server connection...")
    try:
        client = InferenceHTTPClient(
            api_url=INFERENCE_URL,
            api_key=API_KEY
        )
        print(f"✅ Client initialized for {INFERENCE_URL}")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Step 2: Test RTMP stream
    print("\n2️⃣ Testing RTMP stream...")
    rtmp_url = "rtmp://192.168.1.105/live"
    cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"❌ Cannot open RTMP stream: {rtmp_url}")
        print("💡 Try testing with VLC or another media player")
        return False
    
    print(f"✅ RTMP stream opened successfully")
    
    # Step 3: Capture a test frame
    print("\n3️⃣ Capturing test frame...")
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from RTMP stream")
        cap.release()
        return False
    
    print(f"✅ Captured frame: shape={frame.shape}, dtype={frame.dtype}")
    cap.release()
    
    # Step 4: Test inference with captured frame
    print("\n4️⃣ Testing inference with captured frame...")
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": frame
            }
        )
        
        print(f"✅ Inference completed successfully")
        print(f"📊 Result type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"📊 Result keys: {list(result.keys())}")
            
            # Check each output
            for key in result.keys():
                value = result[key]
                print(f"🔍 {key}: {type(value)} - {str(value)[:100]}...")
                
                if isinstance(value, dict):
                    print(f"   └─ Sub-keys: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"   └─ List length: {len(value)}")
                    if len(value) > 0:
                        print(f"   └─ First item: {type(value[0])} - {str(value[0])[:50]}...")
        else:
            print(f"📊 Result content: {result}")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test with a simple synthetic image
    print("\n5️⃣ Testing with synthetic image...")
    try:
        # Create a simple test image with a rectangle
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.putText(test_image, "TEST", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        result2 = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": test_image
            }
        )
        
        print(f"✅ Synthetic image inference completed")
        print(f"📊 Result type: {type(result2)}")
        
        if isinstance(result2, dict):
            print(f"📊 Result keys: {list(result2.keys())}")
        
    except Exception as e:
        print(f"❌ Synthetic image inference failed: {e}")
        return False
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 If inference is working here but not in the GUI, the issue is likely:")
    print("   - Frame processing loop not running")
    print("   - Frame queue not receiving frames")
    print("   - GUI not displaying results")
    
    return True

if __name__ == "__main__":
    success = debug_inference_issue()
    if not success:
        print("\n💥 Debug failed. Please check the error messages above.")
