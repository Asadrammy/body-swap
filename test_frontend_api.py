"""Test frontend API by creating a job directly"""

import requests
import json
from pathlib import Path
import os

# Files
customer_image = r"D:\projects\image\face-body-swap\1760713603491 (1).jpg"
template_image = r"D:\projects\image\face-body-swap\IMG20251019131550.jpg"

print("=" * 80)
print("Testing Frontend API - Creating Swap Job")
print("=" * 80)

# Check files exist
if not Path(customer_image).exists():
    print(f"ERROR: Customer image not found: {customer_image}")
    exit(1)

if not Path(template_image).exists():
    print(f"ERROR: Template image not found: {template_image}")
    exit(1)

print(f"Customer Image: {customer_image}")
print(f"Template Image: {template_image}")
print()

# Create job via API
api_url = "http://localhost:8000/api/v1/swap"
print(f"Calling API: {api_url}")

try:
    # Prepare files
    with open(customer_image, 'rb') as f1, open(template_image, 'rb') as f2:
        files = {
            'customer_photos': ('customer.jpg', f1, 'image/jpeg'),
            'template': ('template.jpg', f2, 'image/jpeg')
        }
        
        data = {}
        
        print("Sending request...")
        response = requests.post(api_url, files=files, data=data, timeout=5)
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"[OK] Job created successfully!")
            print(f"Job ID: {job_id}")
            print()
            print("Monitoring job status...")
            print("=" * 80)
            
            # Monitor job status
            status_url = f"http://localhost:8000/api/v1/jobs/{job_id}"
            max_wait = 300  # 5 minutes max
            waited = 0
            
            while waited < max_wait:
                status_response = requests.get(status_url, timeout=5)
                if status_response.status_code == 200:
                    job = status_response.json()
                    status = job.get('status')
                    progress = job.get('progress', 0)
                    error = job.get('error')
                    stage = job.get('current_stage', 'unknown')
                    
                    print(f"Status: {status} | Progress: {progress:.1%} | Stage: {stage}")
                    
                    if status == 'completed':
                        result_path = job.get('result_path')
                        print()
                        print("=" * 80)
                        print("[OK] JOB COMPLETED!")
                        print("=" * 80)
                        print(f"Result saved to: {result_path}")
                        if result_path:
                            result_file = Path(result_path)
                            if result_file.exists():
                                size = result_file.stat().st_size
                                print(f"Result file size: {size:,} bytes")
                                print(f"Result file: {result_path}")
                        break
                    elif status == 'failed':
                        print()
                        print("=" * 80)
                        print("[ERROR] JOB FAILED!")
                        print("=" * 80)
                        print(f"Error: {error}")
                        break
                    
                    import time
                    time.sleep(2)
                    waited += 2
                else:
                    print(f"Error checking status: {status_response.status_code}")
                    break
            
            if waited >= max_wait:
                print("\n[TIMEOUT] Timeout waiting for job to complete")
        else:
            print(f"[ERROR] Request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Response: {response.text[:500]}")
                
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)

