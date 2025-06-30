import requests
import base64
import json
from PIL import Image
from io import BytesIO
import time
import os
from dotenv import load_dotenv

# Your RunPod endpoint details
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

def generate_image_async(prompt, negative_prompt="", steps=30, guidance_scale=3.5, seed=-1):
    """Generate an image using RunPod serverless endpoint with async handling"""
    
    # Start the job
    run_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }
    }
    
    print(f"Generating image for: {prompt}")
    print("Submitting job to RunPod...")
    start_time = time.time()
    
    try:
        # Submit the job
        response = requests.post(run_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        job_id = result.get('id')
        if not job_id:
            print(f"No job ID in response: {result}")
            return None
            
        print(f"Job submitted! ID: {job_id}")
        print("Note: First run will take 15-20 minutes to build TensorRT engines...")
        print("Polling for results...")
        
        # Poll for results
        status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
        
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            status_response = requests.get(status_url, headers=headers)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data.get('status')
            print(f"Status: {status} ({int(time.time() - start_time)}s elapsed)")
            
            if status == 'COMPLETED':
                output = status_data.get('output')
                if isinstance(output, dict) and "image" in output:
                    image_base64 = output["image"]
                elif isinstance(output, dict) and "error" in output:
                    print(f"Error from endpoint: {output['error']}")
                    return None
                else:
                    print(f"Unexpected output format: {output}")
                    return None
                
                # Convert to PIL Image
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes))
                
                elapsed = time.time() - start_time
                print(f"✅ Generated in {elapsed:.2f} seconds")
                
                return image
                
            elif status == 'FAILED':
                print(f"Job failed: {status_data}")
                return None
                
            elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                # Still processing, continue polling
                continue
                
            else:
                print(f"Unknown status: {status}")
                
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("SD3.5 TensorRT Async Test Client")
    print("=" * 50)
    
    # Test 1: Simple prompt
    print("\nTest 1: First generation (will be slow)")
    image = generate_image_async(
        prompt="A majestic lion standing on a cliff at sunset, photorealistic",
        negative_prompt="blurry, low quality, distorted",
        steps=30,
        guidance_scale=3.5
    )
    
    if image:
        image.save("test_1_lion.png")
        print("✅ Image saved as test_1_lion.png")
        
        # If first test succeeded, run more tests
        print("\n" + "=" * 50)
        print("First generation complete! Now subsequent generations will be MUCH faster!")
        print("=" * 50)
        
        # Test 2: Should be fast now
        print("\nTest 2: Second generation (should be fast)")
        image2 = generate_image_async(
            prompt="A futuristic cyberpunk city at night with neon lights",
            negative_prompt="daylight, rural",
            steps=30,
            guidance_scale=4.0
        )
        
        if image2:
            image2.save("test_2_cyberpunk.png")
            print("✅ Image saved as test_2_cyberpunk.png")
    else:
        print("❌ First generation failed")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("=" * 50)