import runpod
import os
import subprocess
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv


# Global variables for models
engine_built = False
HF_TOKEN = os.getenv('HF_TOKEN')

def build_engines():
    """Build TensorRT engines on first run"""
    global engine_built
    
    if engine_built:
        return
    
    print("Building TensorRT engines (this will take 10-20 minutes on first run)...")
    
    os.chdir("/workspace/TensorRT/demo/Diffusion")
    
    # Build engines with FP8 quantization
    cmd = [
        "python3", "demo_txt2img_sd35.py",
        "test prompt",
        "--version=3.5-large",
        "--fp8",
        "--denoising-steps=30",
        "--guidance-scale", "3.5",
        "--download-onnx-models",
        "--build-static-batch",
        "--use-cuda-graph",
        f"--hf-token={HF_TOKEN}",
        "--onnx-dir", "/workspace/models/onnx_fp8",
        "--engine-dir", "/workspace/models/engine_fp8",
        "--num-warmup-runs", "0"
    ]
    
    subprocess.run(cmd, check=True)
    engine_built = True
    print("Engines built successfully!")

def generate_image(prompt, negative_prompt="", steps=30, guidance_scale=3.5, seed=-1):
    """Generate image using TensorRT SD3.5"""
    
    # Ensure engines are built
    build_engines()
    
    os.chdir("/workspace/TensorRT/demo/Diffusion")
    
    # Create output directory
    os.makedirs("/tmp/output", exist_ok=True)
    
    # Prepare command
    cmd = [
        "python3", "demo_txt2img_sd35.py",
        prompt,
        "--version=3.5-large",
        "--fp8",
        f"--denoising-steps={steps}",
        f"--guidance-scale={guidance_scale}",
        "--build-static-batch",
        "--use-cuda-graph",
        f"--hf-token={HF_TOKEN}",
        "--onnx-dir", "/workspace/models/onnx_fp8",
        "--engine-dir", "/workspace/models/engine_fp8",
        "--num-warmup-runs", "0",
        "--output-dir", "/tmp/output"
    ]
    
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    
    # Run generation
    subprocess.run(cmd, check=True)
    
    # Find the generated image
    output_files = sorted(os.listdir("/tmp/output"))
    if not output_files:
        raise Exception("No image generated")
    
    latest_image = os.path.join("/tmp/output", output_files[-1])
    
    # Convert to base64
    with open(latest_image, "rb") as f:
        img_bytes = f.read()
    
    img_base64 = base64.b64encode(img_bytes).decode()
    
    # Clean up
    os.remove(latest_image)
    
    return img_base64

def handler(job):
    """RunPod serverless handler"""
    try:
        job_input = job["input"]
        
        # Extract parameters
        prompt = job_input.get("prompt", "A beautiful landscape")
        negative_prompt = job_input.get("negative_prompt", "")
        steps = job_input.get("steps", 30)
        guidance_scale = job_input.get("guidance_scale", 3.5)
        seed = job_input.get("seed", -1)
        
        # Generate image
        image_base64 = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        return {
            "image": image_base64,
            "prompt": prompt,
            "seed": seed
        }
        
    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})