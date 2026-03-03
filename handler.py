import runpod
from runpod.serverless.utils import rp_upload
import torch
from diffusers import FluxPipeline
import os
from huggingface_hub import login


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Cargando modelo FLUX nativo...")
        
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        
        pipe.enable_model_cpu_offload() 
        print("Modelo cargado con éxito.")

def handler(job):
    job_input = job["input"]
    job_id = job["id"]
    
    
    prompt = job_input.get("prompt", "A professional photo of a futuristic astronaut, 8k")
    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    steps = job_input.get("num_inference_steps", 20)
    guidance = job_input.get("guidance_scale", 3.5)

    load_model()
    
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        image = output.images[0]

    
    image_path = f"/tmp/{job_id}.png"
    image.save(image_path)

    
    bucket_name = os.getenv("BUCKET_NAME")
    
    
    upload_config = {
        "region_name": "eu-ro-1"
    }

    
    image_url = rp_upload.upload_image(
        job_id, 
        image_path, 
        bucket_name=bucket_name,
        custom_config=upload_config
    )

    
    if os.path.exists(image_path):
        os.remove(image_path)

    
    return {"image_url": image_url}


runpod.serverless.start({"handler": handler})
