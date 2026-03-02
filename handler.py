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
    job_id = job["id"] # Usamos el ID del trabajo para el nombre del archivo
    prompt = job_input.get("prompt", "A professional photo of a futuristic astronaut, 8k")
    
    load_model()
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5
        )
        image = output.images[0]

    
    image_path = f"/tmp/{job_id}.png"
    image.save(image_path)

    
    image_url = rp_upload.upload_image(job_id, image_path)

    
    if os.path.exists(image_path):
        os.remove(image_path)

    return {"image_url": image_url}

runpod.serverless.start({"handler": handler})
