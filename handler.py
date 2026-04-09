import runpod
import base64
import torch
import tempfile
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_PATH = "/runpod-volume/models/Wan-AI/Wan2.2-I2V-A14B-Diffusers"

print("Loading Wan2.2 model...")
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
).to("cuda")
print("Model loaded.")

def handler(job):
    job_input = job["input"]

    img_b64 = job_input["image"]
    prompt = job_input.get("prompt", "natural motion")
    negative_prompt = job_input.get("negative_prompt", "")
    num_frames = job_input.get("num_frames", 81)
    num_inference_steps = job_input.get("num_inference_steps", 40)
    guidance_scale = job_input.get("guidance", 5.0)
    width = job_input.get("width", 1280)
    height = job_input.get("height", 720)

    img_bytes = base64.b64decode(img_b64)
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    image = image.resize((width, height))

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    export_to_video(output.frames[0], tmp_path, fps=16)

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    os.unlink(tmp_path)

    return {"video": video_b64}

runpod.serverless.start({"handler": handler})
