import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference import ModelInferenceManager

app = FastAPI(title="Construction AI API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the data directory to serve static images/metrics
# This assumes backend is run from the project root or backend folder is parallel to data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

# Initialize the inference manager
inference_manager = ModelInferenceManager()

@app.get("/")
def read_root():
    return {"message": "Construction AI API is running"}

@app.post("/api/ae/reconstruct")
async def ae_reconstruct(file: UploadFile = File(...)):
    """Reconstruct an image using the Autoencoder"""
    content = await file.read()
    result = inference_manager.reconstruct_ae(content)
    return JSONResponse(content=result)

@app.post("/api/vae/generate")
async def vae_generate():
    """Generate a novel image using the VAE"""
    result = inference_manager.generate_vae()
    return JSONResponse(content=result)

@app.post("/api/vae/reconstruct")
async def vae_reconstruct(file: UploadFile = File(...)):
    """Reconstruct an image using the VAE"""
    content = await file.read()
    result = inference_manager.reconstruct_vae(content)
    return JSONResponse(content=result)

@app.post("/api/vae/generate_conditional")
async def vae_generate_conditional(
    class_idx:   int   = Form(...),
    temperature: float = Form(default=1.0),
):
    """
    Conditional VAE generation.
    Generates a construction site image conditioned on a specific safety class.

    Args:
        class_idx   : Safety class index 0–9
                      0=Hardhat, 1=Mask, 2=NO-Hardhat, 3=NO-Mask,
                      4=NO-Safety Vest, 5=Person, 6=Safety Cone,
                      7=Safety Vest, 8=machinery, 9=vehicle
        temperature : Sampling temperature (0.5–2.0). Higher = more diverse.
    """
    result = inference_manager.generate_cvae(class_idx=class_idx, temperature=temperature)
    return JSONResponse(content=result)

@app.post("/api/gan/generate")
async def gan_generate():
    """Generate a high-fidelity synthetic image using GAN"""
    result = inference_manager.generate_gan()
    return JSONResponse(content=result)

@app.post("/api/transformer/simulate")
async def transformer_simulate(
    file: UploadFile = File(...),
    stage_idx: int = Form(...),
    delta_days: float = Form(...)
):
    """Simulate future construction progress and risk map"""
    content = await file.read()
    result = inference_manager.simulate_transformer(content, stage_idx, delta_days)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
