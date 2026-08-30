import io
import base64
from PIL import Image

class ModelInferenceManager:
    def __init__(self):
        # In a real scenario, you would load the PyTorch models here.
        # e.g., self.ae = ConvAutoencoder(); self.ae.load_state_dict(torch.load("ae.pkl"))
        self.mock_mode = True

    def _image_to_base64(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def reconstruct_ae(self, image_bytes: bytes):
        """Mock Autoencoder reconstruction"""
        if self.mock_mode:
            # Just return the same image back as a mock "reconstruction"
            img_b64 = self._image_to_base64(image_bytes)
            return {"status": "success", "reconstructed_image": f"data:image/jpeg;base64,{img_b64}"}

    def generate_vae(self):
        """Mock VAE generation - using one of the static synthetic images"""
        if self.mock_mode:
            # We'll return a static path from the data directory for the frontend to render
            return {"status": "success", "generated_image_url": "/data/synthetic/gan/generated_epoch_050.png"}

    def reconstruct_vae(self, image_bytes: bytes):
        """Mock VAE reconstruction"""
        if self.mock_mode:
            img_b64 = self._image_to_base64(image_bytes)
            return {"status": "success", "reconstructed_image": f"data:image/jpeg;base64,{img_b64}"}

    def generate_gan(self):
        """Mock GAN generation"""
        if self.mock_mode:
            return {"status": "success", "generated_image_url": "/data/synthetic/gan/gan_final_generated.png"}

    def simulate_transformer(self, image_bytes: bytes, stage_idx: int, delta_days: float):
        """Mock Progress Transformer simulation"""
        if self.mock_mode:
            # We return static placeholders for simulation and risk map
            img_b64 = self._image_to_base64(image_bytes)
            return {
                "status": "success",
                "future_sim_image": f"data:image/jpeg;base64,{img_b64}",
                "risk_map_image": "/data/processed/features/bbox_spatial.png" # Using spatial map as a mock risk map
            }
