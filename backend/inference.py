import io
import os
import sys
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add project root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder import ConvAutoencoder
from src.models.vae import ConvVAE
from src.models.transformer_models import ConstructionProgressTransformer
from src.models.gan import Generator

class ModelInferenceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing models on {self.device}...")
        
        # Initialize models with architecture defaults
        self.ae = ConvAutoencoder(in_channels=3, latent_dim=256).to(self.device)
        self.vae = ConvVAE(in_channels=3, latent_dim=256).to(self.device)
        self.transformer = ConstructionProgressTransformer(
            img_size=128, patch_size=16, in_channels=3, num_stages=5, embed_dim=256, depth=4, num_heads=8
        ).to(self.device)
        self.gan = Generator(latent_dim=100, out_channels=3, features=64).to(self.device)

        # Try to load real weights, else fallback to random initialization
        try:
            self.ae.load_state_dict(torch.load("ae_weights.pkl", map_location=self.device))
            self.vae.load_state_dict(torch.load("vae_weights.pkl", map_location=self.device))
            self.transformer.load_state_dict(torch.load("transformer_weights.pkl", map_location=self.device))
            self.gan.load_state_dict(torch.load("gan_weights.pkl", map_location=self.device))
            print("Successfully loaded model weights.")
        except FileNotFoundError:
            print("Warning: Model weights (.pkl) not found. Running with randomly initialized weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")

        # Evaluation mode
        self.ae.eval()
        self.vae.eval()
        self.transformer.eval()
        self.gan.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def _tensor_to_base64(self, tensor):
        # Move to CPU and remove batch dimension
        tensor = tensor.detach().cpu().squeeze(0)
        
        # Rescale [-1, 1] to [0, 1] if Tanh was used (e.g. GAN)
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2.0
            
        tensor = torch.clamp(tensor, 0, 1)
        pil_img = transforms.ToPILImage()(tensor)
        
        # For 1-channel risk map, colorize it (basic heatmap approximation)
        if tensor.shape[0] == 1:
            # We can just return grayscale for now or apply a colormap. Grayscale is safest.
            pass
            
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _process_image_bytes(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def reconstruct_ae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            recon, _ = self.ae(tensor)
        return {"status": "success", "reconstructed_image": self._tensor_to_base64(recon)}

    def generate_vae(self):
        with torch.no_grad():
            # sample from standard normal
            recon = self.vae.sample(1, self.device)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(recon)}

    def reconstruct_vae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            recon, _, _ = self.vae(tensor)
        return {"status": "success", "reconstructed_image": self._tensor_to_base64(recon)}

    def generate_gan(self):
        with torch.no_grad():
            z = torch.randn(1, 100, 1, 1, device=self.device)
            fake_img = self.gan(z)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(fake_img)}

    def simulate_transformer(self, image_bytes: bytes, stage_idx: int, delta_days: float):
        tensor = self._process_image_bytes(image_bytes)
        stage_tensor = torch.tensor([int(stage_idx)], dtype=torch.long, device=self.device)
        time_tensor = torch.tensor([[float(delta_days)]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            future_sim, risk_map = self.transformer(tensor, stage_tensor, time_tensor)
            
        return {
            "status": "success",
            "future_sim_image": self._tensor_to_base64(future_sim),
            "risk_map_image": self._tensor_to_base64(risk_map)
        }
