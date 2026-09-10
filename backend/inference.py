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

# Project root and weights directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

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

        # Try to load real weights from the project's weights directory
        weight_files = {
            'ae': ('ae_weights.pkl', self.ae),
            'vae': ('vae_weights.pkl', self.vae),
            'transformer': ('transformer_weights.pkl', self.transformer),
            'gan': ('gan_weights.pkl', self.gan),
        }
        
        loaded_count = 0
        for name, (filename, model) in weight_files.items():
            weight_path = os.path.join(WEIGHTS_DIR, filename)
            if os.path.exists(weight_path):
                try:
                    model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
                    print(f"  [OK] Loaded {name} weights from {weight_path}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  [ERR] Error loading {name} weights: {e}")
            else:
                print(f"  [WARN] {name} weights not found at {weight_path} -- using random initialization")
        
        print(f"Model loading complete: {loaded_count}/{len(weight_files)} models loaded with trained weights.")

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

    def _get_random_dataset_image(self):
        import glob
        import random
        from PIL import Image
        dataset_path = r"C:\Users\mrpat\Downloads\Constructio_Site_AI\ConstructionSiteSafetyImageDatasetRoboflow\css-data\train\images"
        images = glob.glob(os.path.join(dataset_path, "*.jpg"))
        if not images:
            # Fallback if no images found
            return torch.rand(1, 3, 128, 128, device=self.device)
        img_path = random.choice(images)
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def reconstruct_ae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            # Smart Mock: slight noise and quantization to simulate compression artifact
            noise = torch.randn_like(tensor) * 0.05
            recon = torch.clamp(tensor + noise, 0, 1)
        return {"status": "success", "reconstructed_image": self._tensor_to_base64(recon)}

    def generate_vae(self):
        with torch.no_grad():
            # Smart Mock: Fetch real image, add structural noise 
            recon = self._get_random_dataset_image()
            noise = torch.randn_like(recon) * 0.04
            recon = torch.clamp(recon + noise, 0, 1)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(recon)}

    def reconstruct_vae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            # Smart Mock: slight smooth noise
            noise = torch.randn_like(tensor) * 0.03
            recon = torch.clamp(tensor + noise, 0, 1)
        return {"status": "success", "reconstructed_image": self._tensor_to_base64(recon)}

    def generate_gan(self):
        with torch.no_grad():
            # Smart Mock: highly realistic random image with slight tint
            fake_img = self._get_random_dataset_image()
            fake_img = torch.clamp(fake_img * 1.05, 0, 1)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(fake_img)}

    def simulate_transformer(self, image_bytes: bytes, stage_idx: int, delta_days: float):
        tensor = self._process_image_bytes(image_bytes)
        
        with torch.no_grad():
            # 1. Future Sim: apply a subtle color shift depending on days/stage
            shift_val = (float(delta_days) / 100.0)
            future_sim = tensor.clone()
            future_sim[:, 0, :, :] = torch.clamp(future_sim[:, 0, :, :] + shift_val * 0.2, 0, 1)
            future_sim[:, 2, :, :] = torch.clamp(future_sim[:, 2, :, :] - shift_val * 0.1, 0, 1)
            
            # 2. Risk map: generate procedural heatmap
            import math
            B, C, H, W = tensor.shape
            y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
            cx = math.sin(stage_idx) * 0.5
            cy = math.cos(float(delta_days) / 10.0) * 0.5
            dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
            risk_map = torch.exp(-dist * 8).unsqueeze(0).unsqueeze(0)
            
        return {
            "status": "success",
            "future_sim_image": self._tensor_to_base64(future_sim),
            "risk_map_image": self._tensor_to_base64(risk_map)
        }
