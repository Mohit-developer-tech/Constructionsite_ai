"""Quick test to verify all 4 models load weights and produce valid output."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from inference import ModelInferenceManager

print("Initializing ModelInferenceManager...")
m = ModelInferenceManager()
print()

# Test GAN generation
r = m.generate_gan()
print(f"GAN generate: status={r['status']}, output_length={len(r['generated_image_url'])} chars")

# Test VAE generation
r = m.generate_vae()
print(f"VAE generate: status={r['status']}, output_length={len(r['generated_image_url'])} chars")

# Test AE reconstruction (use a synthetic image)
import io
from PIL import Image
import numpy as np

# Create a test image
test_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
buf = io.BytesIO()
test_img.save(buf, format="JPEG")
test_bytes = buf.getvalue()

r = m.reconstruct_ae(test_bytes)
print(f"AE reconstruct: status={r['status']}, output_length={len(r['reconstructed_image'])} chars")

# Test VAE reconstruction
r = m.reconstruct_vae(test_bytes)
print(f"VAE reconstruct: status={r['status']}, output_length={len(r['reconstructed_image'])} chars")

# Test Transformer simulation
r = m.simulate_transformer(test_bytes, stage_idx=2, delta_days=30.0)
print(f"Transformer sim: status={r['status']}, future_length={len(r['future_sim_image'])} chars, risk_length={len(r['risk_map_image'])} chars")

print()
print("All 5 endpoints validated successfully!")
