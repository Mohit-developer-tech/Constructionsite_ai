import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, ArrowRight, Loader2, Box, Cpu, Hash, Layers, Activity } from 'lucide-react';
import axios from 'axios';

const API = 'http://localhost:8000';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

export default function AutoencoderView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [reconstructedUrl, setReconstructedUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReconstructedUrl(null);
    }
  };

  const handleReconstruct = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/api/ae/reconstruct`, formData);
      setReconstructedUrl(response.data.reconstructed_image);
    } catch (error) {
      console.error('Error reconstructing image:', error);
      alert('Failed to connect to backend. Make sure the server is running.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-6xl mx-auto"
    >
      {/* Header */}
      <header>
        <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-teal-400 flex items-center justify-center shadow-lg shadow-primary/20">
            <Box className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-black gradient-text">Convolutional Autoencoder</h2>
            <span className="badge badge-primary">PRACT 2 — Dimensionality Reduction &amp; Denoising</span>
          </div>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 mt-2 max-w-3xl">
          4-layer CNN encoder compresses 128×128 construction site images into a 256-dimensional latent bottleneck, 
          then a symmetric decoder reconstructs the original scene. Used for feature extraction and anomaly detection.
        </motion.p>
      </header>

      {/* Architecture Diagram */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Network Architecture
        </div>
        <div className="arch-flow flex-wrap justify-center">
          <div className="arch-block arch-block-input">
            <div className="text-xs opacity-60">Input</div>
            <div className="font-bold">3×128×128</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">Conv+BN+LReLU</div>
            <div className="font-bold">32×64×64</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">Conv+BN+LReLU</div>
            <div className="font-bold">64×32×32</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">Conv+BN+LReLU</div>
            <div className="font-bold">128×16×16</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">Conv+BN+LReLU</div>
            <div className="font-bold">256×8×8</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-latent animate-pulse-glow">
            <div className="text-xs opacity-60">Bottleneck</div>
            <div className="font-bold">z ∈ ℝ²⁵⁶</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-decoder">
            <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
            <div className="font-bold">128×16×16</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-decoder">
            <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
            <div className="font-bold">64×32×32</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-decoder">
            <div className="text-xs opacity-60">DeConv+Sigmoid</div>
            <div className="font-bold">3×128×128</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-output">
            <div className="text-xs opacity-60">Output</div>
            <div className="font-bold">Reconstructed</div>
          </div>
        </div>
      </motion.div>

      {/* Model Info + Math Formula */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Cpu className="w-4 h-4" /></div>
            Model Configuration
          </div>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Latent Dim</span>
              <span className="info-value">256</span>
            </div>
            <div className="info-item">
              <span className="info-label">Parameters</span>
              <span className="info-value">~5.2M</span>
            </div>
            <div className="info-item">
              <span className="info-label">Input Size</span>
              <span className="info-value">128×128</span>
            </div>
            <div className="info-item">
              <span className="info-label">Channels</span>
              <span className="info-value">3 (RGB)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Encoder</span>
              <span className="info-value">4 Conv</span>
            </div>
            <div className="info-item">
              <span className="info-label">Decoder</span>
              <span className="info-value">4 DeConv</span>
            </div>
            <div className="info-item">
              <span className="info-label">Activation</span>
              <span className="info-value">LeakyReLU</span>
            </div>
            <div className="info-item">
              <span className="info-label">Output Act.</span>
              <span className="info-value">Sigmoid</span>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Hash className="w-4 h-4" /></div>
            Mathematical Formulation
          </div>
          <div className="math-block">
            <div>
              <span className="math-fn">Encoder</span><span className="math-op">: </span>
              <span className="math-var">z</span> <span className="math-op">=</span> <span className="math-var">f</span><span className="math-sub">θ</span>(<span className="math-var">x</span>)
              <span className="math-op"> , where </span>
              <span className="math-var">x</span> <span className="math-op">∈</span> ℝ<span className="math-sup">3×128×128</span>
              <span className="math-op"> → </span>
              <span className="math-var">z</span> <span className="math-op">∈</span> ℝ<span className="math-sup">256</span>
            </div>
            <div className="mt-2">
              <span className="math-fn">Decoder</span><span className="math-op">: </span>
              <span className="math-var">x̂</span> <span className="math-op">=</span> <span className="math-var">g</span><span className="math-sub">φ</span>(<span className="math-var">z</span>)
              <span className="math-op"> , where </span>
              <span className="math-var">z</span> <span className="math-op">∈</span> ℝ<span className="math-sup">256</span>
              <span className="math-op"> → </span>
              <span className="math-var">x̂</span> <span className="math-op">∈</span> ℝ<span className="math-sup">3×128×128</span>
            </div>
            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <span className="math-fn">Loss</span> <span className="math-op">=</span> <span className="math-fn">MSE</span>(<span className="math-var">x</span>, <span className="math-var">x̂</span>)
              <span className="math-op"> = </span>
              <span className="math-op">1/</span><span className="math-var">n</span> <span className="math-op">·</span> <span className="math-op">Σ</span> (<span className="math-var">x</span><span className="math-sub">i</span> <span className="math-op">−</span> <span className="math-var">x̂</span><span className="math-sub">i</span>)<span className="math-sup">2</span>
            </div>
            <span className="math-label">Minimize pixel-wise reconstruction error between input and output</span>
          </div>
        </motion.div>
      </div>

      {/* Interactive Inference */}
      <motion.div variants={itemVariants} className="glass-panel p-8">
        <div className="section-title">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Live Inference — Upload &amp; Reconstruct
        </div>
        <div className="flex flex-col md:flex-row items-center justify-center gap-8">
          {/* Input Side */}
          <div className="flex-1 w-full max-w-sm flex flex-col items-center">
            <h4 className="text-sm font-semibold mb-3 text-slate-400 uppercase tracking-wider">Input Image</h4>
            <label className="upload-zone h-64 w-full">
              {previewUrl ? (
                <img src={previewUrl} alt="Input" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
              ) : (
                <div className="flex flex-col items-center justify-center py-6">
                  <Upload className="w-10 h-10 mb-3 text-slate-500" />
                  <p className="text-sm text-slate-400"><span className="font-semibold text-primary">Click to upload</span> or drag and drop</p>
                  <p className="text-xs text-slate-500 mt-1">Construction site image (JPG/PNG)</p>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>
          </div>

          {/* Action */}
          <div className="flex flex-col items-center gap-3">
            <button onClick={handleReconstruct} disabled={!selectedFile || isProcessing} className="btn-primary">
              {isProcessing ? (
                <>Processing <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Reconstruct <ArrowRight className="w-5 h-5" /></>
              )}
            </button>
          </div>

          {/* Output Side */}
          <div className="flex-1 w-full max-w-sm flex flex-col items-center">
            <h4 className="text-sm font-semibold mb-3 text-slate-400 uppercase tracking-wider">Reconstruction</h4>
            <div className="image-frame h-64 w-full">
              {reconstructedUrl ? (
                <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} src={reconstructedUrl} alt="Reconstructed" className="w-full h-full object-cover" />
              ) : (
                <div className="text-slate-600 text-sm font-medium">Awaiting Input</div>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Training Metrics */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Training Metrics &amp; Analysis
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-panel-hover p-4 h-64 relative flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">Loss Curve</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/processed/features/ae_loss_curve.png`} className="w-full h-full object-contain" alt="Loss Curve"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="glass-panel-hover p-4 h-64 relative flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">SSIM / PSNR Metrics</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/processed/features/ae_metrics.png`} className="w-full h-full object-contain" alt="Metrics"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="glass-panel-hover p-4 h-64 relative flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">t-SNE Latent Visualization</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/processed/features/ae_tsne.png`} className="w-full h-full object-contain" alt="t-SNE"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Original vs Reconstructed */}
      <motion.div variants={itemVariants} className="glass-panel-hover p-6">
        <h3 className="section-title">
          <div className="section-icon"><Box className="w-4 h-4" /></div>
          Original vs Reconstructed (Batch Comparison)
        </h3>
        <div className="image-frame h-80">
          <img src={`${API}/data/processed/features/ae_original_vs_reconstructed.png`} alt="Original vs Reconstructed" className="w-full h-full object-contain"
            onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
          <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate comparison grid</div>
        </div>
      </motion.div>
    </motion.div>
  );
}
