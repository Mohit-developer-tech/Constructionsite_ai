import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, ArrowRight, Loader2 } from 'lucide-react';
import axios from 'axios';

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
      const response = await axios.post('http://localhost:8000/api/ae/reconstruct', formData);
      setReconstructedUrl(response.data.reconstructed_image);
    } catch (error) {
      console.error('Error reconstructing image:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8"
    >
      <header>
        <h2 className="text-3xl font-bold gradient-text">Convolutional Autoencoder</h2>
        <p className="text-slate-400 mt-2">Compresses construction site images into a dense latent bottleneck and reconstructs them.</p>
      </header>

      <div className="glass-panel p-8">
        <div className="flex flex-col md:flex-row items-center justify-center gap-8">
          
          {/* Input Side */}
          <div className="flex-1 w-full max-w-sm flex flex-col items-center">
            <h3 className="text-lg font-semibold mb-4">Input Image</h3>
            <label className="relative flex flex-col items-center justify-center w-full h-64 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-800/50 transition-colors bg-slate-900 overflow-hidden group">
              {previewUrl ? (
                <img src={previewUrl} alt="Input" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
              ) : (
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-10 h-10 mb-3 text-slate-500" />
                  <p className="mb-2 text-sm text-slate-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>
          </div>

          {/* Action Button */}
          <div className="flex flex-col items-center justify-center">
            <button
              onClick={handleReconstruct}
              disabled={!selectedFile || isProcessing}
              className={`flex items-center gap-2 px-6 py-3 rounded-full font-semibold transition-all ${
                !selectedFile ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-primary hover:bg-primary/90 text-slate-950 shadow-[0_0_20px_rgba(20,184,166,0.3)] hover:scale-105'
              }`}
            >
              {isProcessing ? (
                <>Processing <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Reconstruct <ArrowRight className="w-5 h-5" /></>
              )}
            </button>
          </div>

          {/* Output Side */}
          <div className="flex-1 w-full max-w-sm flex flex-col items-center">
            <h3 className="text-lg font-semibold mb-4">Reconstruction</h3>
            <div className="relative w-full h-64 border border-slate-700 rounded-xl bg-slate-900 overflow-hidden shadow-inner flex items-center justify-center">
              {reconstructedUrl ? (
                <img src={reconstructedUrl} alt="Reconstructed" className="w-full h-full object-cover" />
              ) : (
                <div className="text-slate-600 text-sm font-medium">Awaiting Input</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div>
        <h3 className="text-xl font-semibold mb-4 text-slate-200">Training Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-panel p-4 h-64 relative">
            <img 
              src="http://localhost:8000/data/processed/features/ae_loss_curve.png" 
              className="w-full h-full object-contain" 
              alt="Loss Curve"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
          </div>
          <div className="glass-panel p-4 h-64 relative">
            <img 
              src="http://localhost:8000/data/processed/features/ae_metrics.png" 
              className="w-full h-full object-contain" 
              alt="Metrics"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
          </div>
          <div className="glass-panel p-4 h-64 relative">
            <img 
              src="http://localhost:8000/data/processed/features/ae_tsne.png" 
              className="w-full h-full object-contain" 
              alt="t-SNE"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
