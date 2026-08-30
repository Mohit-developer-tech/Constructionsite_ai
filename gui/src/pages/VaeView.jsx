import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2 } from 'lucide-react';
import axios from 'axios';

export default function VaeView() {
  const [generatedUrl, setGeneratedUrl] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post('http://localhost:8000/api/vae/generate');
      setGeneratedUrl('http://localhost:8000' + response.data.generated_image_url);
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsGenerating(false);
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
        <h2 className="text-3xl font-bold gradient-text">Variational Autoencoder (VAE)</h2>
        <p className="text-slate-400 mt-2">Samples from a learned continuous latent space to generate novel construction site structures.</p>
      </header>

      <div className="glass-panel p-8">
        <div className="flex flex-col items-center gap-6">
          <button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="flex items-center gap-2 px-8 py-4 rounded-full font-bold text-lg transition-all bg-gradient-to-r from-secondary to-primary text-white shadow-[0_0_20px_rgba(139,92,246,0.4)] hover:scale-105 hover:shadow-[0_0_30px_rgba(139,92,246,0.6)] disabled:opacity-50 disabled:hover:scale-100"
          >
            {isGenerating ? (
              <>Generating <Loader2 className="w-6 h-6 animate-spin" /></>
            ) : (
              <>Sample Latent Space <Sparkles className="w-6 h-6" /></>
            )}
          </button>

          <div className="relative w-full max-w-2xl aspect-video border-2 border-slate-700/50 rounded-2xl bg-slate-900 overflow-hidden shadow-inner flex items-center justify-center">
            {generatedUrl ? (
              <motion.img 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                src={generatedUrl} 
                alt="Generated VAE Sample" 
                className="w-full h-full object-cover" 
              />
            ) : (
              <div className="text-slate-500 font-medium flex flex-col items-center">
                <Sparkles className="w-12 h-12 mb-3 opacity-20" />
                No sample generated yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div>
        <h3 className="text-xl font-semibold mb-4 text-slate-200">Latent Space Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-panel p-4 h-80 relative">
            <h4 className="text-sm text-slate-400 absolute top-4 left-4 z-10 bg-slate-900/80 px-2 py-1 rounded">ELBO Loss Curve</h4>
            <img 
              src="http://localhost:8000/data/processed/features/vae_loss_curve.png" 
              className="w-full h-full object-contain pt-8" 
              alt="VAE Loss"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
          </div>
          <div className="glass-panel p-4 h-80 relative">
            <h4 className="text-sm text-slate-400 absolute top-4 left-4 z-10 bg-slate-900/80 px-2 py-1 rounded">Latent Manifold (t-SNE)</h4>
            <img 
              src="http://localhost:8000/data/processed/features/vae_tsne.png" 
              className="w-full h-full object-contain pt-8" 
              alt="VAE t-SNE"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
