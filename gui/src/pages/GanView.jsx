import { useState } from 'react';
import { motion } from 'framer-motion';
import { ImageIcon, Loader2 } from 'lucide-react';
import axios from 'axios';

export default function GanView() {
  const [generatedUrl, setGeneratedUrl] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post('http://localhost:8000/api/gan/generate');
      // For mock purposes, adding a timestamp to bypass browser caching
      setGeneratedUrl('http://localhost:8000' + response.data.generated_image_url + "?t=" + new Date().getTime());
    } catch (error) {
      console.error('Error generating GAN image:', error);
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
        <h2 className="text-3xl font-bold gradient-text">Generative Adversarial Network (GAN)</h2>
        <p className="text-slate-400 mt-2">Synthesizes ultra-realistic, high-fidelity construction site images using adversarial training.</p>
      </header>

      <div className="glass-panel p-8 relative overflow-hidden">
        {/* Decorative background glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary/20 blur-[100px] rounded-full pointer-events-none" />
        
        <div className="flex flex-col md:flex-row items-center gap-12 relative z-10">
          
          <div className="flex-1 text-center md:text-left">
            <h3 className="text-2xl font-semibold mb-4 text-slate-100">Generate Synthetic Data</h3>
            <p className="text-slate-400 mb-8 leading-relaxed">
              Use our trained GAN model to generate entirely new, photorealistic images of construction environments. These synthetic images can be used to augment training data for safety classification models, reducing the need for manual data collection.
            </p>
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="flex items-center justify-center w-full md:w-auto gap-2 px-8 py-4 rounded-xl font-bold text-lg transition-all bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-[0_0_20px_rgba(16,185,129,0.4)] disabled:opacity-50"
            >
              {isGenerating ? (
                <>Synthesizing <Loader2 className="w-6 h-6 animate-spin" /></>
              ) : (
                <>Generate Image <ImageIcon className="w-6 h-6" /></>
              )}
            </button>
          </div>

          <div className="flex-1 w-full max-w-md">
            <div className="aspect-square w-full border-4 border-slate-700/50 rounded-2xl bg-slate-900 overflow-hidden shadow-2xl relative flex items-center justify-center">
              {generatedUrl ? (
                <motion.img 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  key={generatedUrl}
                  src={generatedUrl} 
                  alt="Generated GAN Sample" 
                  className="w-full h-full object-cover" 
                />
              ) : (
                <div className="text-slate-500 font-medium flex flex-col items-center">
                  <ImageIcon className="w-16 h-16 mb-4 opacity-20" />
                  Ready to synthesize
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
         <div className="glass-panel p-6">
            <h3 className="text-xl font-semibold mb-4 text-slate-200">Generator vs Discriminator Loss</h3>
            <div className="relative w-full h-64 bg-slate-900 rounded-lg overflow-hidden border border-slate-700/50">
              <img 
                src="http://localhost:8000/data/synthetic/gan/gan_loss_curves.png" 
                alt="GAN Loss Curves" 
                className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
              />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
            </div>
         </div>
         <div className="glass-panel p-6">
            <h3 className="text-xl font-semibold mb-4 text-slate-200">Real vs Synthetic Comparison</h3>
            <div className="relative w-full h-64 bg-slate-900 rounded-lg overflow-hidden border border-slate-700/50">
              <img 
                src="http://localhost:8000/data/synthetic/gan/real_vs_synthetic.png" 
                alt="Real vs Synthetic" 
                className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
              />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Image missing</div>
            </div>
         </div>
      </div>
    </motion.div>
  );
}
