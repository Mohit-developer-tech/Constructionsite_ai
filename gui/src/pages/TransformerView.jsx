import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Play, Clock, AlertTriangle, Loader2 } from 'lucide-react';
import axios from 'axios';

export default function TransformerView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [simulatedUrl, setSimulatedUrl] = useState(null);
  const [riskMapUrl, setRiskMapUrl] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  
  const [stage, setStage] = useState('1');
  const [days, setDays] = useState('30');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setSimulatedUrl(null);
      setRiskMapUrl(null);
    }
  };

  const handleSimulate = async () => {
    if (!selectedFile) return;
    setIsSimulating(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('stage_idx', stage);
    formData.append('delta_days', days);

    try {
      const response = await axios.post('http://localhost:8000/api/transformer/simulate', formData);
      setSimulatedUrl(response.data.future_sim_image);
      setRiskMapUrl('http://localhost:8000' + response.data.risk_map_image);
    } catch (error) {
      console.error('Error simulating progress:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsSimulating(false);
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
        <h2 className="text-3xl font-bold gradient-text">Construction Progress Transformer</h2>
        <p className="text-slate-400 mt-2">Multi-Modal Vision Transformer for predicting future site progress and spatial risk hazards.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Input & Controls */}
        <div className="glass-panel p-6 flex flex-col gap-6">
          <h3 className="text-xl font-semibold text-slate-200">Simulation Parameters</h3>
          
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Current State Image</label>
            <label className="relative flex flex-col items-center justify-center w-full h-40 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-800/50 transition-colors bg-slate-900 overflow-hidden group">
              {previewUrl ? (
                <img src={previewUrl} alt="Input" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
              ) : (
                <div className="flex flex-col items-center justify-center">
                  <Upload className="w-8 h-8 mb-2 text-slate-500" />
                  <span className="text-xs text-slate-400">Upload Image</span>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Target Construction Stage</label>
            <select 
              value={stage} 
              onChange={(e) => setStage(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-slate-200 focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
            >
              <option value="0">Excavation</option>
              <option value="1">Substructure</option>
              <option value="2">Framing</option>
              <option value="3">Facade</option>
              <option value="4">Finishing</option>
            </select>
          </div>

          <div>
            <label className="flex items-center justify-between text-sm font-medium text-slate-400 mb-2">
              <span>Time Horizon (Days)</span>
              <span className="text-primary">{days} Days</span>
            </label>
            <input 
              type="range" 
              min="1" max="90" 
              value={days} 
              onChange={(e) => setDays(e.target.value)}
              className="w-full accent-primary"
            />
          </div>

          <button
            onClick={handleSimulate}
            disabled={!selectedFile || isSimulating}
            className={`mt-4 flex items-center justify-center gap-2 py-4 rounded-xl font-bold text-lg transition-all ${
              !selectedFile ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-primary hover:bg-primary/90 text-slate-950 shadow-[0_0_20px_rgba(20,184,166,0.3)]'
            }`}
          >
            {isSimulating ? (
              <>Computing <Loader2 className="w-5 h-5 animate-spin" /></>
            ) : (
              <>Run Simulation <Play className="w-5 h-5" /></>
            )}
          </button>
        </div>

        {/* Outputs */}
        <div className="lg:col-span-2 flex flex-col gap-8">
          <div className="glass-panel p-6">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-secondary" />
              <h3 className="text-xl font-semibold text-slate-200">Predicted Future State</h3>
            </div>
            <div className="w-full aspect-[21/9] border border-slate-700 rounded-xl bg-slate-900 overflow-hidden flex items-center justify-center relative shadow-inner">
              {simulatedUrl ? (
                <img src={simulatedUrl} alt="Future Simulation" className="w-full h-full object-cover" />
              ) : (
                <div className="text-slate-600 font-medium">Awaiting Simulation</div>
              )}
            </div>
          </div>

          <div className="glass-panel p-6 border-l-4 border-l-rose-500">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="w-5 h-5 text-rose-500" />
              <h3 className="text-xl font-semibold text-slate-200">Spatial Risk Hazard Map</h3>
            </div>
            <div className="w-full aspect-[21/9] border border-slate-700 rounded-xl bg-slate-900 overflow-hidden flex items-center justify-center relative shadow-inner">
              {riskMapUrl ? (
                <div className="relative w-full h-full">
                  <img src={riskMapUrl} alt="Risk Map" className="w-full h-full object-cover opacity-80" />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent pointer-events-none" />
                </div>
              ) : (
                <div className="text-slate-600 font-medium">Awaiting Simulation</div>
              )}
            </div>
          </div>
        </div>

      </div>
    </motion.div>
  );
}
