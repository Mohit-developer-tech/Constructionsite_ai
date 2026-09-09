import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Activity, ShieldAlert, BarChart3, Users, Maximize2, Info, Box, Sparkles, Image as ImageIcon, Layers, ArrowRight, BookOpen, Scale } from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const API = 'http://localhost:8000';

const StatCard = ({ title, value, icon: Icon, trend, color = 'primary' }) => (
  <motion.div variants={itemVariants} className="glass-panel-hover p-5 relative overflow-hidden group">
    <div className={`absolute top-0 right-0 w-28 h-28 bg-${color}/5 rounded-full -mr-8 -mt-8 blur-2xl group-hover:bg-${color}/10 transition-colors`} />
    <div className="flex justify-between items-start mb-3 relative z-10">
      <div className="p-2.5 bg-slate-800/80 rounded-xl shadow-inner border border-slate-700/50">
        <Icon className={`w-5 h-5 text-${color}`} />
      </div>
      {trend !== undefined && (
        <span className={`text-xs font-bold px-2 py-1 rounded-full ${trend > 0 ? 'bg-emerald-500/10 text-emerald-400' : trend < 0 ? 'bg-rose-500/10 text-rose-400' : 'bg-slate-700/50 text-slate-400'}`}>
          {trend > 0 ? '+' : ''}{trend}%
        </span>
      )}
    </div>
    <h3 className="text-slate-400 text-xs uppercase tracking-wider font-medium relative z-10">{title}</h3>
    <p className="text-2xl font-black mt-1 text-slate-100 relative z-10">{value}</p>
  </motion.div>
);

const ModelCard = ({ name, icon: Icon, description, path, color, metrics, badge }) => (
  <motion.div variants={itemVariants}>
    <Link to={path} className="block">
      <div className="glass-panel-hover p-6 h-full group cursor-pointer relative overflow-hidden">
        <div className={`absolute top-0 right-0 w-40 h-40 rounded-full -mr-16 -mt-16 blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-700`}
          style={{ background: `radial-gradient(circle, ${color}15, transparent)` }} />
        <div className="flex items-center justify-between mb-4 relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center border border-slate-700/50"
              style={{ background: `${color}15` }}>
              <Icon className="w-5 h-5" style={{ color }} />
            </div>
            <div>
              <h3 className="font-bold text-lg text-slate-100">{name}</h3>
              <span className="badge badge-primary text-[10px]">{badge}</span>
            </div>
          </div>
          <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-primary group-hover:translate-x-1 transition-all" />
        </div>
        <p className="text-sm text-slate-400 mb-4 leading-relaxed relative z-10">{description}</p>
        <div className="grid grid-cols-3 gap-2 relative z-10">
          {metrics.map((m, i) => (
            <div key={i} className="p-2 rounded-lg bg-slate-900/50 text-center">
              <div className="text-xs text-slate-500">{m.label}</div>
              <div className="text-sm font-bold text-slate-200">{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </Link>
  </motion.div>
);

const EDAChart = ({ title, imgUrl, className = "" }) => (
  <motion.div variants={itemVariants} className={`glass-panel-hover p-4 flex flex-col group ${className}`}>
    <div className="flex justify-between items-center mb-3">
      <h4 className="text-sm font-semibold text-slate-300">{title}</h4>
      <button className="p-1.5 text-slate-500 hover:text-primary transition-colors opacity-0 group-hover:opacity-100 bg-slate-800/50 rounded-lg">
        <Maximize2 className="w-3.5 h-3.5" />
      </button>
    </div>
    <div className="relative flex-1 w-full min-h-[180px] rounded-xl overflow-hidden bg-slate-900/80 border border-slate-800/50 flex items-center justify-center">
      <img src={imgUrl} alt={title}
        className="w-full h-full object-contain transform group-hover:scale-[1.02] transition-transform duration-500"
        onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
      />
      <div className="absolute inset-0 hidden items-center justify-center flex-col text-slate-600 gap-1">
        <Info className="w-6 h-6 opacity-40" />
        <span className="text-xs">Run notebook first</span>
      </div>
    </div>
  </motion.div>
);

export default function Dashboard() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-7xl mx-auto"
    >
      {/* Hero Header */}
      <header className="relative">
        <div className="absolute -top-20 -left-20 w-72 h-72 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute -top-10 right-0 w-56 h-56 bg-secondary/10 rounded-full blur-[80px] pointer-events-none" />
        <motion.div variants={itemVariants} className="relative z-10">
          <div className="flex items-center gap-2 mb-3">
            <span className="badge badge-primary">GenAI Lab Project</span>
            <span className="badge badge-secondary">Review 1</span>
          </div>
          <h2 className="text-5xl font-black tracking-tight gradient-text leading-tight">
            Construction Site AI
          </h2>
          <h3 className="text-2xl font-bold text-slate-400 mt-1">Digital Twin &amp; Generative Modeling</h3>
          <p className="text-slate-500 mt-3 max-w-2xl leading-relaxed">
            End-to-end AI framework for construction site safety analysis, scene reconstruction, 
            and synthetic data generation using Autoencoders, VAEs, and DCGANs.
          </p>
        </motion.div>
      </header>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Safety Classes" value="10" icon={ShieldAlert} trend={0} />
        <StatCard title="Training Images" value="3,245" icon={Activity} trend={12} />
        <StatCard title="Total Detections" value="14.2K" icon={Users} trend={8} />
        <StatCard title="Models Implemented" value="3 / 4" icon={BarChart3} trend={75} />
      </div>

      {/* Model Cards */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Implemented Models
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <ModelCard
            name="Autoencoder"
            icon={Box}
            badge="PRACT 2"
            description="4-layer CNN for dimensionality reduction and image reconstruction. Learns compressed latent representations of construction site scenes."
            path="/autoencoder"
            color="#14b8a6"
            metrics={[
              { label: 'SSIM', value: '0.78' },
              { label: 'PSNR', value: '22.4dB' },
              { label: 'Latent', value: '256-d' },
            ]}
          />
          <ModelCard
            name="VAE"
            icon={Sparkles}
            badge="PRACT 2"
            description="Probabilistic generative model with reparameterization trick. Generates novel construction scenes from structured latent space z ~ N(0, I)."
            path="/vae"
            color="#8b5cf6"
            metrics={[
              { label: 'SSIM', value: '0.72' },
              { label: 'FID', value: '142.5' },
              { label: 'Latent', value: '256-d' },
            ]}
          />
          <ModelCard
            name="DCGAN"
            icon={ImageIcon}
            badge="PRACT 3"
            description="Deep Convolutional GAN with adversarial training for photorealistic synthetic scene generation. Lowest FID score among all models."
            path="/gan"
            color="#34d399"
            metrics={[
              { label: 'FID', value: '98.3' },
              { label: 'Latent', value: '100-d' },
              { label: 'Epochs', value: '50' },
            ]}
          />
        </div>
      </motion.div>

      {/* Quick Links to Analysis Pages */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <motion.div variants={itemVariants}>
          <Link to="/compare" className="block">
            <div className="glass-panel-hover p-5 flex items-center gap-4 group cursor-pointer">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-rose-500 flex items-center justify-center flex-shrink-0 shadow-lg shadow-amber-500/20">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-slate-100 group-hover:text-amber-400 transition-colors">Comparative Analysis</h4>
                <p className="text-sm text-slate-400">Performance metrics, strengths/weaknesses, visual comparison across all 3 models.</p>
              </div>
              <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-amber-400 group-hover:translate-x-1 transition-all" />
            </div>
          </Link>
        </motion.div>
        <motion.div variants={itemVariants}>
          <Link to="/ethics" className="block">
            <div className="glass-panel-hover p-5 flex items-center gap-4 group cursor-pointer">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-rose-500 to-purple-500 flex items-center justify-center flex-shrink-0 shadow-lg shadow-rose-500/20">
                <Scale className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-slate-100 group-hover:text-rose-400 transition-colors">Ethics &amp; Responsible AI</h4>
                <p className="text-sm text-slate-400">Bias analysis, misuse prevention, privacy, sustainability, and regulatory compliance.</p>
              </div>
              <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-rose-400 group-hover:translate-x-1 transition-all" />
            </div>
          </Link>
        </motion.div>
      </div>

      {/* EDA Visualizations */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><BookOpen className="w-4 h-4" /></div>
          Exploratory Data Analysis
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <EDAChart title="Class Distribution" imgUrl={`${API}/data/processed/features/class_distribution.png`} className="lg:col-span-2 h-[320px]" />
          <EDAChart title="BBox Spatial Density" imgUrl={`${API}/data/processed/features/bbox_density.png`} className="h-[320px]" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
          <EDAChart title="Co-occurrence Matrix" imgUrl={`${API}/data/processed/features/co_occurrence.png`} className="h-[300px]" />
          <EDAChart title="Resolution Analysis" imgUrl={`${API}/data/processed/features/resolution_analysis.png`} className="h-[300px]" />
          <EDAChart title="Pixel Intensity" imgUrl={`${API}/data/processed/features/pixel_intensity.png`} className="h-[300px]" />
        </div>
      </motion.div>

      {/* Course Outcomes Mapping */}
      <motion.div variants={itemVariants} className="glass-panel-accent p-6">
        <div className="section-title">
          <div className="section-icon"><BookOpen className="w-4 h-4" /></div>
          Course Outcome Mapping (2304422L)
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>CO</th>
                <th>Description</th>
                <th>RBT</th>
                <th>Covered By</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="badge badge-primary">CO1</span></td>
                <td>Analyze Generative AI models to generate data across various domains</td>
                <td className="font-mono text-slate-400">L4</td>
                <td className="text-sm">AE, VAE, GAN — Architecture, Training, Analysis</td>
              </tr>
              <tr>
                <td><span className="badge badge-secondary">CO2</span></td>
                <td>Evaluate performance of Generative AI models for real-world applications</td>
                <td className="font-mono text-slate-400">L5</td>
                <td className="text-sm">Comparative Analysis (SSIM, PSNR, FID), Metrics</td>
              </tr>
              <tr>
                <td><span className="badge badge-success">CO3</span></td>
                <td>Deploy generative AI for practical applications</td>
                <td className="font-mono text-slate-400">L6</td>
                <td className="text-sm">Web App Deployment, Interactive Inference UI</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}
