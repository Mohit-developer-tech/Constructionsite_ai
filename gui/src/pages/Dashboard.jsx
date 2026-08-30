import { motion } from 'framer-motion';
import { Activity, ShieldAlert, BarChart3, Users } from 'lucide-react';

const StatCard = ({ title, value, icon: Icon, trend }) => (
  <div className="glass-panel p-6">
    <div className="flex justify-between items-start mb-4">
      <div className="p-3 bg-slate-800 rounded-lg">
        <Icon className="w-6 h-6 text-primary" />
      </div>
      <span className={`text-sm font-medium ${trend > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
        {trend > 0 ? '+' : ''}{trend}%
      </span>
    </div>
    <h3 className="text-slate-400 text-sm font-medium">{title}</h3>
    <p className="text-3xl font-bold mt-1 text-slate-100">{value}</p>
  </div>
);

export default function Dashboard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8"
    >
      <header>
        <h2 className="text-3xl font-bold gradient-text">Overview Dashboard</h2>
        <p className="text-slate-400 mt-2">Construction site digital twin metrics and hazard analysis.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Total Safety Classes" value="10" icon={ShieldAlert} trend={0} />
        <StatCard title="Dataset Size (Images)" value="3,245" icon={Activity} trend={12} />
        <StatCard title="Active Workers Detected" value="1,402" icon={Users} trend={-2} />
        <StatCard title="Avg Hazard Score" value="8.4" icon={BarChart3} trend={-5} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6">
          <h3 className="text-xl font-semibold mb-4 text-slate-200">Class Distribution</h3>
          <div className="relative w-full h-64 rounded-lg overflow-hidden bg-slate-900 border border-slate-700/50">
            {/* The backend will serve the data directory at localhost:8000/data/ */}
            <img 
              src="http://localhost:8000/data/processed/features/class_distribution.png" 
              alt="Class Distribution" 
              className="w-full h-full object-contain"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500">
              Backend not connected or image missing
            </div>
          </div>
        </div>

        <div className="glass-panel p-6">
          <h3 className="text-xl font-semibold mb-4 text-slate-200">Bounding Box Density</h3>
          <div className="relative w-full h-64 rounded-lg overflow-hidden bg-slate-900 border border-slate-700/50">
            <img 
              src="http://localhost:8000/data/processed/features/bbox_density.png" 
              alt="BBox Density" 
              className="w-full h-full object-contain"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500">
              Backend not connected or image missing
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
