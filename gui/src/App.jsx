import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Image as ImageIcon, Sparkles, Layers, Box, BarChart3, ShieldAlert, Zap } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

import Dashboard from './pages/Dashboard';
import AutoencoderView from './pages/AutoencoderView';
import VaeView from './pages/VaeView';
import GanView from './pages/GanView';
import TransformerView from './pages/TransformerView';
import ComparativeAnalysis from './pages/ComparativeAnalysis';
import EthicsView from './pages/EthicsView';

function Sidebar() {
  const location = useLocation();

  const navGroups = [
    {
      label: 'Overview',
      items: [
        { path: '/', label: 'Dashboard', icon: LayoutDashboard },
      ],
    },
    {
      label: 'Generative Models',
      items: [
        { path: '/autoencoder', label: 'Autoencoder', icon: Box },
        { path: '/vae', label: 'VAE', icon: Sparkles },
        { path: '/gan', label: 'DCGAN', icon: Zap },
        { path: '/transformer', label: 'Transformer', icon: Layers, badge: 'R2' },
      ],
    },
    {
      label: 'Analysis',
      items: [
        { path: '/compare', label: 'Comparative', icon: BarChart3 },
        { path: '/ethics', label: 'Ethics & AI', icon: ShieldAlert },
      ],
    },
  ];

  return (
    <div className="w-64 h-screen border-r border-slate-200 bg-white flex flex-col shrink-0 shadow-sm">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 pt-6 pb-5">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-lg shadow-primary/20">
          <Layers className="text-white w-5 h-5" />
        </div>
        <div>
          <h1 className="text-lg font-black bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary leading-tight">Construc.AI</h1>
          <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">Digital Twin Lab</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 pb-4">
        {navGroups.map((group, gi) => (
          <div key={gi} className="mb-4">
            <div className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold px-3 mb-2">{group.label}</div>
            <div className="flex flex-col gap-1">
              {group.items.map((item) => {
                const isActive = location.pathname === item.path;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-300 ${
                      isActive 
                      ? 'bg-primary/10 text-primary shadow-[inset_0_0_12px_rgba(0,49,83,0.06)]' 
                      : 'text-slate-600 hover:text-slate-800 hover:bg-slate-100'
                    }`}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="active-indicator"
                        className="absolute left-0 w-1 h-7 bg-primary rounded-r-full"
                        initial={false}
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      />
                    )}
                    <Icon className={`w-4.5 h-4.5 ${isActive ? 'text-primary' : ''}`} />
                    <span className="font-medium text-sm">{item.label}</span>
                    {item.badge && (
                      <span className="ml-auto text-[9px] px-1.5 py-0.5 rounded-md bg-primary/10 text-primary font-bold">{item.badge}</span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
      
      {/* Footer */}
      <div className="px-3 pb-5">
        <div className="text-[10px] text-slate-400 text-center font-medium tracking-wide">
          ConstructionSite AI &copy; 2024
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="flex w-full min-h-screen bg-slate-50 text-slate-800 font-sans">
        <Sidebar />
        <main className="flex-1 h-screen overflow-y-auto overflow-x-hidden relative">
          <div className="absolute inset-0 bg-grid bg-radial-glow pointer-events-none" />
          <div className="relative z-10 p-8 min-h-full">
            <AnimatePresence mode="wait">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/autoencoder" element={<AutoencoderView />} />
                <Route path="/vae" element={<VaeView />} />
                <Route path="/gan" element={<GanView />} />
                <Route path="/transformer" element={<TransformerView />} />
                <Route path="/compare" element={<ComparativeAnalysis />} />
                <Route path="/ethics" element={<EthicsView />} />
              </Routes>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
