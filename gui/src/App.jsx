import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Image as ImageIcon, Sparkles, Layers, Box } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

import Dashboard from './pages/Dashboard';
import AutoencoderView from './pages/AutoencoderView';
import VaeView from './pages/VaeView';
import GanView from './pages/GanView';
import TransformerView from './pages/TransformerView';

function Sidebar() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/autoencoder', label: 'Autoencoder', icon: Box },
    { path: '/vae', label: 'VAE', icon: Sparkles },
    { path: '/gan', label: 'GAN Synthesizer', icon: ImageIcon },
    { path: '/transformer', label: 'Progress Transformer', icon: Layers },
  ];

  return (
    <div className="w-64 h-screen border-r border-slate-800 bg-darker flex flex-col p-4 shrink-0">
      <div className="flex items-center gap-3 mb-10 px-2 mt-4">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
          <Layers className="text-white w-5 h-5" />
        </div>
        <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">Construc.AI</h1>
      </div>

      <nav className="flex flex-col gap-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-300 ${
                isActive 
                ? 'bg-primary/10 text-primary shadow-[inset_0_0_12px_rgba(20,184,166,0.1)]' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-primary' : ''}`} />
              <span className="font-medium">{item.label}</span>
              {isActive && (
                <motion.div
                  layoutId="active-indicator"
                  className="absolute left-0 w-1 h-8 bg-primary rounded-r-full"
                  initial={false}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
            </Link>
          );
        })}
      </nav>
      
      <div className="mt-auto p-4 glass-panel text-sm text-slate-400">
        <p>System Status: <span className="text-emerald-400">Online</span></p>
        <p>Models Loaded: 4/4</p>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="flex w-full min-h-screen bg-darker text-slate-200 font-sans">
        <Sidebar />
        <main className="flex-1 h-screen overflow-y-auto overflow-x-hidden relative">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-800/30 via-darker to-darker pointer-events-none" />
          <div className="relative z-10 p-8 min-h-full">
            <AnimatePresence mode="wait">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/autoencoder" element={<AutoencoderView />} />
                <Route path="/vae" element={<VaeView />} />
                <Route path="/gan" element={<GanView />} />
                <Route path="/transformer" element={<TransformerView />} />
              </Routes>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
