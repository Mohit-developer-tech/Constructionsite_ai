import { motion } from 'framer-motion';
import { ShieldAlert, AlertTriangle, Eye, Leaf, Scale, Users, Lock, Globe, Cpu, FileWarning } from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const EthicsSection = ({ title, icon: Icon, color, colorClass, items, children }) => (
  <motion.div variants={itemVariants} className={`ethics-card ethics-card-${colorClass}`}>
    <div className="flex items-center gap-3 mb-4">
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center bg-${color}/10`}>
        <Icon className={`w-5 h-5 text-${color}`} />
      </div>
      <h3 className="text-xl font-bold text-slate-100">{title}</h3>
    </div>
    {items && (
      <ul className="space-y-3">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-3">
            <span className={`mt-1.5 w-2 h-2 rounded-full bg-${color} flex-shrink-0`}></span>
            <div>
              <span className="font-semibold text-slate-200">{item.title}: </span>
              <span className="text-slate-400 text-sm">{item.desc}</span>
            </div>
          </li>
        ))}
      </ul>
    )}
    {children}
  </motion.div>
);

export default function EthicsView() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-6xl mx-auto"
    >
      {/* Header */}
      <header className="mb-2">
        <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-amber-500 flex items-center justify-center shadow-lg shadow-rose-500/20">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-4xl font-black tracking-tight gradient-text-warm">Ethics &amp; Responsible AI</h2>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 text-lg max-w-3xl">
          Critical examination of ethical implications, potential biases, safety concerns, and responsible deployment 
          guidelines for generative AI in construction site monitoring.
        </motion.p>
      </header>

      {/* Bias Analysis */}
      <EthicsSection
        title="Bias Analysis"
        icon={Scale}
        color="amber-500"
        colorClass="bias"
        items={[
          {
            title: 'Dataset Geographic Bias',
            desc: 'The Roboflow construction safety dataset primarily contains images from specific geographic regions, leading to potential underperformance on construction sites in different climatic or cultural contexts (e.g., different PPE standards in tropical vs. cold regions).'
          },
          {
            title: 'Demographic Bias in PPE Detection',
            desc: 'Safety gear detection models may exhibit varying accuracy across different skin tones, body types, and clothing styles, particularly when trained on imbalanced datasets with limited diversity.'
          },
          {
            title: 'Class Imbalance Bias',
            desc: 'Our EDA reveals significant class imbalance (e.g., "Person" class dominates, while "Safety Cone" is underrepresented). Models trained on imbalanced data tend to be biased toward majority classes, potentially missing critical minority safety violations.'
          },
          {
            title: 'Synthetic Data Bias Amplification',
            desc: 'GAN-generated synthetic images may amplify existing biases in the training data. If the original dataset underrepresents certain scenarios (night shifts, indoor construction), synthetic data will perpetuate these gaps.'
          },
        ]}
      />

      {/* Misuse Prevention */}
      <EthicsSection
        title="Misuse Prevention &amp; Deepfake Risks"
        icon={FileWarning}
        color="rose-500"
        colorClass="misuse"
        items={[
          {
            title: 'Synthetic Image Misuse',
            desc: 'Our GAN can generate realistic construction site images that could potentially be misused to fabricate safety compliance reports, fake progress documentation, or create misleading evidence of construction completion.'
          },
          {
            title: 'Deepfake Construction Progress',
            desc: 'The Transformer-based progress simulation could be misused to generate fake "future state" images to deceive stakeholders about project timelines or completion status.'
          },
          {
            title: 'Mitigation Strategies',
            desc: 'We recommend implementing watermarking on all GAN-generated images, maintaining audit trails of synthetic vs. real data, and restricting model access through authentication. Generated images should carry metadata indicating their synthetic origin.'
          },
          {
            title: 'Content Provenance',
            desc: 'Following the C2PA (Coalition for Content Provenance and Authenticity) standard, all AI-generated construction imagery should include provenance metadata to distinguish synthetic from real captures.'
          },
        ]}
      />

      {/* Safety & Privacy */}
      <EthicsSection
        title="Worker Safety &amp; Privacy"
        icon={Eye}
        color="primary"
        colorClass="safety"
        items={[
          {
            title: 'Surveillance Concerns',
            desc: 'Deploying AI-based monitoring on construction sites raises significant worker privacy concerns. Continuous visual monitoring can create a hostile work environment and may violate labor rights in certain jurisdictions.'
          },
          {
            title: 'Data Protection (GDPR/India IT Act)',
            desc: 'Worker images constitute personal biometric data under GDPR and India\'s Digital Personal Data Protection Act 2023. Collection requires explicit consent, purpose limitation, and data minimization principles must be applied.'
          },
          {
            title: 'False Positive Impact',
            desc: 'Incorrect safety violation predictions can lead to unwarranted disciplinary actions against workers. Models should maintain high precision to minimize false accusations, and human verification should be mandatory before any action.'
          },
          {
            title: 'Worker Agency',
            desc: 'Workers should be informed about AI monitoring, have the right to contest AI-generated safety violation reports, and understand how their data is being used. Transparency is essential for ethical deployment.'
          },
        ]}
      />

      {/* Responsible AI */}
      <EthicsSection
        title="Responsible AI Deployment Guidelines"
        icon={Lock}
        color="secondary"
        colorClass="responsible"
        items={[
          {
            title: 'Model Transparency (Explainability)',
            desc: 'All model predictions should be accompanied by explanation mechanisms. For the Autoencoder, we can visualize reconstruction error heatmaps. For GAN, we should document the training data distribution. Latent space visualizations (t-SNE) provide interpretability for VAE embeddings.'
          },
          {
            title: 'Human-in-the-Loop',
            desc: 'AI-generated safety assessments should be treated as decision-support tools, not autonomous decision-makers. A qualified safety officer should review and validate all AI predictions before enforcement actions.'
          },
          {
            title: 'Model Versioning & Governance',
            desc: 'Maintain version control for all deployed models with documented training data provenance, hyperparameters, and evaluation results. Implement periodic model audits for performance drift and fairness metrics.'
          },
          {
            title: 'EU AI Act Compliance',
            desc: 'Construction site safety monitoring may fall under "high-risk" AI systems in the EU AI Act, requiring conformity assessments, risk management systems, and technical documentation. Our deployment should proactively comply with these emerging regulations.'
          },
          {
            title: 'India\'s National AI Strategy (NITI Aayog)',
            desc: 'Aligning with India\'s responsible AI principles: safety, inclusivity, transparency, accountability, and privacy protection. Our project demonstrates these principles through documented bias analysis and ethical safeguards.'
          },
        ]}
      />

      {/* Sustainability */}
      <EthicsSection
        title="Environmental Sustainability"
        icon={Leaf}
        color="emerald-500"
        colorClass="sustainability"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Cpu className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">~2.5 hrs</div>
            <div className="text-xs text-slate-400">Total GPU Training Time</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Globe className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">~0.8 kg CO₂</div>
            <div className="text-xs text-slate-400">Est. Carbon Footprint</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Leaf className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">128×128</div>
            <div className="text-xs text-slate-400">Resolution (Energy Efficient)</div>
          </div>
        </div>

        <ul className="space-y-3">
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Energy-Efficient Architecture: </span>
              <span className="text-slate-400 text-sm">
                We use 128×128 resolution instead of high-res alternatives (512×512, 1024×1024), reducing GPU 
                memory and compute by ~16x while maintaining sufficient quality for safety analysis.
              </span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Carbon Footprint Awareness: </span>
              <span className="text-slate-400 text-sm">
                Aligned with the Paris Agreement goals, we acknowledge the environmental cost of AI training. 
                Our models were trained with early stopping and efficient batch sizes to minimize unnecessary computation.
              </span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Transfer Learning Potential: </span>
              <span className="text-slate-400 text-sm">
                Pre-trained latent representations from our AE/VAE can be fine-tuned for new construction sites 
                without full retraining, significantly reducing the carbon cost of deployment to new environments.
              </span>
            </div>
          </li>
        </ul>
      </EthicsSection>

      {/* Summary Framework */}
      <motion.div variants={itemVariants} className="glass-panel-accent p-6">
        <div className="section-title">
          <div className="section-icon"><Users className="w-4 h-4" /></div>
          Ethical Framework Summary
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Principle</th>
                <th>Risk Level</th>
                <th>Our Mitigation</th>
                <th>Regulatory Reference</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-semibold text-slate-200">Data Privacy</td>
                <td><span className="badge badge-danger">High</span></td>
                <td>Consent mechanisms, data anonymization, minimal data retention</td>
                <td>GDPR, India DPDPA 2023</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Bias & Fairness</td>
                <td><span className="badge badge-warning">Medium</span></td>
                <td>Data augmentation, class balancing, fairness audits</td>
                <td>EU AI Act (Art. 10)</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Synthetic Misuse</td>
                <td><span className="badge badge-danger">High</span></td>
                <td>Watermarking, provenance metadata, access control</td>
                <td>C2PA Standard, IT Act Sec 66D</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Explainability</td>
                <td><span className="badge badge-warning">Medium</span></td>
                <td>t-SNE visualizations, error heatmaps, model documentation</td>
                <td>EU AI Act (Art. 13)</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Sustainability</td>
                <td><span className="badge badge-success">Low</span></td>
                <td>Efficient architectures, early stopping, transfer learning</td>
                <td>Paris Agreement, NITI Aayog</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>
    </motion.div>
  );
}
