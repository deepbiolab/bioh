import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error

from hybrid_model_v2 import HybridLSTM, HybridFFNN, ExperimentConfig

class HybridSimulator:
    def __init__(self, model, scaler, config, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device
        self.conf = config
        
    def simulate(self, exp_data):
        # Initial State
        initial_row = exp_data.iloc[0]
        
        # Current State pointers
        c_curr = initial_row[self.conf.x_cols].values.astype(np.float32)
        v_curr = float(initial_row[self.conf.vol_col])
        t_curr = float(initial_row[self.conf.time_col])
        
        # Pre-load known future inputs (Controls W and Feeds F)
        w_seq = exp_data[self.conf.w_cols].values # (T, n_w)
        times = exp_data[self.conf.time_col].values
        
        # Prepare F (Feed Mass) matrices
        feed_mass_map = {}
        for f_col, target_x in self.conf.f_mapping.items():
            feed_mass_map[target_x] = exp_data[f_col].values # (T,)
            
        feed_vol_seq = exp_data[self.conf.feed_vol_col].values # (T,)
        
        sim_res = {
            'time': [t_curr],
            'conc_pred': [c_curr],
            'conc_real': exp_data[self.conf.x_cols].values,
            'vol_pred': [v_curr],
            'vol_real': exp_data[self.conf.vol_col].values
        }
        
        hidden = None
        num_steps = len(exp_data)
        
        with torch.no_grad():
            for t in range(num_steps - 1):
                # 1. Prepare Input: [X, V, T, W]
                # W comes from current step (t)
                w_curr = w_seq[t]
                
                input_vec = np.hstack([c_curr, [v_curr], [t_curr], w_curr])
                input_scaled = self.scaler.transform(input_vec.reshape(1, -1))
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 2. Model Predict dMr
                dMr_pred, hidden = self.model(input_tensor, hidden)
                dMr_val = dMr_pred.cpu().numpy().flatten()
                
                # 3. Physics Update
                # Get Feeds for interval t -> t+1
                # (Assuming feed row t+1 represents what was added BY time t+1)
                # Or row t representing what is ADDED at t. 
                # Based on standard logs, usually row i contains cumulative or setpoint. 
                # Let's assume the 'Cmd_Feed' at t+1 is the increment or we use the pre-calculated diff.
                # In preprocess, we treated values as increments. Let's consistency use index t+1.
                
                f_mass_vec = np.zeros_like(c_curr)
                for i, x_name in enumerate(self.conf.x_cols):
                    if x_name in feed_mass_map:
                        f_mass_vec[i] = feed_mass_map[x_name][t+1]
                
                f_vol = feed_vol_seq[t+1]
                
                # Mass Balance: M(t+1) = M(t) + dM_rxn + Feed_Mass
                mass_curr = c_curr * v_curr
                mass_next = mass_curr + dMr_val + f_mass_vec
                
                # Volume Balance
                v_next = v_curr + f_vol # Assuming Sample is negligible or handled
                
                c_next = mass_next / v_next
                
                # Update
                c_curr = c_next
                v_curr = v_next
                t_curr = times[t+1]
                
                sim_res['time'].append(t_curr)
                sim_res['conc_pred'].append(c_curr)
                sim_res['vol_pred'].append(v_curr)
                
        sim_res['conc_pred'] = np.array(sim_res['conc_pred'])
        return sim_res

def evaluate_experiment(sim_res, exp_id, species_names, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    time = sim_res['time']
    c_pred = sim_res['conc_pred']
    c_real = sim_res['conc_real']
    
    r2_all = r2_score(c_real.flatten(), c_pred.flatten())
    rmse_all = np.sqrt(mean_squared_error(c_real.flatten(), c_pred.flatten()))
    print(f"Exp {exp_id} | Overall R2: {r2_all:.4f} | Overall RMSE: {rmse_all:.4f}")
    
    n_plots = len(species_names) 
    cols = 5                     
    rows = (n_plots + cols - 1) // cols 
    
    plt.figure(figsize=(20, 3.5 * rows))
    
    for idx, name in enumerate(species_names):
        y_real = c_real[:, idx]
        y_pred = c_pred[:, idx]
        
        r2 = r2_score(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        
        plt.subplot(rows, cols, idx + 1)
        plt.plot(time, y_real, 'ko', markersize=4, alpha=0.7, label='Exp Data') 
        plt.plot(time, y_pred, 'b-', linewidth=2, label='Hybrid Model') 
        
        clean_name = name.replace('Conc_', '') 
        plt.title(f"{clean_name}\n$R^2$={r2:.2f} | RMSE={rmse:.2f}", fontsize=10)
        
        if idx >= (rows - 1) * cols:
            plt.xlabel('Time (h)', fontsize=9)
        if idx % cols == 0:
            plt.ylabel('Conc (mM/g/L)', fontsize=9)
            
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        if idx == 0:
            plt.legend(loc='best', fontsize=8, frameon=True)
            
    plt.suptitle(f"Digital Twin Simulation: Experiment {exp_id} (All Species)", fontsize=16, y=1.01)
    plt.tight_layout()
    
    file_path = f"{save_dir}/Sim_Exp_{exp_id}_All.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"Saved plot to {file_path}")

if __name__ == "__main__":
    MODEL_TYPE = 'LSTM'
    DATA_FILE = 'data/hist.csv'
    conf = ExperimentConfig()
    
    # Reload processed data (same logic)
    # Ideally save processed df to csv in step 1 to avoid re-process
    from hybrid_model_v2 import preprocess_novel_data
    df = preprocess_novel_data(DATA_FILE, conf)
    
    # Load Artifacts
    with open('s_matrix.pkl', 'rb') as f: S = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('splits.pkl', 'rb') as f: train_ids, val_ids, test_ids = pickle.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridLSTM(S, conf.input_dim, device=device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    sim = HybridSimulator(model, scaler, conf, device)
    
    print("\nRunning Test Simulations...")
    # Select first 3 test runs
    for run_id in test_ids[:10]:
        print(f"Simulating {run_id}...")
        data = df[df['Run_ID'] == run_id].sort_values(conf.time_col)
        res = sim.simulate(data)

        evaluate_experiment(res, run_id, sim.conf.x_cols)
        
        # Simple print R2
        r2 = r2_score(res['conc_real'].flatten(), res['conc_pred'].flatten())
        print(f"Run {run_id} R2: {r2:.4f}")