import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error

# Import both model types
from hybrid_model import HybridLSTM, HybridFFNN, HybridTransformer, get_feed_concs_dict

# ==============================================================================
# 1. 混合模型模拟器
# ==============================================================================
class HybridSimulator:
    def __init__(self, model, scaler, feed_concs_dict, species_cols, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device
        self.species_cols = species_cols
        
        self.feed_conc_vec = np.zeros(len(species_cols))
        for i, col in enumerate(species_cols):
            met_name = col.replace('Conc_', '')
            if met_name not in ['Xv', 'mAb', 'Conc_mAb']:
                self.feed_conc_vec[i] = feed_concs_dict.get(met_name, 0.0)
        
    def simulate(self, exp_data):
        initial_row = exp_data.iloc[0]
        
        c_curr = initial_row[self.species_cols].values.astype(np.float32) 
        v_curr = float(initial_row['V_L'])
        t_curr = float(initial_row['Time'])
        
        hidden = None
        
        sim_results = {
            'time': [],
            'conc_pred': [],
            'vol_pred': [],
            'conc_real': exp_data[self.species_cols].values,
            'vol_real': exp_data['V_L'].values
        }
        
        feed_vols = exp_data['Feed_Vol_L'].values
        sample_vols = exp_data['Sample_Vol_L'].values
        times = exp_data['Time'].values
        
        num_steps = len(exp_data)
        
        sim_results['time'].append(t_curr)
        sim_results['conc_pred'].append(c_curr)
        sim_results['vol_pred'].append(v_curr)
        
        with torch.no_grad():
            for t in range(num_steps - 1):
                input_vec = np.hstack([c_curr, [v_curr], [t_curr]])
                input_scaled = self.scaler.transform(input_vec.reshape(1, -1))
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Model Prediction
                dMr_pred, hidden = self.model(input_tensor, hidden)
                dMr_val = dMr_pred.cpu().numpy().flatten()
                
                # Physics Step
                f_vol = feed_vols[t+1] 
                s_vol = sample_vols[t+1]
                
                mass_curr = c_curr * v_curr
                mass_in = f_vol * self.feed_conc_vec
                mass_out = s_vol * c_curr
                
                mass_next = mass_curr + dMr_val + mass_in - mass_out
                v_next = v_curr + f_vol - s_vol
                c_next = mass_next / v_next
                
                c_curr = c_next
                v_curr = v_next
                t_curr = times[t+1]
                
                sim_results['time'].append(t_curr)
                sim_results['conc_pred'].append(c_curr)
                sim_results['vol_pred'].append(v_curr)
                
        sim_results['conc_pred'] = np.array(sim_results['conc_pred'])
        sim_results['vol_pred'] = np.array(sim_results['vol_pred'])
        
        return sim_results

# ==============================================================================
# 2. 绘图与评估工具
# ==============================================================================
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

# ==============================================================================
# 3. 主执行流程
# ==============================================================================
if __name__ == "__main__":
    # --- MODEL TYPE PARAMETER ---
    MODEL_TYPE = 'LSTM' # Options: 'LSTM', 'FFNN', 'Transformer'
    # ----------------------------

    # Settings
    DATA_FILE = 'processed_data_IR_final.csv'
    ORIGINAL_FILE = 'data/data.xlsx'
    S_MATRIX_FILE = 's_matrix.pkl'
    SCALER_FILE = 'scaler.pkl'
    
    # Auto-detect filename based on type
    MODEL_FILE = f'hybrid_{MODEL_TYPE.lower()}_model.pth'
    
    print(f"Loading resources for {MODEL_TYPE} model...")
    df = pd.read_csv(DATA_FILE)
    feed_concs = get_feed_concs_dict(ORIGINAL_FILE)
    
    with open(S_MATRIX_FILE, 'rb') as f:
        S_matrix = pickle.load(f)
        
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
        
    met_list = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glc', 'Gln', 'Glu', 'Pyr', 
                'Gly', 'His', 'Ile', 'Lac', 'Leu', 'Lys', 'Met', 'Nh4', 'Phe', 
                'Pro', 'Ser', 'Thr', 'Tyr', 'Val']
    species_cols = ['Xv', 'Conc_mAb'] + [f'Conc_{m}' for m in met_list]
    
    # 2. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODEL_TYPE == 'LSTM':
        model = HybridLSTM(S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device=device)
    elif MODEL_TYPE == 'FFNN':
        model = HybridFFNN(S_matrix, input_dim=27, latent_dim=7, device=device)
    elif MODEL_TYPE == 'Transformer':
        model = HybridTransformer(S_matrix, input_dim=27, latent_dim=7, device=device)
        
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print("Model loaded.")
    
    # 3. Run Simulation
    test_exps = [5, 6, 8, 7] 
    
    simulator = HybridSimulator(model, scaler, feed_concs, species_cols, device)
    
    print("\nStarting Simulations...")
    for exp_id in test_exps:
        print(f"\n>>> Simulating Experiment {exp_id}...")
        exp_data = df[df['Experiment'] == exp_id].sort_values('Time')
        
        if len(exp_data) == 0:
            print(f"Warning: No data for Exp {exp_id}")
            continue
            
        results = simulator.simulate(exp_data)
        evaluate_experiment(results, exp_id, species_cols)
        
    print("\n✅ All simulations completed.")