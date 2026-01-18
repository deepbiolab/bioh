import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import os
from sklearn.metrics import r2_score

from hybrid_model_v2 import HybridLSTM, HybridFFNN, ExperimentConfig, preprocess_novel_data
from simulation_v2 import HybridSimulator

def plot_s_matrix(S, x_names, save_path='results/S_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(S, yticklabels=x_names, cmap='RdBu_r', center=0, annot=True, fmt=".2f")
    plt.title('S Matrix (Species x Latent)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_trajectory(sim_res, x_names, run_id, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    time = sim_res['time']
    real = sim_res['conc_real']
    pred = sim_res['conc_pred']
    
    n = len(x_names)
    cols = 3
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 3*rows))
    for i, name in enumerate(x_names):
        plt.subplot(rows, cols, i+1)
        plt.plot(time, real[:, i], 'ko', label='Exp')
        plt.plot(time, pred[:, i], 'b-', label='Sim')
        plt.title(name)
        plt.grid(True, alpha=0.3)
        if i==0: plt.legend()
        
    plt.suptitle(f"Simulation: {run_id}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Sim_{run_id}.png")
    plt.close()

if __name__ == "__main__":
    conf = ExperimentConfig()
    
    # Load all needed
    df = preprocess_novel_data('data/hist.csv', conf)
    with open('s_matrix.pkl', 'rb') as f: S = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('splits.pkl', 'rb') as f: train_ids, val_ids, test_ids = pickle.load(f)
    
    device = torch.device("cpu")
    model = HybridLSTM(S, conf.input_dim, device=device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    sim = HybridSimulator(model, scaler, conf, device)
    
    # 1. Plot S Matrix
    plot_s_matrix(S, conf.x_cols)
    print("S Matrix plotted.")
    
    # 2. Plot Test Trajectories
    for run_id in test_ids[:5]: # Plot first 5 test runs
        data = df[df['Run_ID'] == run_id].sort_values(conf.time_col)
        res = sim.simulate(data)
        plot_trajectory(res, conf.x_cols, run_id)
        print(f"Plotted {run_id}")