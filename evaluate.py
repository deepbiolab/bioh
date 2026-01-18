import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import os
from sklearn.metrics import r2_score, mean_squared_error

from hybrid_model import HybridLSTM, HybridFFNN, HybridTransformer, get_feed_concs_dict
from simulation import HybridSimulator

# ==============================================================================
# 1. S 矩阵可视化 (Heatmap)
# ==============================================================================
def plot_s_matrix_heatmap(S_matrix, species_names, save_path='results/S_matrix_heatmap.png'):
    plt.figure(figsize=(12, 10))
    clean_names = [n.replace('Conc_', '') for n in species_names]
    
    ax = sns.heatmap(S_matrix, 
                     yticklabels=clean_names,
                     xticklabels=[f'PC {i+1}' for i in range(S_matrix.shape[1])],
                     cmap='RdBu_r', center=0, annot=False, cbar=True)
    
    plt.title('Reaction Correlation Matrix $S$\n(Mapping Latent Rates to Species Fluxes)', fontsize=16)
    plt.xlabel('Latent Reactions (PCA Components)', fontsize=12)
    plt.ylabel('Species', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ S Matrix heatmap saved to {save_path}")
    plt.show()


# ==============================================================================
# 2. 全局 Parity Plot (All Experiments - Per Species Breakdown)
# ==============================================================================
def plot_global_parity(simulator, df, exp_splits, species_names, save_path='results/Parity_Global_Per_Species.png'):
    """
    绘制包含 Train/Val/Test 所有数据的 Parity Plot，并按物质 (Species) 分子图展示。
    """
    print(f"Generating Global Parity Plot (Per Species)...")
    
    # 1. 收集所有数据并整理
    # 结构: species_data[species_index] = {'Train': {'real': [], 'pred': []}, ...}
    species_data = {i: {'Train': {'real': [], 'pred': []}, 
                        'Val': {'real': [], 'pred': []}, 
                        'Test': {'real': [], 'pred': []}} 
                    for i in range(len(species_names))}
    
    colors = {'Train': '#1f77b4', 'Val': '#ff7f0e', 'Test': '#2ca02c'} 
    markers = {'Train': 'o', 'Val': '^', 'Test': 's'}
    
    # 遍历每个 Split (Train/Val/Test)
    for split_name, exp_ids in exp_splits.items():
        for exp_id in exp_ids:
            exp_data = df[df['Experiment'] == exp_id].sort_values('Time')
            if len(exp_data) == 0: continue
            
            # 运行模拟
            res = simulator.simulate(exp_data)
            
            # 遍历每个物质，收集数据
            for idx in range(len(species_names)):
                real_vals = res['conc_real'][:, idx]
                pred_vals = res['conc_pred'][:, idx]
                
                species_data[idx][split_name]['real'].extend(real_vals)
                species_data[idx][split_name]['pred'].extend(pred_vals)

    # 2. 绘图配置
    n_plots = len(species_names)
    cols = 5 
    rows = (n_plots + cols - 1) // cols
    
    plt.figure(figsize=(20, 4 * rows)) # 增加高度，确保子图清晰
    
    # 3. 遍历每种物质绘制子图
    for idx, name in enumerate(species_names):
        ax = plt.subplot(rows, cols, idx + 1)
        clean_name = name.replace('Conc_', '')
        
        # 收集当前物质的所有数据以确定坐标轴范围
        all_real = []
        all_pred = []
        
        # 绘制 Train/Val/Test
        for split_name in ['Train', 'Val', 'Test']:
            y_real = np.array(species_data[idx][split_name]['real'])
            y_pred = np.array(species_data[idx][split_name]['pred'])
            
            if len(y_real) > 0:
                all_real.extend(y_real)
                all_pred.extend(y_pred)
                
                # 计算当前 Split 的 R2
                r2 = r2_score(y_real, y_pred) if len(y_real) > 1 else 0
                
                ax.scatter(y_real, y_pred, 
                           alpha=0.6, s=15, 
                           c=colors[split_name], marker=markers[split_name], edgecolors='none',
                           label=f'{split_name}') #  ($R^2$={r2:.2f})'
        
        # 计算该物质的全局指标
        y_real_all = np.array(all_real)
        y_pred_all = np.array(all_pred)
        
        if len(y_real_all) > 0:
            r2_global = r2_score(y_real_all, y_pred_all)
            rmse_global = np.sqrt(mean_squared_error(y_real_all, y_pred_all))
            
            # 确定坐标轴范围
            min_val = min(y_real_all.min(), y_pred_all.min())
            max_val = max(y_real_all.max(), y_pred_all.max())
            limit_min = min_val - 0.05 * (abs(max_val) - abs(min_val) + 1e-6)
            limit_max = max_val + 0.05 * (abs(max_val) - abs(min_val) + 1e-6)
            
            # 绘制对角线
            ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', linewidth=1.5, alpha=0.5)
            
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            ax.set_title(f"{clean_name}\n$R^2$={r2_global:.2f} | RMSE={rmse_global:.2f}", fontsize=10)
        
        # 仅在边缘添加轴标签
        if idx >= (rows - 1) * cols:
            ax.set_xlabel('Measured', fontsize=9)
        if idx % cols == 0:
            ax.set_ylabel('Predicted', fontsize=9)
            
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal') # 保持比例一致
        
        # 仅在第一个子图显示图例
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left', frameon=True)

    plt.suptitle(f"Global Parity Plot by Species (Train/Val/Test)", fontsize=16, y=1.01)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Global Parity Plot (Per Species) saved to {save_path}")

# ==============================================================================
# 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # --- MODEL TYPE PARAMETER ---
    MODEL_TYPE = 'LSTM' # Options: 'LSTM', 'FFNN', 'Transformer'
    # ----------------------------

    DATA_FILE = 'processed_data_IR_final.csv'
    ORIGINAL_FILE = 'data/data.xlsx'
    S_MATRIX_FILE = 's_matrix.pkl'
    SCALER_FILE = 'scaler.pkl'
    
    MODEL_FILE = f'hybrid_{MODEL_TYPE.lower()}_model.pth'
    
    print(f"Loading resources for {MODEL_TYPE} model evaluation...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file {MODEL_FILE} not found.")
        exit()

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if MODEL_TYPE == 'LSTM':
        model = HybridLSTM(S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device=device)
    elif MODEL_TYPE == 'FFNN':
        model = HybridFFNN(S_matrix, input_dim=27, latent_dim=7, device=device)
    elif MODEL_TYPE == 'Transformer':
        model = HybridTransformer(S_matrix, input_dim=27, latent_dim=7, device=device)
        
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    
    simulator = HybridSimulator(model, scaler, feed_concs, species_cols, device)
    
    # 1. Plot S Matrix
    print("\n[Viz] 1. Plotting S Matrix Heatmap...")
    plot_s_matrix_heatmap(S_matrix, species_cols)
    
    # 2. Plot Global Parity
    print("\n[Viz] 2. Plotting Global Parity Plot...")
    exp_splits = {
        'Train': [1, 2, 3, 4, 9],
        'Val': [7],
        'Test': [5, 6, 8]
    }
    plot_global_parity(simulator, df, exp_splits, species_cols)
    
    print("\n✅ Visualization Complete.")