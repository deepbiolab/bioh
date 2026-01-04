import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error

# 引入之前的模型定义 (为了独立运行，这里再次包含类定义，或者你可以从 hybrid_model import)
# 建议保持此文件独立，以免依赖报错
from hybrid_model import HybridLSTM, get_feed_concs_dict

# ==============================================================================
# 1. 混合模型模拟器 (The Digital Twin Engine)
# ==============================================================================
class HybridSimulator:
    def __init__(self, model, scaler, feed_concs_dict, species_cols, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device
        self.species_cols = species_cols
        
        # 构建 Feed Concentration 向量 (25,)
        self.feed_conc_vec = np.zeros(len(species_cols))
        for i, col in enumerate(species_cols):
            met_name = col.replace('Conc_', '')
            if met_name not in ['Xv', 'mAb', 'Conc_mAb']:
                self.feed_conc_vec[i] = feed_concs_dict.get(met_name, 0.0)
        
    def simulate(self, exp_data):
        """
        执行闭环模拟 (Closed-loop Simulation)
        exp_data: 该实验的所有时间点数据 (DataFrame)
        """
        # 1. 初始化状态 (t=0)
        # 获取初始浓度, 体积, 时间
        # 注意：这里我们只取第一行作为初始条件！
        initial_row = exp_data.iloc[0]
        
        c_curr = initial_row[self.species_cols].values.astype(np.float32) # (25,)
        v_curr = float(initial_row['V_L'])
        t_curr = float(initial_row['Time'])
        
        # 隐藏状态 (LSTM Hidden State)
        hidden = None
        
        # 记录结果
        sim_results = {
            'time': [],
            'conc_pred': [],
            'vol_pred': [],
            'conc_real': exp_data[self.species_cols].values,
            'vol_real': exp_data['V_L'].values
        }
        
        # 提取操作序列 (Feed/Sample Volumes)
        # 这些是"已知"的未来操作计划
        feed_vols = exp_data['Feed_Vol_L'].values
        sample_vols = exp_data['Sample_Vol_L'].values
        times = exp_data['Time'].values
        
        # --- 闭环时间循环 ---
        # 我们预测从 t=0 到 t=T-1 的变化，从而得到 t=1 到 t=T 的状态
        num_steps = len(exp_data)
        
        # 记录 t=0
        sim_results['time'].append(t_curr)
        sim_results['conc_pred'].append(c_curr)
        sim_results['vol_pred'].append(v_curr)
        
        with torch.no_grad():
            for t in range(num_steps - 1):
                # A. 准备神经网络输入
                # Input = [Conc(25), Vol(1), Time(1)]
                # 注意：必须使用 scaler 进行归一化，且参数必须是上一步的"预测值"
                input_vec = np.hstack([c_curr, [v_curr], [t_curr]])
                input_scaled = self.scaler.transform(input_vec.reshape(1, -1))
                
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Input Shape: (1, 1, 27)
                
                # B. 神经网络预测 (Predict Reaction)
                # dMr_pred: (1, 1, 25)
                dMr_pred, hidden = self.model(input_tensor, hidden)
                dMr_val = dMr_pred.cpu().numpy().flatten() # (25,)
                
                # C. 物理层更新 (Physics Step - Simulation Mode)
                # 获取当前步的操作量
                f_vol = feed_vols[t+1] # 注意：通常 data 中的 feed vol 记录的是到 t+1 时刻加入的量
                s_vol = sample_vols[t+1]
                dt = times[t+1] - times[t] # 实际上 step1 已处理好，这里作为参考
                
                # 1. 质量平衡 (Mass Balance)
                # Mass_next = Mass_curr + Reaction + Feed - Sample
                # Sample 移除的是当前浓度的液体
                mass_curr = c_curr * v_curr
                mass_in = f_vol * self.feed_conc_vec
                mass_out = s_vol * c_curr
                
                mass_next = mass_curr + dMr_val + mass_in - mass_out
                
                # 2. 体积平衡 (Volume Balance)
                v_next = v_curr + f_vol - s_vol
                
                # 3. 计算新浓度
                c_next = mass_next / v_next
                
                # 4. 更新状态指针
                c_curr = c_next
                v_curr = v_next
                t_curr = times[t+1]
                
                # 记录
                sim_results['time'].append(t_curr)
                sim_results['conc_pred'].append(c_curr)
                sim_results['vol_pred'].append(v_curr)
                
        # 转换为数组
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
    
    # 1. 计算整体 R2 (Global Metric)
    r2_all = r2_score(c_real.flatten(), c_pred.flatten())
    rmse_all = np.sqrt(mean_squared_error(c_real.flatten(), c_pred.flatten()))
    print(f"Exp {exp_id} | Overall R2: {r2_all:.4f} | Overall RMSE: {rmse_all:.4f}")
    
    # 2. 绘图配置 (绘制所有 25 个变量)
    n_plots = len(species_names) # 应该是 25
    cols = 5                     # 固定 5 列
    rows = (n_plots + cols - 1) // cols # 向上取整计算行数 (25/5 = 5行)
    
    # 动态调整画布高度：每行约 3 英寸高，宽度固定 20 英寸
    plt.figure(figsize=(20, 3.5 * rows))
    
    for idx, name in enumerate(species_names):
        # 提取当前变量的数据
        y_real = c_real[:, idx]
        y_pred = c_pred[:, idx]
        
        # 计算单变量指标
        r2 = r2_score(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        
        # 创建子图
        plt.subplot(rows, cols, idx + 1)
        
        # 绘图
        # 实测点：空心黑圈或实心黑点
        plt.plot(time, y_real, 'ko', markersize=4, alpha=0.7, label='Exp Data') 
        # 预测线：蓝色实线
        plt.plot(time, y_pred, 'b-', linewidth=2, label='Hybrid Model') 
        
        # 装饰
        clean_name = name.replace('Conc_', '') # 去掉前缀让标题更短
        plt.title(f"{clean_name}\n$R^2$={r2:.2f} | RMSE={rmse:.2f}", fontsize=10)
        
        # 仅在最后一行添加 X 轴标签，仅在第一列添加 Y 轴标签 (为了整洁)
        if idx >= (rows - 1) * cols:
            plt.xlabel('Time (h)', fontsize=9)
        if idx % cols == 0:
            plt.ylabel('Conc (mM/g/L)', fontsize=9)
            
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # 仅在第一个子图显示图例
        if idx == 0:
            plt.legend(loc='best', fontsize=8, frameon=True)
            
    plt.suptitle(f"Digital Twin Simulation: Experiment {exp_id} (All Species)", fontsize=16, y=1.01)
    plt.tight_layout()
    
    # 保存图片
    file_path = f"{save_dir}/Sim_Exp_{exp_id}_All.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布释放内存
    print(f"Saved plot to {file_path}")

# ==============================================================================
# 3. 主执行流程
# ==============================================================================
if __name__ == "__main__":
    # Settings
    DATA_FILE = 'processed_data_IR_final.csv'
    ORIGINAL_FILE = 'data/data.xlsx'
    S_MATRIX_FILE = 's_matrix.pkl'
    MODEL_FILE = 'hybrid_lstm_model.pth'
    SCALER_FILE = 'scaler.pkl'
    
    # 1. Load Resources
    print("Loading resources...")
    df = pd.read_csv(DATA_FILE)
    feed_concs = get_feed_concs_dict(ORIGINAL_FILE)
    
    with open(S_MATRIX_FILE, 'rb') as f:
        S_matrix = pickle.load(f)
        
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
        
    # Define Columns
    met_list = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glc', 'Gln', 'Glu', 'Pyr', 
                'Gly', 'His', 'Ile', 'Lac', 'Leu', 'Lys', 'Met', 'Nh4', 'Phe', 
                'Pro', 'Ser', 'Thr', 'Tyr', 'Val']
    species_cols = ['Xv', 'Conc_mAb'] + [f'Conc_{m}' for m in met_list]
    
    # 2. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridLSTM(S_matrix, input_dim=27, latent_dim=7, device=device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print("Model loaded.")
    
    # 3. Run Simulation on Test Set
    # Test Exps from paper: 5, 6, 8
    # Val Exp: 7
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
        
    print("\n✅ All simulations completed. Check 'results' folder for plots.")