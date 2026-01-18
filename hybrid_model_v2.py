import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import copy
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

# ==============================================================================
# 0. 配置与命名定义 (Nomenclature Configuration)
# ==============================================================================
class ExperimentConfig:
    def __init__(self):
        # X: State Variables (Concentrations/Counts) - 参与反应和Mass Balance
        self.x_cols = ['VCD', 'product', 'Glc', 'Gln', 'Amm', 'Lac']
        
        # W: Controlled Independent Variables - 作为NN输入，但不参与Mass Balance
        self.w_cols = ['Cmd_Temp', 'Cmd_pH', 'Cmd_Stirring']
        
        # F: Feed/Flow Variables - 直接影响Mass Balance
        # 映射关系: Feed_Column_Name -> Target_X_Name
        self.f_mapping = {
            'Cmd_Feed_Glc_Mass': 'Glc',
            'Cmd_Feed_Gln_Mass': 'Gln'
            # Cmd_Feed_Vol 专门处理体积，不在此映射
        }
        self.vol_col = 'Volume'
        self.feed_vol_col = 'Cmd_Feed_Vol'
        self.time_col = 'time[h]'
        self.group_cols = ['Product_ID', 'Exp_ID'] # 联合主键

    @property
    def input_dim(self):
        # NN Input = X(State) + V(1) + T(1) + W(Controls)
        return len(self.x_cols) + 1 + 1 + len(self.w_cols)

# ==============================================================================
# 1. 数据预处理 (Preprocessing Logic)
# ==============================================================================
def preprocess_novel_data(csv_file, config):
    print(f"Loading and preprocessing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 生成唯一实验标识 (Combine Product_ID and Exp_ID)
    df['Run_ID'] = df[config.group_cols[0]].astype(str) + "_" + df[config.group_cols[1]].astype(str)
    
    unique_runs = df['Run_ID'].unique()
    processed_dfs = []
    
    for run_id in unique_runs:
        group = df[df['Run_ID'] == run_id].sort_values(config.time_col).reset_index(drop=True)
        
        # 1. 提取基础物理量
        # 假设 csv 中的单位已经是标准单位 (e.g., L, g, mmol)
        # 如果需要单位转换，请在此处添加
        V = group[config.vol_col].values 
        
        # 2. 计算 Accum (物理累积量)
        # Accum(t) = Sum(Feed_Mass_in_History)
        # 对于 Glc 和 Gln:
        accum_dict = {col: np.zeros(len(group)) for col in config.x_cols}
        
        # 处理有显式 Feed 的物质
        for feed_col, target_x in config.f_mapping.items():
            # 计算累积加料量 (Cumulative Sum)
            # 假设 feed_col 记录的是每个时间步加入的 *增量* (Mass)
            feed_mass_incr = group[feed_col].values
            accum_dict[target_x] = np.cumsum(feed_mass_incr)
            # 注意：如果第一行数据的 Feed 已经是加入后的，cumsum 逻辑是对的
            # 如果第一行是 0，且 Feed 在 t->t+1 发生，则 Accum(t) 应当包含直到 t 时刻的累积
            # 这里简单处理为 cumsum，具体视数据定义微调
            
        # 3. 计算 Mr (Reacted Mass)
        # Mr = V * C - Accum
        for x_col in config.x_cols:
            conc = group[x_col].values
            accum = accum_dict[x_col]
            
            total_mass = V * conc
            mr = total_mass - accum
            
            group[f'Accum_{x_col}'] = accum
            group[f'Mr_{x_col}'] = mr
            
        processed_dfs.append(group)
        
    df_final = pd.concat(processed_dfs, ignore_index=True)
    return df_final

# ==============================================================================
# 2. S 矩阵构建 (PCA on Mr)
# ==============================================================================
def build_s_matrix(df, config, n_components=4):
    mr_cols = [f'Mr_{c}' for c in config.x_cols]
    X_mr = df[mr_cols].values
    
    # Normalization (Max Abs)
    max_vals = np.max(np.abs(X_mr), axis=0)
    max_vals[max_vals == 0] = 1.0
    X_norm = X_mr / max_vals
    
    # Truncated SVD (Centered=False)
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(X_norm)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"S Matrix Explained Variance: {explained_var:.4%}")
    
    S_matrix = pca.components_.T * max_vals[:, np.newaxis]
    return S_matrix, max_vals

# ==============================================================================
# 3. Dataset
# ==============================================================================
class BioreactorDataset(Dataset):
    def __init__(self, df, config, run_ids, scaler=None):
        self.data_list = []
        self.config = config
        
        for run in run_ids:
            group = df[df['Run_ID'] == run].sort_values(config.time_col)
            
            # --- Inputs (X, V, T, W) ---
            x_data = group[config.x_cols].values.astype(np.float32)
            v_data = group[config.vol_col].values.astype(np.float32).reshape(-1, 1)
            t_data = group[config.time_col].values.astype(np.float32).reshape(-1, 1)
            w_data = group[config.w_cols].values.astype(np.float32)
            
            inputs = np.hstack([x_data, v_data, t_data, w_data])
            if scaler:
                inputs = scaler.transform(inputs)
                
            # --- Targets & Physics ---
            mr_cols = [f'Mr_{c}' for c in config.x_cols]
            accum_cols = [f'Accum_{c}' for c in config.x_cols]
            
            mr_data = group[mr_cols].values.astype(np.float32)
            accum_data = group[accum_cols].values.astype(np.float32)
            
            # Feeds (F) for Simulation
            # 创建一个矩阵 (Seq, n_species) 存储每一步的 Feed Mass 增量
            feed_mass_matrix = np.zeros_like(x_data)
            for feed_col, target_x in config.f_mapping.items():
                idx = config.x_cols.index(target_x)
                feed_mass_matrix[:, idx] = group[feed_col].values
            
            # Feed Vol (Volume change)
            feed_vol_data = group[config.feed_vol_col].values.astype(np.float32).reshape(-1, 1)

            self.data_list.append({
                'inputs': torch.tensor(inputs),
                'mr': torch.tensor(mr_data),
                'accum': torch.tensor(accum_data), # Ground Truth Accum
                'conc': torch.tensor(x_data),
                'vol': torch.tensor(v_data),
                'time': torch.tensor(t_data),
                'feed_mass': torch.tensor(feed_mass_matrix, dtype=torch.float32), # F (Mass)
                'feed_vol': torch.tensor(feed_vol_data), # F (Vol)
                'w_controls': torch.tensor(w_data) # W
            })
            
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def prepare_dataloaders(df, config):
    # Split by Run_ID
    all_runs = df['Run_ID'].unique()
    np.random.shuffle(all_runs)
    
    n = len(all_runs)
    train_runs = all_runs[:int(n*0.7)]
    val_runs = all_runs[int(n*0.7):int(n*0.85)]
    test_runs = all_runs[int(n*0.85):]
    
    print(f"Split: Train={len(train_runs)}, Val={len(val_runs)}, Test={len(test_runs)}")
    
    # Scaler
    train_df = df[df['Run_ID'].isin(train_runs)]
    x_data = train_df[config.x_cols].values
    v_data = train_df[config.vol_col].values.reshape(-1, 1)
    t_data = train_df[config.time_col].values.reshape(-1, 1)
    w_data = train_df[config.w_cols].values
    
    all_inputs = np.hstack([x_data, v_data, t_data, w_data])
    scaler = StandardScaler()
    scaler.fit(all_inputs)
    
    # Datasets
    train_ds = BioreactorDataset(df, config, train_runs, scaler)
    val_ds = BioreactorDataset(df, config, val_runs, scaler)
    test_ds = BioreactorDataset(df, config, test_runs, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, (train_runs, val_runs, test_runs)

# ==============================================================================
# 4. Models & Training (Hybrid Structure)
# ==============================================================================
class HybridLSTM(nn.Module):
    def __init__(self, S_matrix, input_dim, hidden_dim=16, latent_dim=4, device='cpu'):
        super().__init__()
        self.device = device
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=latent_dim, batch_first=True)
        self.fc_out = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x_seq, hidden_state=None):
        batch, seq, _ = x_seq.shape
        feat = self.encoder(x_seq.reshape(-1, x_seq.size(2))).reshape(batch, seq, -1)
        out, hidden = self.lstm(feat, hidden_state)
        scores = self.fc_out(out)
        dMr = torch.matmul(scores, self.S.t())
        return dMr, hidden

class HybridFFNN(nn.Module):
    def __init__(self, S_matrix, input_dim, latent_dim=4, device='cpu'):
        super().__init__()
        self.device = device
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(16, 16), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(16, latent_dim)
        )
    def forward(self, x_seq, hidden=None):
        scores = self.net(x_seq)
        dMr = torch.matmul(scores, self.S.t())
        return dMr, None

class WeightedMSELoss(nn.Module):
    def __init__(self, max_vals, device):
        super().__init__()
        self.weights = 1.0 / (torch.tensor(max_vals, dtype=torch.float32).to(device) ** 2)
        self.weights = torch.clamp(self.weights, max=1e6)
    def forward(self, pred, target):
        return torch.mean(((pred - target) ** 2) * self.weights)

class HybridTrainer:
    def __init__(self, model, train_loader, val_loader, max_mr_vals, device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = WeightedMSELoss(max_mr_vals, device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            inputs = batch['inputs'].to(self.device)
            target_mr = batch['mr'].to(self.device)
            
            dMr_pred, _ = self.model(inputs)
            
            # Teacher Forcing: Pred(t+1) = GT(t) + dMr(t)
            mr_curr = target_mr[:, :-1, :]
            mr_next_target = target_mr[:, 1:, :]
            dMr_curr_pred = dMr_pred[:, :-1, :]
            
            mr_next_pred = mr_curr + dMr_curr_pred
            
            loss = self.criterion(mr_next_pred, mr_next_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['inputs'].to(self.device)
                target_mr = batch['mr'].to(self.device)
                dMr_pred, _ = self.model(inputs)
                mr_next_pred = target_mr[:, :-1, :] + dMr_pred[:, :-1, :]
                val_loss += self.criterion(mr_next_pred, target_mr[:, 1:, :]).item()
        return val_loss / len(self.val_loader)

    def fit(self, epochs=200, patience=20):
        best_loss = float('inf')
        cnt = 0
        for ep in range(epochs):
            tl = self.train_epoch()
            vl = self.validate()
            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vl)
            if ep % 10 == 0: print(f"Ep {ep}: Train {tl:.4f}, Val {vl:.4f}")
            
            if vl < best_loss:
                best_loss = vl
                cnt = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                cnt += 1
                if cnt >= patience: break
        return self.history

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    MODEL_TYPE = 'LSTM' 
    DATA_FILE = 'data/hist.csv'
    
    conf = ExperimentConfig()
    
    # 1. Preprocess
    df_processed = preprocess_novel_data(DATA_FILE, conf)
    
    # 2. S Matrix
    S, max_mr_vals = build_s_matrix(df_processed, conf, n_components=4)
    with open('s_matrix.pkl', 'wb') as f: pickle.dump(S, f)
    
    # 3. DataLoaders
    train_dl, val_dl, test_dl, scaler, splits = prepare_dataloaders(df_processed, conf)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('splits.pkl', 'wb') as f: pickle.dump(splits, f)
    
    # 4. Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Input Dim: {conf.input_dim}, S shape: {S.shape}")
    
    if MODEL_TYPE == 'LSTM':
        model = HybridLSTM(S, conf.input_dim, device=device)
    else:
        model = HybridFFNN(S, conf.input_dim, device=device)
        
    trainer = HybridTrainer(model, train_dl, val_dl, max_mr_vals, device)
    trainer.fit(epochs=500, patience=30)