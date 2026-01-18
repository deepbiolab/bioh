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

# ==============================================================================
# 1. Dataset & Dataloader
# ==============================================================================
class BioreactorDataset(Dataset):
    def __init__(self, df, species_cols, mr_cols, accum_cols, exp_ids, feed_concs_dict, scaler=None):
        self.data_list = []
        
        # 构建 Feed Concentration 向量 (25,)
        # 顺序严格对应 species_cols
        feed_conc_vec = []
        for col in species_cols:
            met_name = col.replace('Conc_', '')
            if met_name in ['Xv', 'mAb', 'Conc_mAb']:
                feed_conc_vec.append(0.0)
            else:
                feed_conc_vec.append(feed_concs_dict.get(met_name, 0.0))
        self.feed_conc_tensor = torch.tensor(feed_conc_vec, dtype=torch.float32)
        
        for exp in exp_ids:
            group = df[df['Experiment'] == exp].sort_values('Time')
            
            # 1. Core Data
            conc = group[species_cols].values.astype(np.float32)
            mr = group[mr_cols].values.astype(np.float32)
            accum = group[accum_cols].values.astype(np.float32) # Ground Truth Physics
            
            vol = group['V_L'].values.astype(np.float32).reshape(-1, 1)
            time = group['Time'].values.astype(np.float32).reshape(-1, 1)
            
            # 2. Simulation Inputs (Feed/Sample Increments)
            # 这里的 Feed_Vol_L 是 process_bioprocess_data 中显式保存的列
            feed_vol = group['Feed_Vol_L'].values.astype(np.float32).reshape(-1, 1)
            sample_vol = group['Sample_Vol_L'].values.astype(np.float32).reshape(-1, 1)
            
            # 3. Model Input: [Conc(25), Vol(1), Time(1)] -> 27 Features
            inputs = np.hstack([conc, vol, time])
            if scaler:
                inputs = scaler.transform(inputs)
            
            self.data_list.append({
                'inputs': torch.tensor(inputs),      # (Seq, 27)
                'mr': torch.tensor(mr),              # (Seq, 25)
                'accum': torch.tensor(accum),        # (Seq, 25)
                'conc': torch.tensor(conc),          # (Seq, 25)
                'vol': torch.tensor(vol),            # (Seq, 1)
                'time': torch.tensor(time),          # (Seq, 1)
                'feed_vol': torch.tensor(feed_vol),  # (Seq, 1)
                'sample_vol': torch.tensor(sample_vol), # (Seq, 1)
                'feed_conc': self.feed_conc_tensor   # (25,)
            })
            
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def prepare_dataloaders(df, species_cols, mr_cols, feed_concs_dict):
    print(f"\n[Data] Preparing Datasets...")
    
    # 0. Auto-generate Accum columns
    accum_cols = []
    for col in species_cols:
        if col == 'Xv': accum_cols.append('Accum_Xv')
        elif col == 'Conc_mAb': accum_cols.append('Accum_mAb')
        else: accum_cols.append(col.replace('Conc_', 'Accum_'))
    
    # Validation
    missing = [c for c in accum_cols if c not in df.columns]
    if missing: raise ValueError(f"Missing Accum columns in DataFrame: {missing}")

    # 1. Split (Paper: Train 12, Val 4, Test 4 - but we have 9 total)
    # Using specific split from Matlab code: Train=[1,2,3,4,9], Val=[7], Test=[5,6,8]
    train_exps = [1, 2, 3, 4, 9]
    val_exps = [7]
    test_exps = [5, 6, 8]
    
    # 2. Fit Scaler (Train only)
    train_df = df[df['Experiment'].isin(train_exps)]
    input_data = []
    for exp in train_exps:
        group = train_df[train_df['Experiment'] == exp].sort_values('Time')
        conc = group[species_cols].values
        vol = group['V_L'].values.reshape(-1, 1)
        time = group['Time'].values.reshape(-1, 1)
        input_data.append(np.hstack([conc, vol, time]))
    
    scaler = StandardScaler()
    scaler.fit(np.vstack(input_data))
    
    # 3. Build Datasets
    train_dataset = BioreactorDataset(df, species_cols, mr_cols, accum_cols, train_exps, feed_concs_dict, scaler)
    val_dataset = BioreactorDataset(df, species_cols, mr_cols, accum_cols, val_exps, feed_concs_dict, scaler)
    test_dataset = BioreactorDataset(df, species_cols, mr_cols, accum_cols, test_exps, feed_concs_dict, scaler)
    
    # 4. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

# ==============================================================================
# 2. Loss Function (Weighted MSE)
# ==============================================================================
class WeightedMSELoss(nn.Module):
    def __init__(self, max_vals, device):
        super().__init__()
        # Paper S1.3: WMSE = Sum ( (y - y_hat)^2 / sigma^2 )
        # Matlab Code: sigma ~= max_val (Normalized MSE)
        # We use 1 / (max_val)^2 as weights
        weights_np = 1.0 / (max_vals ** 2)
        self.weights = torch.tensor(weights_np, dtype=torch.float32).to(device)
        self.weights = torch.clamp(self.weights, max=1e6) # Prevent instability
        
    def forward(self, pred, target):
        # pred, target: (Batch, Seq, 25)
        diff = (pred - target) ** 2
        weighted_diff = diff * self.weights
        return torch.mean(weighted_diff)

# ==============================================================================
# 3. Hybrid Models (LSTM & FFNN)
# ==============================================================================
class HybridLSTM(nn.Module):
    """
    Best LSTM Architecture: In(27)-ReLU(16)-LSTM(7)-Rate(7)
    """
    def __init__(self, S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Register S Matrix (Frozen)
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        # 2. Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 3. LSTM Layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=latent_dim, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.1)
        
        # 4. Output Mapping
        self.fc_out = nn.Linear(latent_dim, latent_dim) 
        
    def forward(self, x_seq, hidden_state=None):
        batch_size, seq_len, _ = x_seq.shape
        
        x_flat = x_seq.reshape(-1, x_seq.size(2))
        features = self.encoder(x_flat)
        features = features.reshape(batch_size, seq_len, -1)
        
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        lstm_out = self.lstm_dropout(lstm_out)
        
        scores = self.fc_out(lstm_out)
        
        # Hybrid Layer: dMr = Scores @ S.T
        delta_mr = torch.matmul(scores, self.S.t())
        
        return delta_mr, new_hidden

class HybridFFNN(nn.Module):
    """
    Best FFNN Architecture: In(27)-Tanh(8)-Tanh(8)-Tanh(8)-Tanh(7)-Rate(7)
    """
    def __init__(self, S_matrix, input_dim=27, latent_dim=7, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Register S Matrix (Frozen)
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        # 2. Architecture (4 Tanh hidden layers)
        # Layer dims: 27 -> 8 -> 8 -> 8 -> 7 -> 7
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(8, 7), # Corresponds to Tanh(7) layer
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(7, latent_dim) # Rate(7) Output Layer
        )
        
    def forward(self, x_seq, hidden_state=None):
        # FFNN treats sequence as batch of independent time points
        batch_size, seq_len, _ = x_seq.shape
        
        x_flat = x_seq.reshape(-1, x_seq.size(2))
        scores_flat = self.net(x_flat)
        scores = scores_flat.reshape(batch_size, seq_len, -1)
        
        # Hybrid Layer: dMr = Scores @ S.T
        delta_mr = torch.matmul(scores, self.S.t())
        
        # Return None for hidden state compatibility
        return delta_mr, None

# ==============================================================================
# Helper for Transformer: Positional Encoding
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵 (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        # 添加位置编码 (Broadcasting)
        x = x + self.pe[:, :x.size(1), :]
        return x

class HybridTransformer(nn.Module):
    """
    Lightweight Transformer: 
    Input(27) -> Linear(16) -> PosEnc -> TransformerEncoder(d_model=16, nhead=2) -> Linear(7)
    参数量设计对标 LSTM (~1.5k params)
    """
    def __init__(self, S_matrix, input_dim=27, latent_dim=7, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. 冻结 S 矩阵
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        # 参数配置 (旨在保持轻量级)
        self.d_model = 32  # 嵌入维度 (对应 LSTM hidden_dim)
        nhead = 4          # 多头注意力头数
        num_layers = 2     # Encoder 层数 (浅层网络)
        dim_feedforward = 32 # FFN 内部维度 (通常是 d_model 的 2-4 倍)
        
        # 2. Input Embedding
        self.embedding = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(0.1)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Projection
        self.fc_out = nn.Linear(self.d_model, latent_dim)
        
    def _generate_square_subsequent_mask(self, sz):
        # 生成因果掩码 (Causal Mask)
        # 确保 t 时刻的预测只能看到 0...t 时刻的数据，不能看到未来
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, x_seq, hidden_state=None):
        # x_seq: (Batch, Seq_Len, 27)
        batch_size, seq_len, _ = x_seq.shape
        
        # Embedding & Positional Encoding
        x = self.embedding(x_seq) # (B, S, 16)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Causal Masking (关键：防止这一步看到未来的数据)
        mask = self._generate_square_subsequent_mask(seq_len)
        
        # Transformer Forward
        # Output: (B, S, 16)
        x_trans = self.transformer_encoder(x, mask=mask)
        
        # Output Projection -> Scores
        scores = self.fc_out(x_trans) # (B, S, 7)
        
        # Hybrid Layer: dMr = Scores @ S.T
        delta_mr = torch.matmul(scores, self.S.t())
        
        # Transformer 是无状态的 (Stateless)，hidden_state 仅为保持接口一致返回 None
        return delta_mr, None

# ==============================================================================
# 4. Physics-Informed Trainer
# ==============================================================================
class HybridTrainer:
    def __init__(self, model, train_loader, val_loader, max_mr_vals, device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Paper uses Adam with defaults
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = WeightedMSELoss(max_mr_vals, device)
        
        self.history = {'train_loss': [], 'val_loss': []}

    def physics_step(self, mr_curr, dMr_pred, accum_next=None):
        """
        Discrete Physics Update
        """
        mr_next_pred = mr_curr + dMr_pred
        return mr_next_pred

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs = batch['inputs'].to(self.device)   # (B, T, 27)
            target_mr = batch['mr'].to(self.device)    # (B, T, 25)
            
            # Predict dMr sequence
            pred_dMr_seq, _ = self.model(inputs)
            
            # Teacher Forcing
            mr_curr = target_mr[:, :-1, :]       
            mr_next_target = target_mr[:, 1:, :] 
            dMr_pred = pred_dMr_seq[:, :-1, :]   
            
            # Physics Update
            mr_next_pred = self.physics_step(mr_curr, dMr_pred)
            
            # Loss Calculation
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
                
                pred_dMr_seq, _ = self.model(inputs)
                
                mr_curr = target_mr[:, :-1, :]
                mr_next_target = target_mr[:, 1:, :]
                dMr_pred = pred_dMr_seq[:, :-1, :]
                
                mr_next_pred = self.physics_step(mr_curr, dMr_pred)
                loss = self.criterion(mr_next_pred, mr_next_target)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def fit(self, epochs=200, patience=20):
        print(f"Starting training on {self.device}...")
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        self.model.load_state_dict(best_model_wts)
        return self.history

# ==============================================================================
# Helper: Load Feed Concs
# ==============================================================================
def get_feed_concs_dict(file_path):
    df_raw = pd.read_excel(file_path, sheet_name='feed conc')
    names = df_raw.columns.values
    values = df_raw.iloc[1].values
    target_mets = {'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glc', 'Gln', 'Glu', 'Pyr', 
                   'Gly', 'His', 'Ile', 'Lac', 'Leu', 'Lys', 'Met', 'Nh4', 'Phe', 
                   'Pro', 'Ser', 'Thr', 'Tyr', 'Val'}
    feed_concs = {}
    for n, v in zip(names, values):
        if isinstance(n, str) and n.strip() in target_mets:
            try: feed_concs[n.strip()] = float(v)
            except: pass
    return feed_concs

# ==============================================================================
# 5. Execution Block
# ==============================================================================
if __name__ == "__main__":
    # --- MODEL TYPE PARAMETER ---
    MODEL_TYPE = 'LSTM' # Options: 'LSTM', 'FFNN', 'Transformer'
    # ----------------------------

    # Settings
    DATA_FILE = 'processed_data_IR_final.csv'
    ORIGINAL_FILE = 'data/data.xlsx'
    S_MATRIX_FILE = 's_matrix.pkl'
    
    # 1. Load Resources
    if not os.path.exists(DATA_FILE) or not os.path.exists(S_MATRIX_FILE):
        print("Error: Files missing. Run Step 1 & 2 first.")
        exit()
        
    df = pd.read_csv(DATA_FILE)
    with open(S_MATRIX_FILE, 'rb') as f:
        S_matrix = pickle.load(f)
    
    feed_concs = get_feed_concs_dict(ORIGINAL_FILE)
    
    met_list = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glc', 'Gln', 'Glu', 'Pyr', 
                'Gly', 'His', 'Ile', 'Lac', 'Leu', 'Lys', 'Met', 'Nh4', 'Phe', 
                'Pro', 'Ser', 'Thr', 'Tyr', 'Val']
    species_cols = ['Xv', 'Conc_mAb'] + [f'Conc_{m}' for m in met_list]
    mr_cols = ['Mr_Xv', 'Mr_mAb'] + [f'Mr_{m}' for m in met_list]
    
    # 2. Prepare Data
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(df, species_cols, mr_cols, feed_concs)
    
    # 3. Calculate Max Vals
    all_train_mr = []
    for batch in train_loader:
        all_train_mr.append(batch['mr'].numpy())
    all_train_mr = np.vstack(all_train_mr)
    all_train_mr = all_train_mr.reshape(-1, 25)
    max_mr_vals = np.max(np.abs(all_train_mr), axis=0)
    max_mr_vals[max_mr_vals == 0] = 1.0
    
    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model] Initializing Hybrid{MODEL_TYPE}...")
    
    if MODEL_TYPE == 'LSTM':
        model = HybridLSTM(S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device=device)
        save_name = 'hybrid_lstm_model.pth'
    elif MODEL_TYPE == 'FFNN':
        model = HybridFFNN(S_matrix, input_dim=27, latent_dim=7, device=device)
        save_name = 'hybrid_ffnn_model.pth'
    elif MODEL_TYPE == 'Transformer':
        model = HybridTransformer(S_matrix, input_dim=27, latent_dim=7, device=device)
        save_name = 'hybrid_transformer_model.pth'        
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    print(f"[Training] Starting loop on {device}...")
    trainer = HybridTrainer(model, train_loader, val_loader, max_mr_vals, device)
    history = trainer.fit(epochs=500, patience=50) 
    
    # 5. Save
    torch.save(model.state_dict(), save_name)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\n✅ Verification Passed: {MODEL_TYPE} Model trained and saved to {save_name}.")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE')
    plt.title(f'Hybrid {MODEL_TYPE} Training Progress')
    plt.legend()
    plt.savefig('training_curve.png')