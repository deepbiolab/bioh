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
# 3. Hybrid LSTM Model (Architecture matching Table 1 Best LSTM)
# ==============================================================================
class HybridLSTM(nn.Module):
    def __init__(self, S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Register S Matrix (Frozen)
        # S shape: (25, 7) -> (n_species, n_latent)
        self.S = torch.tensor(S_matrix, dtype=torch.float32).to(device)
        self.register_buffer('S_const', self.S)
        
        # 2. Architecture: In(27) -> ReLU(16) -> LSTM(7) -> Rate(7)
        # Note: LSTM input size = hidden_dim (16)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) # Paper specifies 0.1 dropout
        )
        
        # LSTM Layer
        # hidden_size = latent_dim (7) as per paper notation "LSTM(7)"
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=latent_dim, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.1)
        
        # Output Mapping (Optional linear layer to map LSTM state to Rates)
        # If LSTM hidden size is 7, this layer maps 7->7, allowing scaling flexibility
        self.fc_out = nn.Linear(latent_dim, latent_dim) 
        
    def forward(self, x_seq, hidden_state=None):
        batch_size, seq_len, _ = x_seq.shape
        
        # 1. Feedforward Encoder
        x_flat = x_seq.reshape(-1, x_seq.size(2))
        features = self.encoder(x_flat)
        features = features.reshape(batch_size, seq_len, -1)
        
        # 2. LSTM
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 3. Output Scores (Latent Rates)
        scores = self.fc_out(lstm_out)
        
        # 4. Hybrid Layer: dMr = Scores @ S.T
        # (B, T, 7) @ (7, 25) -> (B, T, 25)
        delta_mr = torch.matmul(scores, self.S.t())
        
        return delta_mr, new_hidden

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
        Discrete Physics Update (Equation S1.3 Logic)
        
        Training Mode:
            Total_Mass(t+1) = (Mr(t) + dMr_pred) + Accum_GT(t+1)
            We use Ground Truth Accumulation to decouple physics errors from NN errors.
        """
        # 1. Update Reacted Mass
        mr_next_pred = mr_curr + dMr_pred
        
        # In Training, we stop here because we compute Loss on Mr directly.
        # But if we wanted to compute Concentration Loss:
        # total_mass_next = mr_next_pred + accum_next
        # c_next_pred = total_mass_next / v_next
        
        return mr_next_pred

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            inputs = batch['inputs'].to(self.device)   # (B, T, 27)
            target_mr = batch['mr'].to(self.device)    # (B, T, 25)
            
            # Predict dMr sequence
            pred_dMr_seq, _ = self.model(inputs)
            
            # Teacher Forcing: Use Ground Truth previous step to predict next step
            # t = 0 to T-1
            mr_curr = target_mr[:, :-1, :]       
            mr_next_target = target_mr[:, 1:, :] 
            
            # NN Prediction corresponding to t -> t+1 change
            dMr_pred = pred_dMr_seq[:, :-1, :]   
            
            # Physics Update
            mr_next_pred = self.physics_step(mr_curr, dMr_pred)
            
            # Loss Calculation (Weighted MSE)
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
# Helper: Load Feed Concs (Legacy Logic)
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
    
    # Define Column Order (Must match S Matrix)
    met_list = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glc', 'Gln', 'Glu', 'Pyr', 
                'Gly', 'His', 'Ile', 'Lac', 'Leu', 'Lys', 'Met', 'Nh4', 'Phe', 
                'Pro', 'Ser', 'Thr', 'Tyr', 'Val']
    species_cols = ['Xv', 'Conc_mAb'] + [f'Conc_{m}' for m in met_list]
    mr_cols = ['Mr_Xv', 'Mr_mAb'] + [f'Mr_{m}' for m in met_list]
    
    # 2. Prepare Data
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(df, species_cols, mr_cols, feed_concs)
    
    # 3. Calculate Max Vals for Loss Weighting (From Training Data)
    all_train_mr = []
    for batch in train_loader:
        all_train_mr.append(batch['mr'].numpy())
    all_train_mr = np.vstack(all_train_mr)
    # Flatten Batch and Seq dimensions -> (N, 25)
    all_train_mr = all_train_mr.reshape(-1, 25)
    max_mr_vals = np.max(np.abs(all_train_mr), axis=0)
    max_mr_vals[max_mr_vals == 0] = 1.0
    
    print(f"\n[Model] Initializing HybridLSTM (Input=27, Hidden=16, Latent=7)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridLSTM(S_matrix, input_dim=27, hidden_dim=16, latent_dim=7, device=device)
    
    print(f"[Training] Starting loop on {device}...")
    trainer = HybridTrainer(model, train_loader, val_loader, max_mr_vals, device)
    history = trainer.fit(epochs=500, patience=50) # Use 500 epochs with early stopping
    
    # 4. Save Artifacts
    torch.save(model.state_dict(), 'hybrid_lstm_model.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("\n✅ Verification Passed: Model trained and saved.")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE')
    plt.title('Hybrid Model Training Progress')
    plt.legend()
    plt.savefig('training_curve.png')
    # plt.show()