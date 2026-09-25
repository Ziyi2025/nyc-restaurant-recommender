import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
import random
import os
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 配置 ====================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA = os.path.join(DATA_DIR, "allcuisine.csv")
MODEL_DIR = os.path.join(DATA_DIR, "final_models_5feat")
os.makedirs(MODEL_DIR, exist_ok=True)

HIDDEN_DIM = 64
LEARNING_RATE = 0.01
SEEDS = [42, 123, 456, 789, 2024]
EARTH_RADIUS = 6371000
NEIGHBOR_THRESHOLD_RATIO = 2.0

# 20种菜系 + 网格大小配置（和你之前跑的一致）
CUISINE_GRID_CONFIG = {
    "American": 0.005,
    "Chinese": 0.005,
    "Pizza": 0.005,
    "Coffee/Tea": 0.005,
    "Mexican": 0.005,
    "Bakery Products/Desserts": 0.005,
    "Chicken": 0.005,
    "Latin American": 0.005,
    "Donuts": 0.005,
    "Italian": 0.005,
    "Japanese": 0.005,
    "Caribbean": 0.005,
    "Hamburgers": 0.005,
    "Sandwiches": 0.005,
    "Juice, Smoothies, Fruit Salads": 0.005,
    "Spanish": 0.008,
    "Asian/Asian Fusion": 0.008,
    "Thai": 0.008,
    "Indian": 0.008,
    "Korean": 0.01,
}

# ==================== 加载数据 ====================
print("=" * 60)
print("Step 1: 加载最新数据")
print("=" * 60)

df_all = pd.read_csv(MAIN_DATA)
df_all = df_all.sort_values('INSPECTION DATE', ascending=False).drop_duplicates(subset=['CAMIS'], keep='first')

df_all['Latitude'] = pd.to_numeric(df_all['Latitude'], errors='coerce')
df_all['Longitude'] = pd.to_numeric(df_all['Longitude'], errors='coerce')

# 过滤缺失值和0
df_all = df_all.dropna(subset=['Latitude', 'Longitude'])
df_all = df_all[(df_all['Latitude'] != 0) & (df_all['Longitude'] != 0)]
df_all = df_all.dropna(subset=['CUISINE DESCRIPTION'])

print(f"✅ 全量餐厅: {len(df_all)}")
print(f"   最新检查日期: {df_all['INSPECTION DATE'].max()}")

# ==================== GCN模型 ====================
class GridGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1).numpy()
        true = data.y.numpy()
        mask_np = mask.numpy()
        pred_m, true_m = pred[mask_np], true[mask_np]
        if len(np.unique(true_m)) < 2:
            return None, None, None, None, None
        return (accuracy_score(true_m, pred_m), precision_score(true_m, pred_m, zero_division=0),
                recall_score(true_m, pred_m, zero_division=0), f1_score(true_m, pred_m, zero_division=0),
                roc_auc_score(true_m, pred_m))

# ==================== 对每个菜系重新训练 ====================
all_results = []

for target, grid_size in CUISINE_GRID_CONFIG.items():
    print(f"\n{'='*70}")
    print(f"🍕 {target} | 网格: {grid_size}")
    print('='*70)

    # 网格划分
    df_all['grid_x'] = (df_all['Longitude'] / grid_size).astype(int)
    df_all['grid_y'] = (df_all['Latitude'] / grid_size).astype(int)
    df_all['grid_id'] = df_all['grid_x'].astype(str) + '_' + df_all['grid_y'].astype(str)

    # 网格聚合
    grid_info = df_all.groupby('grid_id').agg(
        center_lat=('Latitude', 'mean'),
        center_lon=('Longitude', 'mean'),
        n_restaurants=('CAMIS', 'count'),
        avg_score=('SCORE', lambda x: pd.to_numeric(x, errors='coerce').mean()),
    ).reset_index()
    grid_info['avg_score'] = grid_info['avg_score'].fillna(0)

    target_col = 'n_target'
    counts = df_all[df_all['CUISINE DESCRIPTION'] == target].groupby('grid_id').size().reset_index(name=target_col)
    grid_info = grid_info.merge(counts, on='grid_id', how='left')
    grid_info[target_col] = grid_info[target_col].fillna(0).astype(int)

    diversity = df_all.groupby('grid_id')['CUISINE DESCRIPTION'].nunique().reset_index()
    diversity.columns = ['grid_id', 'cuisine_diversity']
    grid_info = grid_info.merge(diversity, on='grid_id')

    grid_info['label'] = (grid_info[target_col] > 0).astype(int)
    n_pos = grid_info['label'].sum()
    print(f"  网格数: {len(grid_info)}, 正样本: {n_pos} ({n_pos/len(grid_info)*100:.1f}%)")

    if n_pos < 50:
        print(f"  ⚠️ 正样本不足，跳过")
        continue

    feature_cols = ['n_restaurants', 'avg_score', 'cuisine_diversity', target_col]
    features = grid_info[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X = torch.tensor(features_scaled, dtype=torch.float32)
    y = torch.tensor(grid_info['label'].values, dtype=torch.int64)

    # 邻接图
    grid_coords = np.radians(grid_info[['center_lat', 'center_lon']].values)
    tree = BallTree(grid_coords, metric='haversine')
    neighbor_threshold = grid_size * NEIGHBOR_THRESHOLD_RATIO * 111000
    neighbor_indices = tree.query_radius(grid_coords, r=neighbor_threshold/EARTH_RADIUS)

    edge_index_list = []
    for i, idx in enumerate(neighbor_indices):
        for j in idx:
            if i < j:
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # 划分
    COARSE_GRID = 0.02
    grid_info['coarse_x'] = (grid_info['center_lon'] / COARSE_GRID).astype(int)
    grid_info['coarse_y'] = (grid_info['center_lat'] / COARSE_GRID).astype(int)
    grid_info['coarse_id'] = grid_info['coarse_x'].astype(str) + '_' + grid_info['coarse_y'].astype(str)

    all_coarse = grid_info['coarse_id'].unique()
    train_coarse, temp_coarse = train_test_split(all_coarse, test_size=0.4, random_state=42)
    val_coarse, test_coarse = train_test_split(temp_coarse, test_size=0.5, random_state=42)

    grid_info['split'] = 'train'
    grid_info.loc[grid_info['coarse_id'].isin(val_coarse), 'split'] = 'val'
    grid_info.loc[grid_info['coarse_id'].isin(test_coarse), 'split'] = 'test'

    train_mask = torch.tensor((grid_info['split'] == 'train').values, dtype=torch.bool)
    test_mask = torch.tensor((grid_info['split'] == 'test').values, dtype=torch.bool)

    n_pos_train = (y[train_mask] == 1).sum().item()
    n_neg_train = (y[train_mask] == 0).sum().item()
    pos_weight = n_neg_train / n_pos_train if n_pos_train > 0 else 1.0
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)

    data = Data(x=X, edge_index=edge_index, y=y)
    in_channels = len(feature_cols)

    # 多seed训练
    seed_results = []
    for seed in SEEDS:
        set_seed(seed)
        model = GridGCN(in_channels=in_channels, hidden_channels=HIDDEN_DIM, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weights)
            loss.backward()
            optimizer.step()
        acc, prec, rec, f1, auc = evaluate(model, data, test_mask)
        seed_results.append({'acc': acc, 'f1': f1, 'auc': auc})

    seed_df = pd.DataFrame(seed_results)
    print(f"  Test: Acc={seed_df['acc'].mean():.4f}, F1={seed_df['f1'].mean():.4f}, AUC={seed_df['auc'].mean():.4f}")

    all_results.append({
        'cuisine': target,
        'grid_size': grid_size,
        'n_grids': len(grid_info),
        'n_pos': n_pos,
        'pos_ratio': n_pos/len(grid_info),
        'test_acc': seed_df['acc'].mean(),
        'test_f1': seed_df['f1'].mean(),
        'test_auc': seed_df['auc'].mean(),
        'test_auc_std': seed_df['auc'].std(),
    })

    # 用固定seed训练最终模型
    set_seed(42)
    final_model = GridGCN(in_channels=in_channels, hidden_channels=HIDDEN_DIM, out_channels=2)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(300):
        final_model.train()
        optimizer.zero_grad()
        out = final_model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = final_model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    grid_info['suitability_score'] = probs * 100

    # 保存评分文件
    score_path = os.path.join(DATA_DIR, f"final_scores_{target.replace('/', '_').replace(' ', '_')}.csv")
    grid_info[['grid_id', 'center_lat', 'center_lon', 'n_restaurants', 'avg_score',
               'cuisine_diversity', target_col, 'suitability_score', 'label']].to_csv(score_path, index=False)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, f"final_model_{target.replace('/', '_').replace(' ', '_')}.pth")
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'target_cuisine': target,
        'grid_size': grid_size,
        'feature_cols': feature_cols,
        'hidden_dim': HIDDEN_DIM,
        'in_channels': in_channels,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    }, model_path)

# ==================== 汇总 ====================
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(DATA_DIR, "retrain_results.csv"), index=False)

# 保存训练日期
with open(os.path.join(DATA_DIR, "model_trained_date.txt"), "w") as f:
    f.write(pd.Timestamp.now().strftime('%Y-%m-%d'))

print("\n" + "=" * 70)
print("🎉 所有菜系重训完成！")
print("=" * 70)
print(results_df.to_string(index=False))
