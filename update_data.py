import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta
import os
import shutil

# 配置
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA_PATH = os.path.join(DATA_DIR, "allcuisine.csv")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
UPDATE_LOG = os.path.join(DATA_DIR, "update_log.csv")

# NYC Open Data 配置
DATASET_ID = "43nn-pn8j"
DOMAIN = "data.cityofnewyork.us"

UPDATE_WINDOW_DAYS = 30

os.makedirs(BACKUP_DIR, exist_ok=True)

# 加载现有数据
print("=" * 60)
print("Step 1: 加载现有数据")
print("=" * 60)

if not os.path.exists(MAIN_DATA_PATH):
    print("主数据文件不存在，无法增量更新")
    exit(1)

df_old = pd.read_csv(MAIN_DATA_PATH)
print(f"现有数据: {len(df_old)} 条记录")
print(f"   最新检查日期: {df_old['INSPECTION DATE'].max()}")

# ==================== 从 NYC Open Data 抓取最新数据 ====================
print("\n" + "=" * 60)
print("Step 2: 抓取最新数据")
print("=" * 60)

client = Socrata(DOMAIN, "HRg1MQFPwhvbO9LBHxpSFZUqM")

last_date = pd.to_datetime(df_old['INSPECTION DATE'].max())
start_date = (last_date - timedelta(days=UPDATE_WINDOW_DAYS)).strftime('%Y-%m-%dT00:00:00')

print(f"抓取 {start_date} 之后的数据...")

try:
    results = client.get(
        DATASET_ID,
        where=f"inspection_date > '{start_date}'",
        limit=50000
    )
    df_new = pd.DataFrame.from_records(results)
    print(f"抓取到 {len(df_new)} 条新记录")
except Exception as e:
    print(f"抓取失败: {e}")
    exit(1)

if len(df_new) == 0:
    print("没有新数据需要更新")
    exit(0)

print("\n" + "=" * 60)
print("Step 3: 数据字段对齐")
print("=" * 60)

# 列名映射
column_mapping = {
    'camis': 'CAMIS',
    'dba': 'DBA',
    'boro': 'BORO',
    'building': 'BUILDING',
    'street': 'STREET',
    'zipcode': 'ZIPCODE',
    'phone': 'PHONE',
    'cuisine_description': 'CUISINE DESCRIPTION',
    'inspection_date': 'INSPECTION DATE',
    'action': 'ACTION',
    'violation_code': 'VIOLATION CODE',
    'violation_description': 'VIOLATION DESCRIPTION',
    'critical_flag': 'CRITICAL FLAG',
    'score': 'SCORE',
    'grade': 'GRADE',
    'grade_date': 'GRADE DATE',
    'record_date': 'RECORD DATE',
    'inspection_type': 'INSPECTION TYPE',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'community_board': 'Community Board',
    'council_district': 'Council District',
    'census_tract': 'Census Tract',
    'bin': 'BIN',
    'bbl': 'BBL',
    'nta': 'NTA',
    'location': 'Location',
}

df_new = df_new.rename(columns=column_mapping)

common_cols = [c for c in df_old.columns if c in df_new.columns]
df_new = df_new[common_cols]

print(f"字段对齐完成，保留 {len(common_cols)} 列")

print("\n" + "=" * 60)
print("Step 4: 合并数据")
print("=" * 60)

# 备份旧数据
backup_path = os.path.join(BACKUP_DIR, 
    f"allcuisine_backup_{datetime.now().strftime('%Y%m%d')}.csv")
shutil.copy(MAIN_DATA_PATH, backup_path)
print(f"已备份到: {backup_path}")

# 合并
df_merged = pd.concat([df_old, df_new], ignore_index=True)
print(f"合并前: {len(df_old)} 条")
print(f"新数据: {len(df_new)} 条")
print(f"合并后: {len(df_merged)} 条")

# 去重
df_merged = df_merged.drop_duplicates(subset=['CAMIS', 'INSPECTION DATE'], keep='last')
print(f"去重后: {len(df_merged)} 条")

# 清洗坐标
df_merged['Latitude'] = pd.to_numeric(df_merged['Latitude'], errors='coerce')
df_merged['Longitude'] = pd.to_numeric(df_merged['Longitude'], errors='coerce')
df_merged = df_merged.dropna(subset=['Latitude', 'Longitude'])
df_merged = df_merged[(df_merged['Latitude'] != 0) & (df_merged['Longitude'] != 0)]

# 保存
print("\n" + "=" * 60)
print("Step 5: 保存更新")
print("=" * 60)

df_merged.to_csv(MAIN_DATA_PATH, index=False)
print(f"已更新: {MAIN_DATA_PATH}")

# 记录更新日志
log_entry = pd.DataFrame([{
    'update_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'new_records': len(df_new),
    'total_records': len(df_merged),
    'latest_inspection_date': df_merged['INSPECTION DATE'].max()
}])

if os.path.exists(UPDATE_LOG):
    log_entry.to_csv(UPDATE_LOG, mode='a', header=False, index=False)
else:
    log_entry.to_csv(UPDATE_LOG, index=False)

print(f"更新日志已记录: {UPDATE_LOG}")

print("\n" + "=" * 60)
print("数据更新完成！")
print("=" * 60)
