import os
import math
import pandas as pd
import numpy as np

def find_dataset():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    if not os.path.exists(raw_dir):
        raw_dir = os.path.join("data", "raw")
    
    files = [
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f)) and not f.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No files found in raw dataset directory: {raw_dir}")
    
    csv_files = [f for f in files if f.endswith(".csv")]
    target_file = csv_files[0] if csv_files else files[0]
    
    return os.path.join(raw_dir, target_file)

def inspect():
    filepath = find_dataset()
    filename = os.path.basename(filepath)
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print("=" * 100)
    print("CRICKET PLAYER ML PREDICTION SYSTEM - DATASET INSPECTION REPORT")
    print("=" * 100)
    print(f"1. Dataset Filename: {filename}")
    print(f"2. File Size:        {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    
    chunksize = 100_000
    total_rows = 0
    sample_df = None
    column_names = []
    dtypes_map = {}
    missing_counts = None
    
    num_stats = {}
    unique_sets = {}
    MAX_UNIQUE_TRACK = 50_000

    print("Reading dataset efficiently in chunks...")
    for chunk_idx, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, low_memory=False)):
        if chunk_idx == 0:
            column_names = list(chunk.columns)
            sample_df = chunk.head(5)
            missing_counts = pd.Series(0, index=column_names, dtype=int)
            
            for col in column_names:
                dtypes_map[col] = str(chunk[col].dtype)
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    num_stats[col] = {
                        'count': 0,
                        'sum': 0.0,
                        'sum_sq': 0.0,
                        'min': float('inf'),
                        'max': float('-inf')
                    }
                else:
                    unique_sets[col] = set()

        total_rows += len(chunk)
        missing_counts += chunk.isna().sum()

        for col in column_names:
            series = chunk[col].dropna()
            if col in num_stats:
                if not series.empty:
                    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                    if not numeric_series.empty:
                        cnt = len(numeric_series)
                        s = float(numeric_series.sum())
                        s_sq = float((numeric_series ** 2).sum())
                        mn = float(numeric_series.min())
                        mx = float(numeric_series.max())
                        
                        st = num_stats[col]
                        st['count'] += cnt
                        st['sum'] += s
                        st['sum_sq'] += s_sq
                        if mn < st['min']: st['min'] = mn
                        if mx > st['max']: st['max'] = mx
            elif col in unique_sets:
                if len(unique_sets[col]) < MAX_UNIQUE_TRACK:
                    vals = series.unique()
                    unique_sets[col].update(vals[:MAX_UNIQUE_TRACK - len(unique_sets[col])])

    num_cols = len(column_names)
    print(f"3. Total Row Count:  {total_rows:,}")
    print(f"4. Number of Cols:   {num_cols}")
    print("\n5 & 6 & 7. Column Names, Data Types, and Missing Values:")
    print("-" * 90)
    print(f"{'#':<4} {'Column Name':<35} {'Data Type':<15} {'Missing Count':<15} {'Missing %':<10}")
    print("-" * 90)
    for idx, col in enumerate(column_names, 1):
        m_cnt = missing_counts[col]
        m_pct = (m_cnt / total_rows * 100) if total_rows > 0 else 0.0
        dtype_str = dtypes_map.get(col, 'unknown')
        print(f"{idx:<4} {col:<35} {dtype_str:<15} {m_cnt:<15,} {m_pct:.2f}%")

    print("\n8. Small Sample of Rows (First 5 rows):")
    print("-" * 90)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(sample_df)

    print("\n9. Unique Value Counts for Categorical / Object Columns:")
    print("-" * 90)
    print(f"{'Column Name':<35} {'Unique Count (tracked)':<30}")
    print("-" * 90)
    for col, u_set in unique_sets.items():
        count_str = f"{len(u_set):,}" if len(u_set) < MAX_UNIQUE_TRACK else f">{MAX_UNIQUE_TRACK:,}"
        print(f"{col:<35} {count_str:<30}")

    print("\n10. Basic Numerical Statistics:")
    print("-" * 90)
    print(f"{'Column Name':<30} {'Count':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 90)
    for col, st in num_stats.items():
        cnt = st['count']
        if cnt > 0:
            mean = st['sum'] / cnt
            var = (st['sum_sq'] - (st['sum'] ** 2) / cnt) / (cnt - 1) if cnt > 1 else 0.0
            std = math.sqrt(max(0.0, var))
            mn = st['min']
            mx = st['max']
            print(f"{col:<30} {cnt:<12,} {mean:<15.4f} {std:<15.4f} {mn:<15.4f} {mx:<15.4f}")
        else:
            print(f"{col:<30} {'0':<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    print("=" * 100)

if __name__ == "__main__":
    inspect()
