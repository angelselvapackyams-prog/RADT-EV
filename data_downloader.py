import os
import glob
import numpy as np
import pandas as pd
import torch
import kagglehub

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def download_kaggle_datasets():
    print("=== Step 1: Downloading Kaggle Datasets ===")

    # 1. Electric Vehicle Charging Dataset
    print("Downloading Electric Vehicle Charging Dataset (mexwell/electric-vehicle-charging-dataset)...")
    try:
        ev_path = kagglehub.dataset_download('mexwell/electric-vehicle-charging-dataset')
        print(f"Downloaded to: {ev_path}")
    except Exception as e:
        print(f"Warning downloading EV charging dataset: {e}")
        ev_path = None

    # 2. EV Battery Charging Dataset
    print("Downloading EV Battery Charging Dataset (programmer3/ev-battery-charging-dataset)...")
    try:
        batt_path = kagglehub.dataset_download('programmer3/ev-battery-charging-dataset')
        print(f"Downloaded to: {batt_path}")
    except Exception as e:
        print(f"Warning downloading battery dataset: {e}")
        batt_path = None

    # 3. Smart Grid Stability Dataset
    print("Downloading Smart Grid Stability Dataset (pcbreviglieri/smart-grid-stability)...")
    try:
        grid_path = kagglehub.dataset_download('pcbreviglieri/smart-grid-stability')
        print(f"Downloaded to: {grid_path}")
    except Exception as e:
        print(f"Warning downloading grid dataset: {e}")
        grid_path = None

    return ev_path, batt_path, grid_path

def generate_reliability_dataset(num_samples=15000, num_steps=24, seed=42):
    print("\n=== Step 2: Generating Reliability-Aware Charging Dataset (Table 7) ===")
    np.random.seed(seed)
    
    # Time vector for diurnal cycles
    t_hours = np.tile(np.linspace(0, 24, num_steps), num_samples // num_steps + 1)[:num_samples]

    # EV Arrival Rate (Poisson distribution, 5-100 EVs/hr)
    base_arrival = 20 + 30 * np.sin(2 * np.pi * (t_hours - 8) / 24)
    base_arrival = np.clip(base_arrival, 5, 100)
    ev_arrivals = np.random.poisson(base_arrival)

    # Initial SOC (Uniform random 10-90%)
    initial_soc = np.random.uniform(0.10, 0.90, size=num_samples)

    # Energy Demand (Normal distribution, 5-80 kWh)
    energy_demand = np.clip(np.random.normal(loc=35.0, scale=15.0, size=num_samples), 5.0, 80.0)

    # Solar Renewable Generation (Time-varying curve + Gaussian noise, 0-500 kW)
    solar_base = np.maximum(0, 450 * np.sin(np.pi * (t_hours - 6) / 12))
    solar_base[ (t_hours < 6) | (t_hours > 18) ] = 0
    solar_gen = np.clip(solar_base + np.random.normal(0, 25.0, size=num_samples), 0.0, 500.0)

    # BESS Storage SOC (Uniform random 20-100%)
    bess_soc = np.random.uniform(0.20, 1.00, size=num_samples)

    # Dynamic Electricity Price (Time-of-Use ToU pricing model, 4-20 Currency/kWh)
    tou_price = 6.0 + 8.0 * ( (t_hours >= 17) & (t_hours <= 21) ).astype(float) + \
                4.0 * ( (t_hours >= 8) & (t_hours <= 12) ).astype(float) + \
                np.random.uniform(-1.0, 2.0, size=num_samples)
    grid_price = np.clip(tou_price, 4.0, 20.0)

    # Grid Available Power (Constraint-based model, 100-1000 kW)
    grid_power = np.clip(800 - 300 * np.sin(2 * np.pi * (t_hours - 18) / 24) + np.random.normal(0, 50, size=num_samples), 100.0, 1000.0)

    # Infrastructure Health Index (Reliability degradation model, 0.5-1.0)
    infra_health = np.clip(0.95 - 0.005 * (np.arange(num_samples) / num_samples) - np.random.exponential(0.02, size=num_samples), 0.50, 1.00)

    # Battery Temperature (Thermal Evolution Model, 20-60 °C)
    battery_temp = np.clip(25.0 + 15.0 * (energy_demand / 80.0) + np.random.normal(0, 3.0, size=num_samples), 20.0, 60.0)

    # Compute Ground Truth Hidden Reliability State H(t) and DNA Profile rho_i(t)
    health_index = infra_health * (1.0 - 0.002 * (battery_temp - 25.0))
    health_index = np.clip(health_index, 0.4, 1.0)
    risk_score = np.clip(1.0 - health_index + 0.1 * (grid_price / 20.0), 0.0, 1.0)
    rul_hours = np.clip(health_index * 5000.0 + np.random.normal(0, 100, size=num_samples), 500, 5000)

    reliability_df = pd.DataFrame({
        'time_hour': t_hours,
        'ev_arrivals': ev_arrivals,
        'initial_soc': initial_soc,
        'energy_demand': energy_demand,
        'solar_gen': solar_gen,
        'bess_soc': bess_soc,
        'grid_price': grid_price,
        'grid_power': grid_power,
        'infra_health': infra_health,
        'battery_temp': battery_temp,
        'health_index': health_index,
        'risk_score': risk_score,
        'rul_hours': rul_hours
    })

    rel_csv_path = os.path.join(DATA_DIR, "reliability_dataset.csv")
    reliability_df.to_csv(rel_csv_path, index=False)
    print(f"Generated reliability dataset saved to: {rel_csv_path} (Shape: {reliability_df.shape})")
    
    # Also ensure IEEE EV multi-domain datasets are generated
    try:
        from generate_ieee_ev_dataset import generate_battery_degradation, generate_charging_behavior, generate_energy_consumption
        generate_battery_degradation(num_samples=10000)
        generate_charging_behavior(num_samples=10000)
        generate_energy_consumption(num_samples=10000)
    except Exception as e:
        print(f"Warning generating IEEE EV datasets: {e}")
        
    return reliability_df

def preprocess_and_save():
    print("=== Step 3: Preprocessing Data and Saving Splits ===")
    ensure_dirs()
    
    # Download Kaggle datasets
    ev_path, batt_path, grid_path = download_kaggle_datasets()
    
    # Generate reliability dataset
    rel_df = generate_reliability_dataset(num_samples=15000, num_steps=24, seed=42)

    # Feature columns for modeling: 10 input dimensions
    feature_cols = ['time_hour', 'ev_arrivals', 'initial_soc', 'energy_demand', 
                    'solar_gen', 'bess_soc', 'grid_price', 'grid_power', 
                    'infra_health', 'battery_temp']
    target_cols = ['health_index', 'risk_score', 'rul_hours']

    X = rel_df[feature_cols].values.astype(np.float32)
    Y = rel_df[target_cols].values.astype(np.float32)

    # Normalize features (Zero mean, unit variance)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-6
    X_norm = (X - X_mean) / X_std

    # Splits: 70% Train, 15% Val, 15% Test
    n_total = len(X_norm)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    X_train, Y_train = X_norm[:n_train], Y[:n_train]
    X_val, Y_val = X_norm[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X_norm[n_train+n_val:], Y[n_train+n_val:]

    # Save to PyTorch tensors
    torch.save({
        'X_train': torch.tensor(X_train),
        'Y_train': torch.tensor(Y_train),
        'X_val': torch.tensor(X_val),
        'Y_val': torch.tensor(Y_val),
        'X_test': torch.tensor(X_test),
        'Y_test': torch.tensor(Y_test),
        'X_mean': torch.tensor(X_mean),
        'X_std': torch.tensor(X_std),
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }, os.path.join(PREPROCESSED_DIR, "preprocessed_data.pt"))

    print(f"Saved preprocessed dataset tensor to: {os.path.join(PREPROCESSED_DIR, 'preprocessed_data.pt')}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

if __name__ == "__main__":
    preprocess_and_save()
