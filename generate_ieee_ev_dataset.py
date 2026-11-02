import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_battery_degradation(num_samples=10000, seed=42):
    np.random.seed(seed)
    cycles = np.linspace(1, 2000, num_samples)
    time_hours = cycles * 1.5 + np.random.uniform(-0.5, 0.5, size=num_samples)
    
    # Capacity degradation curve with noise (100 Ah -> ~75 Ah)
    capacity_ah = 100.0 * np.exp(-0.00012 * cycles) + np.random.normal(0, 0.2, size=num_samples)
    capacity_ah = np.clip(capacity_ah, 60.0, 102.0)
    
    soh = capacity_ah / 100.0
    
    # Internal resistance growth (0.015 ohm -> ~0.040 ohm)
    internal_resistance = 0.015 + 0.00001 * cycles + np.random.normal(0, 0.0005, size=num_samples)
    internal_resistance = np.clip(internal_resistance, 0.01, 0.06)
    
    voltage = 3.7 + 0.4 * np.sin(cycles * 0.1) + np.random.normal(0, 0.02, size=num_samples)
    voltage = np.clip(voltage, 3.0, 4.2)
    
    current = np.random.uniform(10.0, 80.0, size=num_samples)
    temp_c = 25.0 + 0.15 * current + np.random.normal(0, 2.0, size=num_samples)
    temp_c = np.clip(temp_c, 15.0, 55.0)
    
    soc = np.random.uniform(0.1, 0.95, size=num_samples)
    
    df = pd.DataFrame({
        'cycle_id': cycles.astype(int),
        'time_hours': np.round(time_hours, 2),
        'voltage_v': np.round(voltage, 3),
        'current_a': np.round(current, 2),
        'temperature_c': np.round(temp_c, 2),
        'state_of_charge': np.round(soc, 4),
        'capacity_ah': np.round(capacity_ah, 3),
        'internal_resistance_ohm': np.round(internal_resistance, 5),
        'state_of_health_soh': np.round(soh, 4)
    })
    
    out_path = os.path.join(DATA_DIR, "ev_battery_degradation.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved EV battery degradation dataset to {out_path} (Shape: {df.shape})")
    return out_path

def generate_charging_behavior(num_samples=10000, seed=43):
    np.random.seed(seed)
    
    session_ids = [f"SESS_{100000 + i}" for i in range(num_samples)]
    user_ids = [f"USR_{np.random.randint(1000, 9999)}" for _ in range(num_samples)]
    station_ids = [f"STN_{np.random.randint(100, 500)}" for _ in range(num_samples)]
    
    charger_types = np.random.choice(["Level 1 (AC 1.4kW)", "Level 2 (AC 7.2kW)", "DC Fast (50kW)", "DC Ultra-Fast (150kW)"], 
                                     size=num_samples, p=[0.15, 0.55, 0.20, 0.10])
    
    arrival_soc = np.random.uniform(0.05, 0.45, size=num_samples)
    departure_soc = np.random.uniform(0.75, 1.00, size=num_samples)
    
    # Battery pack size assumed ~75 kWh
    energy_delivered = (departure_soc - arrival_soc) * 75.0 + np.random.normal(0, 2.0, size=num_samples)
    energy_delivered = np.clip(energy_delivered, 3.0, 85.0)
    
    duration_hours = np.zeros(num_samples)
    for i, c_type in enumerate(charger_types):
        if "Level 1" in c_type:
            duration_hours[i] = energy_delivered[i] / 1.4 + np.random.normal(0, 0.5)
        elif "Level 2" in c_type:
            duration_hours[i] = energy_delivered[i] / 7.2 + np.random.normal(0, 0.2)
        elif "50kW" in c_type:
            duration_hours[i] = energy_delivered[i] / 40.0 + np.random.normal(0, 0.1)
        else:
            duration_hours[i] = energy_delivered[i] / 110.0 + np.random.normal(0, 0.05)
    
    duration_hours = np.clip(duration_hours, 0.2, 12.0)
    
    tariff_usd_kwh = np.random.uniform(0.12, 0.42, size=num_samples)
    total_cost_usd = energy_delivered * tariff_usd_kwh
    
    start_times = pd.date_range(start='2024-01-01', periods=num_samples, freq='35min')
    
    df = pd.DataFrame({
        'session_id': session_ids,
        'timestamp': start_times,
        'user_id': user_ids,
        'station_id': station_ids,
        'charger_type': charger_types,
        'arrival_soc': np.round(arrival_soc, 4),
        'departure_soc': np.round(departure_soc, 4),
        'charging_duration_hours': np.round(duration_hours, 2),
        'energy_delivered_kwh': np.round(energy_delivered, 2),
        'electricity_tariff_usd_per_kwh': np.round(tariff_usd_kwh, 3),
        'total_cost_usd': np.round(total_cost_usd, 2)
    })
    
    out_path = os.path.join(DATA_DIR, "ev_charging_behavior.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved EV charging behavior dataset to {out_path} (Shape: {df.shape})")
    return out_path

def generate_energy_consumption(num_samples=10000, seed=44):
    np.random.seed(seed)
    
    trip_ids = [f"TRIP_{200000 + i}" for i in range(num_samples)]
    vehicle_ids = [f"EV_{np.random.randint(100, 999)}" for _ in range(num_samples)]
    
    timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='25min')
    
    distance_km = np.random.exponential(scale=25.0, size=num_samples) + 2.0
    distance_km = np.clip(distance_km, 1.0, 250.0)
    
    avg_speed_kmh = np.random.normal(loc=55.0, scale=20.0, size=num_samples)
    avg_speed_kmh = np.clip(avg_speed_kmh, 15.0, 130.0)
    
    ambient_temp_c = np.random.uniform(-10.0, 40.0, size=num_samples)
    
    # HVAC consumption increases at extreme temperatures
    hvac_power_kw = 0.5 + 0.1 * np.maximum(0, 20.0 - ambient_temp_c) + 0.12 * np.maximum(0, ambient_temp_c - 24.0)
    hvac_power_kw = np.clip(hvac_power_kw, 0.2, 4.5) + np.random.normal(0, 0.1, size=num_samples)
    
    elevation_change_m = np.random.normal(loc=0.0, scale=150.0, size=num_samples)
    
    # Base energy consumption per km (kWh/km)
    base_rate = 0.15 + 0.0008 * (avg_speed_kmh - 50.0)**2 / 50.0 + (hvac_power_kw * 0.02)
    base_rate += (elevation_change_m / 1000.0) * 0.05
    consumption_rate = np.clip(base_rate + np.random.normal(0, 0.015, size=num_samples), 0.10, 0.45)
    
    total_energy_kwh = distance_km * consumption_rate
    regen_energy_kwh = np.maximum(0, -elevation_change_m * 0.0003 * distance_km) + np.random.uniform(0.1, 1.5, size=num_samples)
    regen_energy_kwh = np.clip(regen_energy_kwh, 0.0, total_energy_kwh * 0.3)
    
    net_energy_kwh = total_energy_kwh - regen_energy_kwh
    
    df = pd.DataFrame({
        'trip_id': trip_ids,
        'timestamp': timestamps,
        'vehicle_id': vehicle_ids,
        'distance_km': np.round(distance_km, 2),
        'avg_speed_kmh': np.round(avg_speed_kmh, 2),
        'ambient_temp_c': np.round(ambient_temp_c, 1),
        'hvac_power_kw': np.round(hvac_power_kw, 2),
        'elevation_change_m': np.round(elevation_change_m, 1),
        'consumption_kwh_per_km': np.round(consumption_rate, 4),
        'total_energy_kwh': np.round(total_energy_kwh, 2),
        'regen_energy_kwh': np.round(regen_energy_kwh, 2),
        'net_energy_kwh': np.round(net_energy_kwh, 2)
    })
    
    out_path = os.path.join(DATA_DIR, "ev_energy_consumption.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved EV energy consumption dataset to {out_path} (Shape: {df.shape})")
    return out_path

if __name__ == "__main__":
    print("Generating IEEE EV Multi-Domain dataset files...")
    generate_battery_degradation()
    generate_charging_behavior()
    generate_energy_consumption()
    print("All EV datasets generated successfully!")
