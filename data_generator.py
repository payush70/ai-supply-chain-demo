import pandas as pd
import numpy as np

def generate_primary_data(n=5000):
    np.random.seed(42)
    lead_time = np.random.normal(14, 5, n)
    order_volume = np.random.randint(100, 10000, n)
    weather_index = np.random.uniform(0, 1, n)
    supplier_reliability = np.random.uniform(0, 1, n)
    disruption_prob = (1 - supplier_reliability) * 0.6 + (weather_index * 0.4)
    disruption_prob += np.random.normal(0, 0.1, n)
    disruption_status = (disruption_prob > 0.5).astype(int)
    
    df = pd.DataFrame({
        'lead_time': lead_time.round(1),
        'order_volume': order_volume,
        'weather_index': weather_index.round(2),
        'supplier_reliability': supplier_reliability.round(2),
        'disruption_status': disruption_status
    })
    
    df.to_csv('primary_supply_chain_data.csv', index=False)
    print("Primary dataset generated: 'primary_supply_chain_data.csv'")

if __name__ == "__main__":
    generate_primary_data()