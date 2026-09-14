import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ecommerce_dataset(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    categories = {
        "Technology": ["Phones", "Accessories", "Machines", "Copiers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Paper", "Binders", "Storage", "Art", "Appliances"]
    }
    regions = ["North", "South", "East", "West", "Central"]
    
    start_date = datetime(2025, 1, 1)
    
    records = []
    for i in range(1, n_rows + 1):
        cat = np.random.choice(list(categories.keys()))
        sub_cat = np.random.choice(categories[cat])
        region = np.random.choice(regions)
        order_date = start_date + timedelta(days=int(np.random.randint(0, 365)))
        customer_id = f"CUST-{np.random.randint(100, 300)}"
        sales = round(float(np.random.exponential(scale=150) + 15), 2)
        quantity = int(np.random.randint(1, 10))
        discount = round(float(np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])), 2)
        profit_margin = np.random.uniform(-0.15, 0.45)
        profit = round(sales * profit_margin, 2)
        
        records.append({
            "order_id": f"ORD-{2025000 + i}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "customer_id": customer_id,
            "region": region,
            "category": cat,
            "sub_category": sub_cat,
            "sales": sales,
            "quantity": quantity,
            "discount": discount,
            "profit": profit
        })
        
    df = pd.DataFrame(records)
    
    # Introduce controlled real-world quirks for profiling & cleaning demo
    # Missing values
    df.loc[df.sample(frac=0.04, random_state=1).index, "profit"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=2).index, "region"] = np.nan
    # Inconsistent category casing
    mask = df.sample(frac=0.05, random_state=3).index
    df.loc[mask, "category"] = df.loc[mask, "category"].str.lower()
    # Outliers
    outlier_idx = df.sample(n=3, random_state=4).index
    df.loc[outlier_idx, "sales"] = df.loc[outlier_idx, "sales"] * 15
    # Duplicates
    dup_rows = df.iloc[:4].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

def generate_customer_churn_dataset(n_rows: int = 400) -> pd.DataFrame:
    np.random.seed(101)
    genders = ["Male", "Female"]
    geographies = ["France", "Spain", "Germany"]
    
    records = []
    for i in range(1, n_rows + 1):
        age = int(np.random.normal(38, 9))
        age = max(18, min(80, age))
        tenure = int(np.random.randint(0, 11))
        balance = round(float(np.random.choice([0.0, np.random.uniform(20000, 150000)])), 2)
        salary = round(float(np.random.uniform(30000, 180000)), 2)
        num_products = int(np.random.choice([1, 2, 3, 4], p=[0.5, 0.4, 0.08, 0.02]))
        is_active = int(np.random.choice([0, 1], p=[0.48, 0.52]))
        exited = 1 if (age > 45 and balance > 50000 and is_active == 0) or np.random.rand() < 0.15 else 0
        
        records.append({
            "customer_id": f"USR-{10000 + i}",
            "geography": np.random.choice(geographies),
            "gender": np.random.choice(genders),
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "num_of_products": num_products,
            "has_credit_card": int(np.random.choice([0, 1], p=[0.3, 0.7])),
            "is_active_member": is_active,
            "estimated_salary": salary,
            "churn": exited
        })
    df = pd.DataFrame(records)
    # Add a few missing values
    df.loc[df.sample(frac=0.03, random_state=5).index, "estimated_salary"] = np.nan
    return df

def generate_sales_dataset(n_rows: int = 800) -> pd.DataFrame:
    np.random.seed(202)
    segments = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
    channels = ["Direct Sales", "Partner", "Online Web", "Field Team"]
    countries = ["United States", "United Kingdom", "Germany", "Japan", "Canada"]
    products = ["Quantum AI Pro", "DataPulse Suite", "CloudStream", "NeuralSync", "VectorEngine"]
    
    start_date = datetime(2025, 1, 1)
    records = []
    for i in range(1, n_rows + 1):
        order_date = start_date + timedelta(days=int(np.random.randint(0, 365)))
        product = np.random.choice(products)
        segment = np.random.choice(segments, p=[0.25, 0.35, 0.25, 0.15])
        channel = np.random.choice(channels)
        country = np.random.choice(countries)
        units = int(np.random.randint(1, 20))
        unit_price = {"Quantum AI Pro": 450.0, "DataPulse Suite": 280.0, "CloudStream": 120.0, "NeuralSync": 620.0, "VectorEngine": 850.0}[product]
        revenue = round(float(units * unit_price * np.random.uniform(0.9, 1.05)), 2)
        cost = round(float(revenue * np.random.uniform(0.4, 0.7)), 2)
        profit = round(revenue - cost, 2)
        
        records.append({
            "transaction_id": f"TXN-{100000 + i}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "product": product,
            "segment": segment,
            "channel": channel,
            "country": country,
            "units_sold": units,
            "revenue": revenue,
            "cost": cost,
            "profit": profit
        })
    df = pd.DataFrame(records)
    return df

def generate_finance_dataset(n_rows: int = 600) -> pd.DataFrame:
    np.random.seed(303)
    asset_classes = ["Equities", "Fixed Income", "Commodities", "Currencies", "Real Estate"]
    risk_ratings = ["Low Risk", "Medium Risk", "High Risk", "Critical Speculative"]
    portfolios = ["Alpha Growth", "Balanced Yield", "Conservative Bond", "Global Macro"]
    
    records = []
    for i in range(1, n_rows + 1):
        asset = np.random.choice(asset_classes)
        portfolio = np.random.choice(portfolios)
        risk = np.random.choice(risk_ratings, p=[0.35, 0.4, 0.2, 0.05])
        allocation = round(float(np.random.uniform(10000, 500000)), 2)
        roi_pct = round(float(np.random.normal(7.5, 4.2)), 2)
        volatility = round(float(np.random.uniform(2.0, 18.5)), 2)
        sharpe_ratio = round(float(roi_pct / max(0.5, volatility)), 2)
        
        records.append({
            "asset_id": f"AST-{5000 + i}",
            "asset_class": asset,
            "portfolio": portfolio,
            "risk_tier": risk,
            "capital_allocated": allocation,
            "expected_return_pct": roi_pct,
            "volatility_index": volatility,
            "sharpe_ratio": sharpe_ratio
        })
    return pd.DataFrame(records)

def generate_marketing_dataset(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(404)
    platforms = ["Google Ads", "LinkedIn B2B", "Meta Ads", "Organic Search", "Email Campaign"]
    campaign_types = ["Brand Awareness", "Lead Generation", "Product Launch", "Retargeting"]
    
    records = []
    for i in range(1, n_rows + 1):
        platform = np.random.choice(platforms)
        campaign = np.random.choice(campaign_types)
        spend = round(float(np.random.uniform(200, 8000)), 2)
        impressions = int(spend * np.random.uniform(15, 45))
        clicks = int(impressions * np.random.uniform(0.015, 0.045))
        conversions = max(1, int(clicks * np.random.uniform(0.03, 0.12)))
        cpc = round(spend / max(1, clicks), 2)
        cpa = round(spend / conversions, 2)
        cac = round(cpa * 1.25, 2)
        
        records.append({
            "campaign_id": f"CMP-{8000 + i}",
            "platform": platform,
            "campaign_objective": campaign,
            "ad_spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "cost_per_click": cpc,
            "acquisition_cost": cac
        })
    return pd.DataFrame(records)

def generate_healthcare_dataset(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(505)
    departments = ["Cardiology", "Neurology", "Oncology", "Orthopedics", "General Medicine"]
    statuses = ["Discharged", "Admitted", "Critical Care", "Outpatient Followup"]
    insurance_types = ["Private PPO", "HMO Premium", "Medicare", "Corporate Health"]
    
    records = []
    for i in range(1, n_rows + 1):
        age = int(np.random.normal(52, 16))
        age = max(18, min(95, age))
        dept = np.random.choice(departments)
        status = np.random.choice(statuses, p=[0.55, 0.25, 0.08, 0.12])
        insurance = np.random.choice(insurance_types)
        length_of_stay = int(np.random.exponential(4) + 1)
        treatment_cost = round(float(length_of_stay * np.random.uniform(850, 3200)), 2)
        patient_satisfaction = round(float(np.random.uniform(3.0, 5.0)), 1)
        
        records.append({
            "patient_id": f"PAT-{20000 + i}",
            "age": age,
            "department": dept,
            "patient_status": status,
            "insurance_provider": insurance,
            "length_of_stay_days": length_of_stay,
            "treatment_cost": treatment_cost,
            "satisfaction_score": patient_satisfaction
        })
    return pd.DataFrame(records)

