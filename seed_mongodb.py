#!/usr/bin/env python3
"""
Seed MongoDB with realistic IoT sensor data for Text2Query benchmark.
Creates: plalion_klaen_sensor, plalion_company_sensor, lighting_weatherapi
"""
import random
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from config_utils import get_db_name, required_env

load_dotenv(override=True)

DB_NAME = get_db_name()

def seed_klaen_sensor(db, n=5000):
    """Indoor Klaen sensor: temperature, humidity, co2, voc, dust, ozone."""
    print(f"  Seeding plalion_klaen_sensor ({n} docs)...")
    col = db["plalion_klaen_sensor"]
    if col.count_documents({}) > 0:
        print(f"    Already has {col.count_documents({})} docs, skipping")
        return
    
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    docs = []
    for i in range(n):
        ts = base_time + timedelta(hours=i*0.3)
        hour = ts.hour
        # Simulate daily patterns
        temp_base = 22 + 3 * (1 if 10 <= hour <= 18 else -1) + random.gauss(0, 0.5)
        hum_base = 45 + 10 * (1 if 6 <= hour <= 14 else -1) + random.gauss(0, 3)
        co2_base = 400 + 200 * (1 if 8 <= hour <= 18 else 0) + random.gauss(0, 30)
        voc_base = 0.3 + 0.15 * random.gauss(0, 1)
        dust_base = 30 + 10 * random.gauss(0, 1)
        ozone_base = 0.04 + 0.02 * random.gauss(0, 1)
        
        docs.append({
            "timestamp": ts,
            "temperature": round(temp_base, 1),
            "humidity": round(hum_base, 1),
            "co2": round(max(300, co2_base), 0),
            "voc": round(max(0, voc_base), 2),
            "dust": round(max(0, dust_base), 1),
            "ozone": round(max(0, ozone_base), 3),
        })
    
    col.insert_many(docs)
    print(f"    ✓ Inserted {len(docs)} docs")

def seed_company_sensor(db, n=5000):
    """Company office sensor: temperature, humidity, co2, dust, voc, ozone."""
    print(f"  Seeding plalion_company_sensor ({n} docs)...")
    col = db["plalion_company_sensor"]
    if col.count_documents({}) > 0:
        print(f"    Already has {col.count_documents({})} docs, skipping")
        return
    
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    docs = []
    for i in range(n):
        ts = base_time + timedelta(hours=i*0.3)
        hour = ts.hour
        weekday = ts.weekday()
        is_workday = weekday < 5
        is_workhours = 8 <= hour <= 18
        
        if is_workday and is_workhours:
            temp_base = 23 + random.gauss(0, 0.8)
            hum_base = 50 + random.gauss(0, 5)
            co2_base = 600 + random.gauss(0, 100)
        else:
            temp_base = 20 + random.gauss(0, 0.5)
            hum_base = 40 + random.gauss(0, 3)
            co2_base = 350 + random.gauss(0, 30)
        
        docs.append({
            "timestamp": ts,
            "temperature": round(temp_base, 1),
            "humidity": round(hum_base, 1),
            "co2": round(max(300, co2_base), 0),
            "dust": round(max(0, 25 + random.gauss(0, 8)), 1),
            "voc": round(max(0, 0.25 + random.gauss(0, 0.1)), 2),
            "ozone": round(max(0, 0.03 + random.gauss(0, 0.01)), 3),
        })
    
    col.insert_many(docs)
    print(f"    ✓ Inserted {len(docs)} docs")

def seed_weatherapi(db, n=5000):
    """Weather API data: temp_c, humidity, condition, wind_kph, precip_mm, etc."""
    print(f"  Seeding lighting_weatherapi ({n} docs)...")
    col = db["lighting_weatherapi"]
    if col.count_documents({}) > 0:
        print(f"    Already has {col.count_documents({})} docs, skipping")
        return
    
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    conditions = ["Sunny", "Partly cloudy", "Cloudy", "Overcast", "Mist", 
                  "Fog", "Light rain", "Moderate rain", "Heavy rain", "Thunderstorm"]
    docs = []
    
    for i in range(n):
        ts = base_time + timedelta(hours=i)
        month = ts.month
        hour = ts.hour
        
        # Seasonal temperature pattern (Northern hemisphere)
        season_factor = 15 + 15 * (1 if 4 <= month <= 9 else -1) * 0.5
        daily_factor = 3 * (1 if 10 <= hour <= 16 else -1)
        temp = season_factor + daily_factor + random.gauss(0, 2)
        
        # Humidity inversely correlated with temperature
        hum = 70 - (temp - 15) * 0.8 + random.gauss(0, 5)
        hum = max(20, min(95, hum))
        
        # Weather condition
        if hum > 75:
            cond = random.choice(["Rain", "Mist", "Cloudy", "Overcast", "Light rain"])
        elif hum > 55:
            cond = random.choice(["Partly cloudy", "Cloudy", "Overcast"])
        else:
            cond = random.choice(["Sunny", "Partly cloudy", "Clear"])
        
        wind = max(0, 8 + random.gauss(0, 4))
        precip = max(0, random.gauss(0, 2)) if "rain" in cond.lower() or "thunder" in cond.lower() else 0
        pressure = 1013 + random.gauss(0, 8)
        visibility = max(2, 15 - hum * 0.1 + random.gauss(0, 2))
        uv_index = max(0, (8 if 11 <= hour <= 14 else 3) * (1 if "sunny" in cond.lower() or "clear" in cond.lower() else 0.3) + random.gauss(0, 1))
        aqi = max(0, int(50 + random.gauss(0, 25)))
        pm25 = max(0, 15 + random.gauss(0, 8))
        pm10 = max(0, pm25 * 1.5 + random.gauss(0, 5))
        feelslike = temp - wind * 0.2 + random.gauss(0, 0.5)
        
        docs.append({
            "timestamp": ts,
            "temp_c": round(temp, 1),
            "feelslike_c": round(feelslike, 1),
            "humidity": round(hum, 0),
            "condition": cond,
            "wind_kph": round(wind, 1),
            "wind_degree": random.randint(0, 360),
            "pressure_mb": round(pressure, 0),
            "precip_mm": round(precip, 1),
            "humidity_inside": round(45 + random.gauss(0, 5), 0),
            "visibility_km": round(visibility, 1),
            "uv_index": round(uv_index, 1),
            "aqi": aqi,
            "pm2_5": round(pm25, 1),
            "pm10": round(pm10, 1),
            "co": round(max(0, 200 + random.gauss(0, 50)), 0),
            "no2": round(max(0, 20 + random.gauss(0, 8)), 1),
            "so2": round(max(0, 5 + random.gauss(0, 3)), 1),
            "o3": round(max(0, 40 + random.gauss(0, 15)), 1),
        })
    
    col.insert_many(docs)
    print(f"    ✓ Inserted {len(docs)} docs")

def main():
    print("="*60)
    print("  SEEDING TEXT2QUERY MONGODB DATA")
    print("="*60)
    
    client = MongoClient(required_env("MONGODB_URI"), serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    
    # Create indexes
    for col_name in ["plalion_klaen_sensor", "plalion_company_sensor", "lighting_weatherapi"]:
        db[col_name].create_index("timestamp")
        print(f"  Created timestamp index on {col_name}")
    
    seed_klaen_sensor(db, 5000)
    seed_company_sensor(db, 5000)
    seed_weatherapi(db, 5000)
    
    # Verify
    print("\n  Verification:")
    for col_name in ["plalion_klaen_sensor", "plalion_company_sensor", "lighting_weatherapi"]:
        count = db[col_name].count_documents({})
        sample = db[col_name].find_one()
        print(f"    {col_name}: {count} docs")
        if sample:
            print(f"      Fields: {list(sample.keys())}")
            print(f"      Sample: {sample['timestamp']}")
    
    client.close()
    print("\n  ✅ Seeding complete!")

if __name__ == "__main__":
    main()
