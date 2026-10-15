import os
import requests
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")

train_number = "12919"

url = f"https://api.railradar.in/v1/train/{train_number}/live"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(
    url,
    headers=headers,
    timeout=30
)

print("STATUS:", response.status_code)

response.raise_for_status()

data = response.json()["data"]

print("\n========== AVAILABLE DATA ==========")
print(data.keys())

print("\n========== TRAIN ==========")

train = data.get("train", {})

for key, value in train.items():
    print(f"{key}: {value}")

print("\n========== CURRENT LOCATION ==========")

current = data.get("currentLocation")

if current:
    for key, value in current.items():
        print(f"{key}: {value}")
else:
    print("No current location data")

print("\n========== NEXT HALT ==========")

next_halt = data.get("nextHalt")

if next_halt:
    for key, value in next_halt.items():
        print(f"{key}: {value}")
else:
    print("No next halt data")

print("\n========== LIVE STATUS ==========")

for key, value in data.items():

    if key not in ["train", "route", "currentLocation", "nextHalt"]:
        print(f"{key}: {value}")

print("\n========== ROUTE ==========")

route = data.get("route", [])

print("Route entries:", len(route))

for station in route[:5]:

    print(
        station.get("sequence"),
        "|",
        station.get("stationCode"),
        "|",
        station.get("stationName"),
        "|",
        station.get("distance"),
        "km",
        "|",
        station.get("status")
    )