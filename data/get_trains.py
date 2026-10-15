import os
import requests
import csv
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")

url = "https://railradar.in/api/v1/lookup/trains/ntes"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

data = response.json()["data"]

with open("data/trains.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["train_number", "train_name"])

    for train_number, train_name in data.items():
        writer.writerow([train_number, train_name])

print(f"Saved {len(data)} trains to data/trains.csv")