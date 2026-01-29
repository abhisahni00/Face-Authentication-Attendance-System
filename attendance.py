import pandas as pd
from datetime import datetime
import os

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
    else:
        df = pd.DataFrame(columns=["Name", "Date", "In", "Out"])

    mask = (df["Name"] == name) & (df["Date"] == date)

    if mask.any():
        df.loc[mask, "Out"] = time
    else:
        df.loc[len(df)] = [name, date, time, "-"]

    df.to_csv("attendance.csv", index=False)
