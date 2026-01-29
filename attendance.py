import pandas as pd
from datetime import datetime

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:
        df = pd.read_csv("attendance.csv")
    except:
        df = pd.DataFrame(columns=["Name","Date","In","Out"])

    if ((df["Name"]==name)&(df["Date"]==date)).any():
        df.loc[(df["Name"]==name)&(df["Date"]==date),"Out"] = time
    else:
        df.loc[len(df)] = [name,date,time,"-"]

    df.to_csv("attendance.csv", index=False)

