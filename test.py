import pandas as pd

#df=pd.read_csv("encoded_labels.csv")
#print(df.head())

import sqlite3

conn = sqlite3.connect('issues.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM issues")
print(cursor.fetchall())
conn.close()
