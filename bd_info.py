import pandas as pd
import json

# Предположим, что ваш DataFrame называется df
# Если у вас есть данные, замените следующую строку на загрузку ваших данных
df = pd.read_csv('df_for_work.csv')
selected = ['proc_id','executor','ktru','customer','price']
print(df[selected])
df_filter = df[selected]
df_filter.to_json("output.json", orient="records", indent=2, force_ascii=False)

