import pandas as pd
import re

data = pd.read_excel("ResultVS H.xlsx")  # content type
abstracts = data['Topic']

df_target = pd.DataFrame(columns=['id','title','assignee','priority date','content','content_cutted','Topic'])
delete=[]


count=0
for i in range(-1,0):
    for index,item in enumerate(abstracts):
        # Remove empty digests
        if(item==i):
            df_target.loc[count] = data.loc[index]
            count=count+1
    count=0
    path="D:/python/Bertopic/K VS H/ResultH "+str(i)+".xlsx"
    df_target.to_excel(path, index=False)
    df_target = pd.DataFrame(columns=df_target.columns)

    print(i)
