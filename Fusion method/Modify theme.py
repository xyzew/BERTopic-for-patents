import pandas as pd

count=0
for item in range(0,20):
    file_name = "D:/python/Bertopic/K VS H/"+"AResultH"+str(item)+".xlsx"

    # Read data
    df = pd.read_excel(file_name)
    df['Topic']=item

    # Save the processed data to a new file or overwrite the original file
    df.to_excel("D:/python/Bertopic/K VS H/" + "FAResultH" + str(item) + ".xlsx", index=False)  # 保存为新文件
    print(count)
    count = count + 1