import pandas as pd

#Create an empty collection to store the patent numbers encountered
seen_patent_numbers = set()

# List=["D:/python/Bertopic/K VS H/ResultH -1.xlsx"]
# #
count=0
# Read data
data = pd.read_excel("D:/python/Bertopic/K VS H/ResultH -1.xlsx")
# Mark the patent numbers in the file that already exist in the collection
data['is_duplicated'] = data['id'].isin(seen_patent_numbers)
data_cleaned = data[~data['is_duplicated']].drop('is_duplicated', axis=1)
# Update the collection and add newly encountered patent numbers
seen_patent_numbers.update(data[~data['is_duplicated']]['id'])

for item in range(0,20):
    file_name = "D:/python/Bertopic/K VS H/"+"AH"+str(item)+"P.xlsx"
    # Read data
    df = pd.read_excel(file_name)
    # Mark the patent numbers in the file that already exist in the collection
    df['is_duplicated'] = df['id'].isin(seen_patent_numbers)

    # # Update the collection and add newly encountered patent numbers
    # seen_patent_numbers.update(df[~df['is_duplicated']]['id'])
    # Delete patent numbers marked as duplicates
    df = df[df['is_duplicated']].drop('is_duplicated', axis=1)



    file_nameH = "D:/python/Bertopic/K VS H/" + "ResultH" + str(item) + ".xlsx"
    RH = pd.read_excel(file_nameH)
    RH = pd.concat([RH, df], ignore_index=True)


    # Save the processed data to a new file or overwrite the original file
    RH.to_excel("D:/python/Bertopic/K VS H/" + "AResultH" + str(item) + ".xlsx", index=False)  # 保存为新文件
    print(count)
    count = count + 1
