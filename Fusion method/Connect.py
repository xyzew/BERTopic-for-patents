import pandas as pd
#Create an empty DataFrame to store all data
all_data = pd.DataFrame()

# df2=pd.read_excel("DocumentName.xlsx")
# List=[]
# for item11 in df2['keywords']:
#     List.append(item11)
# List=["D:/python/Bertopic/K VS H/AesultH0 X.xlsx","D:/python/Bertopic/K VS H/AesultH1 X.xlsx","D:/python/Bertopic/K VS H/AesultH2 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH3 X.xlsx","D:/python/Bertopic/K VS H/AesultH4 X.xlsx","D:/python/Bertopic/K VS H/AesultH5 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH6 X.xlsx","D:/python/Bertopic/K VS H/AesultH7 X.xlsx","D:/python/Bertopic/K VS H/AesultH8 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH9 X.xlsx","D:/python/Bertopic/K VS H/AesultH10 X.xlsx","D:/python/Bertopic/K VS H/AesultH11 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH12 X.xlsx","D:/python/Bertopic/K VS H/AesultH13 X.xlsx","D:/python/Bertopic/K VS H/AesultH14 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH15 X.xlsx","D:/python/Bertopic/K VS H/AesultH16 X.xlsx","D:/python/Bertopic/K VS H/AesultH17 X.xlsx",
#       "D:/python/Bertopic/K VS H/AesultH18 X.xlsx","D:/python/Bertopic/K VS H/AesultH19 X.xlsx"]

# list=["D:/python/Bertopic/K VS H/ResultH19.xlsx"]
# # Iterate through all files in a directory
# for item in list:
#     data = pd.read_excel(item)
#     # Append data to all_data DataFrame
#     all_data = all_data._append(data, ignore_index=True)
# all_data.to_excel('D:/python/Bertopic/K VS H/AH19P.xlsx', index=False)

# Iterate through all files in the directory
for item in range(0,20):
    filename="D:/python/Bertopic/K VS H/" + "FAResultH" + str(item) + ".xlsx"
    data = pd.read_excel(filename)
    #Append data to all_data DataFrame
    all_data = all_data._append(data, ignore_index=True)
all_data.to_excel('D:/python/Bertopic/K VS H/ZFinal.xlsx', index=False)
