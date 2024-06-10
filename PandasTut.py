import numpy as np
import pandas as pd

twoD = (np.random.randint(low = 0, high=101,size=(3,4)))
colNames = ['Eleanor','Chidi','Tahani','Jason']
myDataFrame = pd.DataFrame(data=twoD,columns=colNames)
print(myDataFrame)
print(myDataFrame['Eleanor'][1])
myDataFrame["Janet"] = myDataFrame["Tahani"] + myDataFrame["Jason"]
print(myDataFrame)