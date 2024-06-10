import numpy as np
import pandas as pd

twoD = ([np.random.randint(101,size=4),np.random.randint(101,size=4),np.random.randint(101,size=4)])
colNames = ['Eleanor','Chidi','Tahani','Jason']
myDataFrame = pd.DataFrame(data=twoD,columns=colNames)
print(myDataFrame)
print(myDataFrame.iloc[[1]])