import numpy as np
import pandas as pd



x = np.random.uniform(0, 3.14, 5000)
y = np.sin(x)
print(x)
print(y)



# write to file
dataframe = pd.DataFrame({'x':x, 'y':y})
dataframe.to_csv("trainingset5000.csv", index=False, sep=',')
