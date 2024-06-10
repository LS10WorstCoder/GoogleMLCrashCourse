import numpy as np

featureArray = np.arange(6, 21)
labelArray = 3*(featureArray)+4
noiseArray = labelArray + (np.random.random([15]) * 4) - 2
print(featureArray)
print(labelArray)
print(noiseArray)