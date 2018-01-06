import numpy as np
from scipy import  stats


case = [1,2,3,4]

y = stats.moment(case,2)

mean_case = np.mean(case)

moment =0
for i in range(0,4):

    moment += (case[1] - mean_case)**2



print(mean_case)
print(y)
print(moment)