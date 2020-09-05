import pandas as pd
import numpy as np

val1 = "012345"





# for i in range(0, len(val)) :
#    print(val[i])
def reverse(val):
    if len(val) == 1:
        return val[0]
    else:
        return reverse(val[1:len(val)]) + val[0]


print(reverse(val1))
