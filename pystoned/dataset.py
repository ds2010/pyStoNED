import pandas as pd
import numpy as np

def firm(inputlist=['OPEX', 'CAPEX'], output='Energy'):
    url = 'https://raw.githubusercontent.com/ds2010/pyStoNED-Tutorials/master/Data/firms.csv'
    data = pd.read_csv(url, error_bad_lines=False)

    # output 
    y = data[output]

    # input 
    x = []
    for item in inputlist:
        x.append(np.asmatrix(data[item]).T)
    x = np.concatenate((x), axis=1)

    return x, y
