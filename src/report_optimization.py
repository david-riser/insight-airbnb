#!/usr/bin/env python 

import pandas as pd 

if __name__ == '__main__':

    data = pd.read_csv('./data/metrics/optimize.csv')
    data.sort_values('test_score', inplace = True, ascending = True)
    print(data.head(24))
