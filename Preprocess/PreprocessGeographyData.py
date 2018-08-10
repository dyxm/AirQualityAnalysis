# Created by Yuexiong Ding
# Date: 2018/8/7
# Description: preprocessing geography data

import pandas as pd
import numpy as np

raw_data = pd.read_csv('../DataSet/RawData/Geography/county_geography_2016.txt', low_memory=False, dtype=str,
                       delimiter="\t")
new_data_df = pd.DataFrame({'State County Code': raw_data['GEOID'], 'Year': ['2016'] * len(raw_data),
                            'ALAND': raw_data['ALAND'], 'AWATER': raw_data['AWATER'], 'INTPTLAT': raw_data['INTPTLAT'],
                            'INTPTLONG': raw_data['INTPTLONG']})
new_data_df.to_csv('../DataSet/ProcessedData/Geography/county_geography_2016.csv', index=False)
print(new_data_df)


