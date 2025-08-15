import pandas as pd

import nfl_td_lambda.data_collection as data_collection
import nfl_data_py as nfl

# x = data_collection.get_depth_chart_data([2023,2024])
# print(x.head(10))
# #print depth chart player id '00-0038597'
# print(x[x['player_id'] == '00-0038597'])

x=data_collection.get_depth_chart_data([2020,2021,2022,2023,2024])
print(x[x['player_id'] == '00-0039040'].drop_duplicates().sort_values(by=['season', 'week']))
