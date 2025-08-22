import pandas as pd

import nfl_td_lambda.data_collection as data_collection
import nfl_data_py as nfl

# x = data_collection.get_depth_chart_data([2023,2024])
# print(x.head(10))
# #print depth chart player id '00-0038597'
# print(x[x['player_id'] == '00-0038597'])

unscaled = pd.read_csv('unscaled.csv')
predictions = pd.read_csv('predictions.csv')

player_id = '00-0035700'

unscaled_player = unscaled[unscaled['player_id'] == player_id]
predictions_player = predictions[predictions['player_id'] == player_id]

print("Unscaled Data:")
print(unscaled_player)

print("\nPredictions Data:")
print(predictions_player)

print("\nAverage Carries (Unscaled Data):")
print(unscaled_player['avg_carries'])

print("\nAverage Carries (Predictions Data):")
print(predictions_player['avg_carries'])