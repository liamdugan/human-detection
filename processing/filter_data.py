import random
import collections
import numpy as np
import pandas as pd

full_df = pd.read_csv('unfiltered_data.csv')

# Filter out annotators who abused bugs
full_df = full_df[~full_df['annotator'].isin([4334])]

# Separate out group C
gpt3_df = full_df[full_df['group'] == 'C']
full_df = full_df[full_df['group'] != 'C']

# Filter out extra all-human examples
recipe_frequencies = collections.Counter(full_df[full_df["dataset"]=="Recipes"]["true_boundary_index"])
avg_freq_non_final = int(np.round(np.mean([recipe_frequencies[i] for i in range(10)])))

random.seed(2342)
def filter_fn(row):
  if (row["dataset"] == "Recipes") and (row["true_boundary_index"] == 9):
    return random.random() < (avg_freq_non_final / recipe_frequencies[9])
  else:
    return True
full_df = full_df[full_df.apply(filter_fn, axis=1)]
recipe_frequencies = collections.Counter(full_df[full_df["dataset"]=="Recipes"]["true_boundary_index"])

# Filter out all spans of 5 annotations that are the same exact value
def is_percentage(annotations, percentage):
  counter = collections.Counter(annotations)
  percent_list = [(counter[i] / len(annotations)) >= percentage for i in range(10)]
  return any(percent_list)

def sliding_window(df, annotator, percentage, n):
  # This function returns all indices where any window of length n +/- that index is over percentage
  # amount a single value
  indices_to_remove = set()
  ann_list = df[df["annotator"] == annotator].sort_values("date")["predicted_boundary_index"]
  if len(ann_list) < n: 
    return set()
  for i in range(len(ann_list) - n):
    if is_percentage(ann_list.iloc[i:i+n], percentage):
      indices_to_remove = indices_to_remove.union(set(ann_list.iloc[i:i+n].index.tolist()))
  return indices_to_remove

annotators = set(full_df["annotator"].tolist())
indices_to_remove = set()
for a in annotators:
  indices_to_remove = indices_to_remove.union(sliding_window(full_df, a, 1, 5))

full_df = full_df[~full_df.index.isin(indices_to_remove)]

# Merge group C back with A and B
full_df = pd.concat([full_df, gpt3_df])

full_df.to_csv('roft.csv', index=False)