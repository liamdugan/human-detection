import json
import pandas as pd

with open('6-21-22.json', 'r') as f:
    next(f) # Skip first line of file because "READING from remote database"
    db = json.load(f)

def normalize_df(df):
    return pd.json_normalize(df.fields).assign(pk=df.pk.values)

df = pd.DataFrame(db)

# Load database entries into dataframes
annotation_df = normalize_df(df[df.model == 'core.annotation'])
generation_df = normalize_df(df[df.model == 'core.generation'])
prompt_df = normalize_df(df[df.model == 'core.prompt'])
decstrat_df = normalize_df(df[df.model == 'core.decodingstrategy'])
playlist_df = normalize_df(df[df.model == 'core.playlist'])
user_df = normalize_df(df[df.model == 'auth.user'])

# Modify column names to avoid duplicates across tables
prompt_df = prompt_df.rename(columns={"body": "prompt_body"})
generation_df = generation_df.rename(columns={"body": "gen_body"})
decstrat_df = decstrat_df.rename(columns={"name": "dec_strat", "value": "dec_strat_value"})
annotation_df["date"] = pd.to_datetime(annotation_df["date"])

# Merge tables together
full_df = annotation_df.join(generation_df.set_index('pk'), on='generation')
full_df = full_df.join(prompt_df.set_index('pk'), on='prompt')
full_df = full_df.join(decstrat_df.set_index('pk'), on='decoding_strategy')
full_df = full_df.join(user_df.set_index('pk'), on='annotator')

# Process playlist information (dataset name and group of annotation)
gen_to_playlist = {}
for _, row in playlist_df.iterrows():
  for gen_id in row["generations"]:
    gen_to_playlist[gen_id] = (row["shortname"], row["version"])

full_df['playlist_name'] = full_df['generation'].apply(lambda x: gen_to_playlist[x][0])
full_df['playlist_version'] = full_df['generation'].apply(lambda x: gen_to_playlist[x][1])

# Add in survey results
survey_df = pd.read_csv('survey.csv')
full_df = full_df.join(survey_df.set_index('username'), 'username', how="inner")

# Remove all annotators that haven't *explicitly* given us consent to use data
full_df = full_df[full_df['agreed_to_research'] == 'Yes']

# Remap survey responses to be labels from 1 to 4
def remap_familiarity_labels(x):
  if x == "I've never heard of them.":
    return 1
  elif x == "I've read about them in the news or a blog post.":
    return 2
  elif x == "I’ve been excitedly following them.":
    return 3
  elif x == "I’ve used them before (either with the OpenAI API, HuggingFace Transformers, etc.).":
    return 4
  else:
    return -1

def remap_genre_fam_labels(x):
  if x == "Never":
    return 1
  elif x == "Once to a few times per year":
    return 2
  elif x == "Once to a few times per month":
    return 3
  elif x == "Once to a few times per week":
    return 4
  elif x == "Daily":
    return 5
  else:
    return -1

full_df['recipe_familiarity'] = full_df['recipe_familiarity'].apply(remap_genre_fam_labels)
full_df['news_familiarity'] = full_df['news_familiarity'].apply(remap_genre_fam_labels)
full_df['stories_familiarity'] = full_df['stories_familiarity'].apply(remap_genre_fam_labels)
full_df['gen_familiarity'] = full_df['gen_familiarity'].apply(remap_familiarity_labels)

# Recover CTRL labels because we did not originally save them in the DB
CTRL_GEN_FILES_NOCODE = ['generations/generations-ctrl-nocode-speeches-dev.json', 
                          'generations/generations-ctrl-nocode-speeches-train.json', 
                          'generations/generations-ctrl-nocode-speeches-test.json']
CTRL_GEN_FILES_CODE = ['generations/generations-ctrl-Politics-speeches-dev.json', 
                       'generations/generations-ctrl-Politics-speeches-train.json', 
                       'generations/generations-ctrl-Politics-speeches-test.json']

reverse_lookup_dict = dict()
for fname in CTRL_GEN_FILES_CODE:
  with open(fname, 'r') as f:
    gen_data = json.load(f)
    for gen in gen_data['generations']:
      # Insert the first generated sentence into the dictionary as key. 
      # Value is a boolean. True = generation was made w/ Politics control code
      # Reasoning for try block is in the case of all-human examples
      try: reverse_lookup_dict[gen['generation'][0]] = True
      except IndexError: pass

for fname in CTRL_GEN_FILES_NOCODE:
  with open(fname, 'r') as f:
    gen_data = json.load(f)
    for gen in gen_data['generations']:
      # Insert the first generated sentence into the dictionary as key. 
      # Value is a boolean. True = generation was made w/ Politics control code
      try: reverse_lookup_dict[gen['generation'][0]] = False
      except IndexError: pass

# Use the reverse lookup dict to reassign labels for the CTRL generations
# (If all human then randomly assign to either)
def reassign_ctrl_label(x):
  new_model_label = x['system']
  if x['system'] == 'ctrl':
    if x['gen_body']:
      if reverse_lookup_dict[x['gen_body'].split('_SEP_')[0]]:
        new_model_label = 'ctrl-Politics'
      else:
        new_model_label = 'ctrl-nocode'
    else:
      new_model_label = 'human'
  return new_model_label

full_df['system'] = full_df.apply(reassign_ctrl_label, axis=1)

# Rename model to "human" if all human and rename "easy" to "baseline"
full_df['system'] = full_df.apply(lambda x: x['system'] if x['gen_body'] else 'human', axis=1)
full_df['system'] = full_df['system'].apply(lambda x: 'baseline' if x == 'easy' else x)

# Recover information about groups A, B, and C
full_df = full_df[full_df['playlist_version'].isin(['0.2', '0.6'])]
full_df['group'] = full_df['date'] < '2021-10-1'
full_df['group'] = full_df['group'].apply(lambda x: 'A' if x else 'B')
full_df['group'] = full_df.apply(lambda x: x['group'] if x['playlist_version'] == '0.2' else 'C', axis=1)

# Drop columns we do not need
columns_to_drop = ['attention_check', 'dec_strat', 'password', 'last_login', 'is_superuser', 'is_staff', 'first_name', 
                   'last_name', 'email', 'is_active', 'groups', 'user_permissions', 'agreed_to_research', 'dataset', 
                   'decoding_strategy', 'Timestamp', 'Email Address', 'major', 'date_joined', 'source_prompt_index', 
                   'username', 'playlist', 'playlist_version']
full_df = full_df.drop(columns_to_drop, axis=1)

# Rename columns and edit boundary index to make more sense
full_df = full_df.rename(columns={'system':'model','playlist_name':'dataset', 'boundary':'predicted_boundary_index', 'num_sentences':'true_boundary_index', 'english':'native_speaker'})
full_df['true_boundary_index'] = full_df['true_boundary_index'] - 1

# Reorder columns
column_ordering = ['date', 'model', 'dataset', 'annotator', 'group', 'dec_strat_value', 'predicted_boundary_index', 'true_boundary_index', 
                   'points', 'reason', 'prompt', 'prompt_body', 'generation', 'gen_body', 'recipe_familiarity', 'news_familiarity', 'stories_familiarity', 
                   'gen_familiarity', 'native_speaker', 'read_guide']
full_df = full_df[column_ordering]

# Write to csv
full_df.to_csv('unfiltered_data.csv', index=False)