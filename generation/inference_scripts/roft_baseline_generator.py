'''
RoFT Generation Script

Example Usage:
  python roft_baseline_generator.py --dataset nyt --split test --num_gens 1000
'''

import subprocess, json, time, argparse, os
from tqdm.auto import tqdm
from datetime import date

MIN_NUM_SENTS = 10 # The total min number of sentences of prompt + generation
TRUNCATE = True # Do we truncate the prompt+generation combinations to MIN_NUM_SENTS or include all output in the json

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', help="Dataset ('nyt', 'reddit-stories', 'speeches', or 'wikihow')", type=str, required=True)
parser.add_argument('-s','--split', help="Split ('train','dev','test')", type=str, required=True)
parser.add_argument('-n','--num_gens', help="The number of generations you would like to output", type=int, required=True)

args = parser.parse_args()

print(args)

# Get the name of the prompts json file
filename = 'prompts-{}.json'.format('-'.join([args.dataset, args.split]))
local_file_path = './' + args.dataset + '/' + filename
file_url = 'gs://roft_datasets/prompts/' + args.dataset + '/' + filename

# If we do not have the file already, download from gcloud bucket
if not os.path.exists(local_file_path):
  command = "gsutil cp {0} {1}".format(file_url, local_file_path)
  process = subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()

# Parse the json file into a dict
with open(local_file_path, 'r') as f:
  data = json.load(f)

# Make sure num_gens isn't larger than 2x the total number of prompts in the file
num_gens = min(args.num_gens, int(len(data['prompts']) / 2))
prompts_per_length = num_gens / MIN_NUM_SENTS
generations = []
failure_causes = [0] * 6

# For each batch of generations
for i in tqdm(range(0, num_gens)):

  # Calculate prompt_length and p_value deterministically from index
  prompt_length = int((i / prompts_per_length) + 1)

  # If either of the prompts aren't long enough to support the prompt_length we want, skip them
  if len(data['prompts'][i]) < prompt_length or len(data['prompts'][i + num_gens]) < MIN_NUM_SENTS - prompt_length:
    failure_causes[5] += 1
    continue

  # Sample the prompt
  prompt = data['prompts'][i][:prompt_length]

  # If the prompt is all human, don't bother sampling the 2nd prompt
  if prompt_length >= MIN_NUM_SENTS:
    generations.append({'prompt': prompt, 'generation': [], 'p': -1, 'prompt-index': i, 'prompt-index-2': i+num_gens})
    continue

  # Sample the second prompt (the "generation") from another article (with index i + num_gens)
  generation = data['prompts'][i + num_gens]

  # Reject all generations that don't meet the minimum sentence length requirements
  if len(generation) + len(prompt) < MIN_NUM_SENTS:
    failure_causes[1] += 1
    continue

  if TRUNCATE: generation = generation[:MIN_NUM_SENTS-len(prompt)]

  generations.append({'prompt': prompt, 'generation': generation, 'p': -1, 'prompt-index': i, 'prompt-index-2': i+num_gens})

print("Failure Rate: " + str(float(sum(failure_causes)) / float(num_gens)))
print("Causes:")
print("Prompt over 256 chars: " + str(failure_causes[0]))
print("Generation too short: " + str(failure_causes[1]))
print("Line too short: " + str(failure_causes[2]))
print("Repetitive: " + str(failure_causes[3]))
print("No Verb Present: " + str(failure_causes[4]))
print("Prompt too short: " + str(failure_causes[5]))

# Save the prompts to the json file
to_save = {
  'prompts-file': 'https://storage.googleapis.com/' + file_url[5:],
  'dataset': args.dataset,
  'split': args.split,
  'date-generated': date.today().strftime("%d/%m/%Y"),
  'generation-model': 'easy',
  'generations': generations
}

with open('generations-{}.json'.format('-'.join(['easy', args.dataset, args.split])), 'w') as f:
  json.dump(to_save, f, indent=2)
