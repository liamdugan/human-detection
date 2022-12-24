'''
RoFT Generation Script

Example Usage:
  python roft_gpt3_generator.py --dataset nyt --split test --num_gens 1000 --model_name {ada, curie, babbage, davinci}

Before running this script, make sure you have run the following:
  export OPENAI_API_KEY=<your_key>
  pip install -U spacy
  pip install openai
  pip install transformers
  python -m spacy download en_core_web_trf 
'''

import subprocess, json, time, argparse, os, spacy, math, openai
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datetime import date
from operator import eq
from itertools import chain

openai.api_key = os.getenv("OPENAI_API_KEY")

MIN_NUM_SENTS = 10 # The total min number of sentences of prompt + generation
RANDOM_SEED = 42 # The random seed for the generations
TRUNCATE = True # Do we truncate the prompt+generation combinations to MIN_NUM_SENTS or include all output in the json

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', help="Dataset ('nyt', 'reddit-stories', 'speeches', or 'wikihow')", type=str, required=True)
parser.add_argument('-s','--split', help="Split ('train','dev','test')", type=str, required=True)
parser.add_argument('-m','--model_name', help="Model Name ('ada','babbage','curie','davinci')", type=str, required=True)
parser.add_argument('-n','--num_gens', help="The number of generations you would like to output", type=int, required=True)
parser.add_argument('-p','--vary_p', help="Vary the value of the Nucleus Sampling parameter", action='store_true')

args = parser.parse_args()

print(args)

if args.vary_p and args.num_gens < 11 * MIN_NUM_SENTS:
  print("Warning: Please set args.num_gens higher, unable to make equal spread across p values and prompt lengths")
  exit(0)

# Get the name of the prompts json file
filename = 'prompts-{}.json'.format('-'.join([args.dataset, args.split]))
local_file_path = './' + args.dataset + '/' + filename
file_url = 'gs://roft_datasets/prompts/' + args.dataset + '/' + filename

# If we do not have the file already, download from gcloud bucket
if not os.path.exists(local_file_path):
  command = "gsutil cp {0} {1}".format(file_url, local_file_path)
  process = subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()

# Initialize the spacy sentence tokenization model
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['ner', 'lemmatizer'])

# Parse the json file into a dict
with open(local_file_path, 'r') as f:
  data = json.load(f)

# Calculate the average length of a sentence in our dataset in tokens
tokenizer = AutoTokenizer.from_pretrained('gpt2') 
lens = []
for prompt in data['prompts']:
  for s in prompt:
    lens.append(len(tokenizer(s)['input_ids']))

sorted_lens = sorted(lens)
ninety_pcnt_len = sorted_lens[int(0.9*len(sorted_lens))]
avg_len = sum(lens) / len(lens)

# Make sure num_gens isn't larger than the total number of prompts in the file
num_gens = min(args.num_gens, len(data['prompts']))
prompts_per_length = num_gens / MIN_NUM_SENTS
prompt_lens_sampled = dict.fromkeys(range(1,11), 0)
p_values_sampled = {k: dict.fromkeys([0.0, 0.4, 1.0], 0) for k in range(1,11)}
generations = []
failure_causes = [0] * 3

# For each batch of generations
i = -1
pbar = tqdm(total = num_gens)
while len(generations) < num_gens and i < len(data['prompts']):
  i += 1
  # Pick prompt_length and p_value by whatever has least number of gens so far
  prompt_length = min(prompt_lens_sampled, key=prompt_lens_sampled.get)
  p_value = min(p_values_sampled[prompt_length], key=p_values_sampled[prompt_length].get) if args.vary_p else 0.4
  print("Pmpt #{0}; Length {1}; and P Value {2}".format(i, prompt_length, p_value))

  # If the prompt isn't long enough to support the prompt_length we want, skip it
  if len(data['prompts'][i]) < prompt_length:
    failure_causes[0] += 1
    continue

  # Sample the prompt
  prompt = data['prompts'][i][:prompt_length]

  # If we're sampling an all-human prompt, don't bother with generation
  if prompt_length >= MIN_NUM_SENTS:
    prompt_lens_sampled[prompt_length] += 1
    p_values_sampled[prompt_length][p_value] += 1
    generations.append({'prompt': prompt, 'generation': [], 'p': p_value, 'prompt-index': i})
    pbar.update(1)
    continue

  # Generate the outputs
  generated_text = openai.Completion.create(
    engine=args.model_name,
    prompt = '\n\n'.join(prompt),
    top_p = p_value,
    max_tokens = max(int(avg_len * (MIN_NUM_SENTS - prompt_length)), 100),
    frequency_penalty = 1.2,
    presence_penalty = 0,
    logit_bias={"50256": -100} # prevents <|endoftext|> from being generated
  )["choices"][0]["text"].strip()
  
  processed_lines = list(nlp.pipe(generated_text.split('\n\n')))
  generated_sents = list(chain.from_iterable([line.sents for line in processed_lines]))
  generation = [str(sent).replace('\n', '') for sent in generated_sents]

  # Reject all generations that don't meet the minimum sentence length requirements
  if len(generation) + len(prompt) < MIN_NUM_SENTS:
    print("Rejected #{0}: Prompt Len {1} and Gen Len {1} < 10".format(i, len(prompt), len(generation)))
    failure_causes[1] += 1
    continue

  truncated = generation[:MIN_NUM_SENTS-len(prompt)]

  # Reject generations with lines that are too short
  if min([len(s) for s in truncated]) <= 3:
    failure_causes[2] += 1
    continue

  if TRUNCATE: generation = truncated

  # If generation is accepted, increment number in the prompt_len and p_value dict
  prompt_lens_sampled[prompt_length] += 1
  p_values_sampled[prompt_length][p_value] += 1

  generations.append({'prompt': prompt, 'generation': generation, 'p': p_value, 'prompt-index': i})
  pbar.update(1)

print("Failure Rate: {0}/{1}".format(sum(failure_causes), i))
print("Causes:")
print("Prompt too short: " + str(failure_causes[0]))
print("Generation too short: " + str(failure_causes[1]))
print("Line too short: " + str(failure_causes[2]))

# Save the prompts to the json file
to_save = {
  'prompts-file': 'https://storage.googleapis.com/' + file_url[5:],
  'dataset': args.dataset,
  'split': args.split,
  'date-generated': date.today().strftime("%d/%m/%Y"),
  'generation-model': args.model_name,
  'generations': generations
}

with open('generations-{}.json'.format('-'.join([args.model_name, args.dataset, args.split])), 'w') as f:
  json.dump(to_save, f, indent=2)
