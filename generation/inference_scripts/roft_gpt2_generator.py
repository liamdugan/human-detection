'''
RoFT Generation Script

Example Usage:
  python roft_gpt2_generator.py --dataset nyt --split test --num_gens 1000 --model_name gpt2-xl

Before running this script, make sure you have run the following:
  pip install transformers
  pip install -U spacy
  python -m spacy download en_core_web_trf 
'''

import subprocess, json, time, argparse, os, spacy, math
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm
from datetime import date
from operator import eq
from itertools import chain

def verb_filter(nlp, generation):
  processed_lines = list(nlp.pipe(generation))
  for line in processed_lines:
    pos = [token.pos_ for token in line]
    if "VERB" not in pos and "AUX" not in pos:
      return True
  return False

def cutoff_prompt(prompt, generation):
  ''' This takes in raw prompt text and raw output text made with that prompt
      and determines where the prompt ends and the generation begins '''
  prompt_cutoff = prompt[int(len(prompt)*0.9):]
  start = generation.find(prompt_cutoff)
  if start == -1:
    return generation[start+len(prompt):]
  return generation[start+len(prompt_cutoff):]

def fix_quotation_marks(generation):
  ''' This is a quick postprocessing step we do to improve spacy sentence
  tokenization on output generations. It looks for a line with a single
  quotation mark and tries to attach it to either the previous or next line '''
  for i, s in enumerate(generation):
    if s == '"':
      if i > 0:
        if generation[i-1].count('"') % 2 == 1:
          generation[i-1] += generation[i]
          generation[i] = "REMOVE"
          continue
      if i < len(generation) - 1:
        next_quot_count = generation[i+1].count('"')
        if generation[i+1].count('"') % 2 == 1:
          generation[i+1] = generation[i] + generation[i+1]
          generation[i] = "REMOVE"
          continue
  return [s for s in generation if s != "REMOVE"]

MIN_NUM_SENTS = 10 # The total min number of sentences of prompt + generation
RANDOM_SEED = 42 # The random seed for the generations
TRUNCATE = False # Do we truncate the prompt+generation combinations to MIN_NUM_SENTS or include all output in the json
REJECT_IF_REPETITIVE = True # Do we reject a generation that generates the same exact sentence twice in a row?
SPACY_VERB_FILTER = True # Do we reject a generation that contains an incomplete sentence?

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', help="Dataset ('nyt', 'reddit-stories', 'speeches', or 'wikihow')", type=str, required=True)
parser.add_argument('-s','--split', help="Split ('train','dev','test')", type=str, required=True)
parser.add_argument('-m','--model_name', help="Model Name ('gpt2','gpt2-medium','gpt2-large','gpt2-xl')", type=str, required=True)
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

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, return_dict=True).cuda()
set_seed(RANDOM_SEED)

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['ner', 'lemmatizer'])

# Parse the json file into a dict
with open(local_file_path, 'r') as f:
  data = json.load(f)

# Make sure num_gens isn't larger than the total number of prompts in the file
num_gens = min(args.num_gens, len(data['prompts']))
prompts_per_length = num_gens / MIN_NUM_SENTS
generations = []
failure_causes = [0] * 6

# For each batch of generations
for i in tqdm(range(0, num_gens)):

  # Calculate prompt_length and p_value deterministically from index
  prompt_length = int((i / prompts_per_length) + 1)
  p_value = round(math.floor(float(float(i%prompts_per_length) / (float(prompts_per_length) / 11.0))) / 10.0, 1) if args.vary_p else 0.4

  # If the prompt isn't long enough to support the prompt_length we want, skip it
  if len(data['prompts'][i]) < prompt_length:
    failure_causes[5] += 1
    continue

  # Sample and tokenize the prompt
  prompt = data['prompts'][i][:prompt_length]
  inputs = tokenizer.encode(' '.join(prompt), return_tensors="pt", truncation=True)

  if prompt_length >= MIN_NUM_SENTS:
    generations.append({'prompt': prompt, 'generation': [], 'p': p_value, 'prompt-index': i})
    continue

  # Generate the outputs
  output_sequences = model.generate(
    inputs.to(model.device),
    do_sample=True,
    top_p=min(p_value,1.0),
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    max_length=1024,
    min_length=1024
  )

  # Decode the batched outputs (making sure to skip the special padding token)
  outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]

  generated_text = cutoff_prompt(' '.join(prompt), outputs[0]).strip()
  processed_lines = list(nlp.pipe(generated_text.split('\n\n')))
  generated_sents = list(chain.from_iterable([line.sents for line in processed_lines]))[len(prompt):]
  generation = fix_quotation_marks([str(sent).replace('\n', '') for sent in generated_sents])

  # Reject all generations that don't meet the minimum sentence length requirements
  if len(generation) + len(prompt) < MIN_NUM_SENTS:
    failure_causes[1] += 1
    continue

  truncated = generation[:MIN_NUM_SENTS-len(prompt)]

  # Reject generations with lines that are too short
  if min([len(s) for s in truncated]) <= 3:
    failure_causes[2] += 1
    continue

  # Reject generations that are repetitive
  if REJECT_IF_REPETITIVE and any(map(eq, truncated, truncated[1:])):
    failure_causes[3] += 1
    continue

  if SPACY_VERB_FILTER and verb_filter(nlp, truncated):
    failure_causes[4] += 1
    continue

  if TRUNCATE: generation = truncated

  generations.append({'prompt': prompt, 'generation': generation, 'p': p_value, 'prompt-index': i})

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
  'generation-model': args.model_name,
  'generations': generations
}

with open('generations-{}.json'.format('-'.join([args.model_name, args.dataset, args.split])), 'w') as f:
  json.dump(to_save, f, indent=2)
