import subprocess, json, time, argparse, random, copy
from transformers import CTRLTokenizer, TFCTRLLMHeadModel, set_seed
from tqdm.auto import tqdm
import spacy
from spacy.pipeline import Sentencizer
from datetime import date
from operator import eq
from itertools import chain
import math

def verb_filter(generation):
  nlp = spacy.load("en_core_web_sm")
  for line in generation:
    pos = [token.pos_ for token in nlp(str(line))]
    if "VERB" not in pos and "AUX" not in pos:
      return True
  return False

''' https://github.com/salesforce/ctrl/blob/master/control_codes.py '''
CONTROL_CODES = ["Pregnancy","Christianity","Explain","Fitness","Saving","Ask",
                  "Ass","Joke","Questions","Thoughts","Retail","Feminism",
                  "Writing","Atheism","Netflix","Computing","Opinion","Alone",
                  "Funny","Gaming","Human","India","Joker","Diet","Legal",
                  "Norman","Tip","Weight","Movies","Running","Science",
                  "Horror","Confession","Finance","Scary","Support",
                  "Technologies","Teenage","Event","Learned","Notion","Wikipedia",
                  "Books","Extract","Confessions","Conspiracy","Links","Narcissus",
                  "Relationship","Relationships","Reviews","News","Translation",
                  "multilingual"]

def ctrl_process_prompt(control_code, prompt):
  if not control_code: control_code = random.choice(CONTROL_CODES)
  prompt[0] = control_code + ' Title: ' + prompt[0]
  if len(prompt) > 1: prompt[1] = 'Text: ' + prompt[1]
  return control_code, prompt

def cutoff_prompt(prompt, generation):
  ''' This takes in raw prompt text and raw output text made with that prompt
      and determines where the prompt ends and the generation begins '''
  prompt_cutoff = prompt[int(len(prompt)*0.9):]
  start = generation.find(prompt_cutoff)
  if start == -1:
    # print("Error: couldn't find the substring!")
    # print(repr(prompt))
    # print(repr(generation))
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
parser.add_argument('-n','--num_gens', help="The number of generations you would like to output", type=int, required=True)
parser.add_argument('-m','--model_name', help="The CTRL model you would like to generate with ('ctrl', 'sshleifer/tiny-ctrl')")
parser.add_argument('-p','--vary_p', help="Vary the value of the Nucleus Sampling parameter", action='store_true')
parser.add_argument('-c','--control_code', help="Control code for CTRL ('Politics','Gaming','Books', etc.)", type=str, default='')

args = parser.parse_args()

print(args)

if args.vary_p and args.num_gens < 11 * MIN_NUM_SENTS:
  print("Warning: Please set args.num_gens higher, unable to make equal spread across p values and prompt lengths")
  exit(0)

if args.control_code and args.control_code not in CONTROL_CODES and args.control_code != "Politics":
  print("Warning: Invalid control code selection, please select a valid control code")
  exit(0)

# Download the sampling file from the Google Cloud bucket
filename = 'prompts-{}.json'.format('-'.join([args.dataset, args.split]))
file_url = 'gs://roft_datasets/prompts/' + args.dataset + '/' + filename
local_file_path = './' + args.dataset + '/' + filename
command = "gsutil cp {0} {1}".format(file_url, local_file_path)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

# Initialize the tokenizer and the model
tokenizer = CTRLTokenizer.from_pretrained(args.model_name)
model = TFCTRLLMHeadModel.from_pretrained(args.model_name, return_dict=True)

# Calculate the 99 percentile length of a prompt with MIN_NUM_SENTS sentences in it
with open(local_file_path, 'r') as f:
  data = json.load(f)
  tokenized_prompts = []
  for prompt in data['prompts']:
    inputs = tokenizer(' '.join([args.control_code] + prompt[:MIN_NUM_SENTS]), return_tensors="tf")
    tokenized_prompts.append(float(len(inputs['input_ids'][0])) / float(MIN_NUM_SENTS))
sorted_lens = sorted(tokenized_prompts)
ninety_percentile_sent_len = sorted_lens[int(0.90*len(sorted_lens))]

nlp = spacy.load("en")

set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

generations = []
failure_causes = [0] * 6
with open(local_file_path, 'r') as f:
  data = json.load(f)

  # Calculate some important values
  num_gens = len(data['prompts'][:args.num_gens])
  prompts_per_length = num_gens / MIN_NUM_SENTS

  # For each batch of generations
  for i in tqdm(range(0, num_gens)):

    # Calculate prompt_length and p_value deterministically from index
    prompt_length = int((i / prompts_per_length) + 1)
    p_value = round(math.floor(float(float(i%prompts_per_length) / (float(prompts_per_length) / 11.0))) / 10.0, 1) if args.vary_p else 0.4

    # If the prompt isn't long enough to support the prompt_length we want, continue
    if len(data['prompts'][i]) < prompt_length:
      failure_causes[5] += 1
      continue

    # Sample and tokenize the prompts for this batch
    raw_prompt = data['prompts'][i][:prompt_length]
    control_code, prompt = ctrl_process_prompt(args.control_code, copy.deepcopy(raw_prompt))
    inputs = tokenizer.encode(' '.join(prompt), return_tensors="tf")

    # Calculate the max_length for this prompt using the 90th percentile sentence length in the corpus
    longest_prompt_len = max([len(ids) for ids in inputs])
    max_len = longest_prompt_len + int((MIN_NUM_SENTS - prompt_length) * ninety_percentile_sent_len)

    if longest_prompt_len >= 256:
      failure_causes[0] += 1
      continue

    if prompt_length >= MIN_NUM_SENTS:
      generations.append({'prompt': raw_prompt, 'generation': [], 'p': p_value, 'prompt-index': i, 'control-code': control_code})
      continue

    # Generate the outputs
    output_sequences = model.generate(
      inputs,
      do_sample=True,
      top_p=min(p_value,1.0),
      top_k=0,
      repetition_penalty=1.2,
      pad_token_id=tokenizer.eos_token_id,
      max_length=min(max_len, 256)
    )

    # Decode the batched outputs (making sure to skip the special padding token)
    outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]

    generated_text = cutoff_prompt(' '.join(prompt), outputs[0]).strip()
    processed_lines = [nlp(line) for line in generated_text.split('\n\n')]
    generated_sents = list(chain.from_iterable([line.sents for line in processed_lines]))
    generation = fix_quotation_marks([str(sent).replace('\n', '') for sent in generated_sents])

    # Reject all generations that don't meet the minimum sentence length requirements
    if len(generation) + len(prompt) < MIN_NUM_SENTS:
      failure_causes[1] += 1
      continue

    # calculate truncated generation based on MIN_NUM_SENTS
    truncated = generation[:MIN_NUM_SENTS-len(prompt)]

    # Reject generations with lines that are too short
    if min([len(s) for s in truncated]) <= 3:
      failure_causes[2] += 1
      continue

    # Reject generations that are repetitive
    if REJECT_IF_REPETITIVE and any(map(eq, truncated, truncated[1:])):
      failure_causes[3] += 1
      continue

    if SPACY_VERB_FILTER and verb_filter(truncated):
      failure_causes[4] += 1
      continue

    # Truncate the generation if we want to truncate it
    if TRUNCATE: generation = truncated

    generations.append({'prompt': raw_prompt, 'generation': generation, 'p': p_value, 'prompt-index': i, 'control-code': control_code})

  print("Failure Rate: " + str(float(sum(failure_causes)) / float(num_gens)))
  print("Causes:")
  print("Prompt over 256 chars: " + str(failure_causes[0]))
  print("Generation too short: " + str(failure_causes[1]))
  print("Line too short: " + str(failure_causes[2]))
  print("Repetitive: " + str(failure_causes[3]))
  print("No Verb Present: " + str(failure_causes[4]))

# Save the prompts to the json file
to_save = {
  'prompts-file': 'https://storage.googleapis.com/' + file_url[5:],
  'dataset': args.dataset,
  'split': args.split,
  'date-generated': date.today().strftime("%d/%m/%Y"),
  'generation-model': args.model_name,
  'generations': generations
}

control_code = args.control_code if args.control_code else "nocode"
with open('generations-{}.json'.format('-'.join([args.model_name, control_code, args.dataset, args.split])), 'w') as f:
  json.dump(to_save, f, indent=2)
