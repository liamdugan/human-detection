import argparse, json, subprocess, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

''' Downloads the prompts json file from the roft GCloud bucket '''
def download_sampling_file(dataset, split):
  filename = 'prompts-{}.json'.format('-'.join([dataset, split]))
  file_url = 'gs://roft_datasets/prompts/' + dataset + '/' + filename
  local_path = './' + dataset + '/' + filename
  if not os.path.exists(local_path):
    command = "gsutil cp {0} {1}".format(file_url, local_path)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()
  return local_path

''' Encode function used to process data before finetuning.
    Each input example is tokenized and then truncated/padded to "max_length" '''
def encode(batch):
    return tokenizer(batch['prompts'], truncation=True, padding=True, max_length=256)

''' Converts the prompts json file into a HuggingFace Dataset object '''
def roft_prompts_json_to_dataset(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
    prompts = [' '.join(x) for x in data['prompts']]
  return Dataset.from_dict(dict(prompts=prompts))

''' Tokenize and preprocess the Dataset object to prep for finetuning '''
def tokenize_dataset(dataset, tokenizer):
  tokenized = dataset.map(encode, batched=True, batch_size=1000)
  tokenized.set_format('torch', columns=['input_ids', 'attention_mask'])
  return tokenized

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', help="Dataset ('nyt', 'reddit-stories', 'speeches', or 'wikihow')", type=str, required=True)
parser.add_argument('-m','--model_name', help="Model Name ('gpt2','gpt2-medium','gpt2-large','gpt2-xl')", type=str, required=True)
args = parser.parse_args()

# Download the prompt files from the roft GCloud bucket
train_local_file_path = download_sampling_file(args.dataset, 'train')
dev_local_file_path = download_sampling_file(args.dataset, 'dev')

# Load the prompt json files into dataset objects
train_dataset = roft_prompts_json_to_dataset(train_local_file_path)
dev_dataset = roft_prompts_json_to_dataset(dev_local_file_path)

# Initialize the tokenizer, model, and data collator
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Tokenize the datasets with the appropriate tokenizer
train_tok = tokenize_dataset(train_dataset, tokenizer)
dev_tok = tokenize_dataset(dev_dataset, tokenizer)

# Initialize the Trainer
training_args = TrainingArguments(
    output_dir='./',
    overwrite_output_dir=True,
    num_train_epochs=1,
    save_total_limit=1,
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=100,
    eval_steps=10000,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tok,
    eval_dataset=dev_tok,
)

# Train the model
trainer.train()

# Save the best model
trainer.save_model('./finetuned')
