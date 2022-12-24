# RoFT Generation
Example generation for RoFT is split up into three sections. Pre-processing, Prompt Sampling, and Language Model Inference


## Step 1: Pre-Processing
Scripts for data pre-processing can be found in the `process_datasets` folder.

These scripts process the raw corpora to be one article per line and split it into train, dev, and test. Certain pre-processing scripts also filter out web text artifacts and detokenize.

All of the pre-processed datasets are publicly hosted on our google cloud bucket `gs://roft_datasets/data`

### Example Usage
    python process-NYT.py

## Step 2: Prompt Sampling
In order to sample prompts, run the `ROFT_Prompt_Sampler.ipynb` notebook. The notebook will automatically download the pre-processed corpora from our google cloud bucket and output a json file of prompts. Just edit the preferences listed below
```
DATASET = 'speeches'
SPLIT = 'test'
NUM_SAMPLES = 1000000
MAX_LEN = 10
PERCENT_MAX = 1.0
BOW_FILTER_TOGGLE = False
VERB_FILTER_TOGGLE = False
REJECT_TOO_SHORT = False
```
### Prompt Output Format
`prompts-speeches-dev.json`
```
{
  "sample-file": "https://storage.googleapis.com/roft_datasets/data/speeches/dev.txt",
  "dataset": "speeches",
  "split": "dev",
  "date-sampled": "dd/mm/yyyy",
  "prompts": [
    [
      "\"Remarks at U. S. Air Force Academy\" by President John F. Kennedy on June 5, 1963.",
      ...
      "\"It is signed \"Sincerely, Cadet Marvin B. Hopkins,\" who's obviously going to be a future General."
    ],
    ...
  ]
}
```
All sampled prompts used are publicly hosted on our google cloud bucket `gs://roft_datasets/prompts`

## Step 3: Language Model Inference
LM Inference is done by running one of the inference scripts found at `inference_scripts/`. These scripts automatically download prompt json files from the google cloud bucket and output a generation json file. You can edit the `dataset`, `split`, and `model_name` arguments to customize the model size and the dataset.

`roft_gpt2_generator.py` requires PyTorch and `roft_ctrl_generator.py` requires Tensorflow

All generations used in the RoFT project are publicly hosted on our google cloud bucket `gs://roft_datasets/generations_v3`

### Example Usage
    python roft_gpt2_generator.py --dataset nyt --split test --num_gens 1000 --model_name gpt2-xl

### Generation JSON Output Format
`generations-gpt2-xl-nyt-dev.json`
```
{
  "prompts-file": "https://storage.googleapis.com/roft_datasets/prompts/nyt/prompts-nyt-dev.json",
  "dataset": "nyt",
  "split": "dev",
  "date-generated": "dd/mm/yyyy",
  "generation-model": "gpt2-xl",
  "generations": [
    {
      "prompt": [ ... ],
      "generation": [ ... ],
      "p": 0.0,
      "prompt-index": 0
    },
    ...
  ]
}
```

## Step 4: Language Model Fine-Tuning
LM Fine-Tuning is done by running the script found at `finetuning_scripts/roft_gpt2_finetuning.py`. This script automatically downloads prompt json files from the roft google cloud bucket and outputs a fine-tuned model at `./finetuned`. You can edit the `dataset`, and `model_name` arguments to customize the model size and the dataset.

Training is performed on the `train` split of the given dataset and the model is periodically evaluated on the `dev` split of the dataset. The checkpoint that performs the best on the `dev` split will be saved as the final output.

The default arguments for fine-tuning are listed below, feel free to customize any arguments with a `*` next to the name as these settings can (and should) vary depending on the available disk space and GPU memory of your machine.
```
  output_dir='./',
  overwrite_output_dir=True,
  num_train_epochs=1,
* save_total_limit=1,
  evaluation_strategy="steps",
* per_device_train_batch_size=1,
* per_device_eval_batch_size=1,
  logging_steps=100,
* eval_steps=10000,
  weight_decay=0.01,
  logging_dir='./logs',
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss"
```

All fine-tuned models used in the RoFT project are publicly hosted on our google cloud bucket `gs://roft_saved_models/`

### Example Usage
    python roft_gpt2_finetuning.py --dataset recipes --model_name gpt2-xl

## Step 5: Interactive Model Testing
If at any point you want to experiment with generation parameters you can run `interactive_test.py` to view a model's output in an interactive manner. This will allow you to tune parameters such as `temperature`, `top_p`, `repetition_penalty`, and others to ensure highest quality generations.

This script supports both pre-trained models `gpt2`, `gpt2-xl`, `ctrl` and fine-tuned models.

### Example Usage
    python interactive_test.py --model_name ./finetuned

## Data Sources
1. [New York Times Annotated Corpus (Sandhaus, 2008)](https://catalog.ldc.upenn.edu/LDC2008T19)
2. [Reddit Writing Prompts (Fan et al., 2018)](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz)
3. [Corpus of Presidential Speeches (Brown, 2016)](http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus)
4. [Recipe1M+ (Marin et al., 2019)](http://pic2recipe.csail.mit.edu/)

## Language Models
1. [GPT2 (Radford et al., 2019)](https://openai.com/blog/better-language-models/)
2. [GPT2-XL (Radford et al., 2019)](https://openai.com/blog/better-language-models/)
3. [CTRL (Keskar et al., 2019)](https://blog.einstein.ai/introducing-a-conditional-transformer-language-model-for-controllable-generation/)
