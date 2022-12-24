# Real or Fake Text? Dataset and Analysis

This repository contains the data and code for the AAAI 2023 paper "Real or Fake Text?: Investigating Human Ability to Detect Boundaries Between Human-Written and Machine-Generated Text". In our paper we use a dataset of over 21,000 human annotations of generated text to show that humans can be trained to improve at detection and that certain genres influence generative models to make different types of errors.

To download the data, either clone the repository or use the link [here](seas.upenn.edu/~ldugan/roft.csv)!

## Sample Analysis Notebook

To help get started with reading in the dataset, we've provided a sample analysis notebook at `analysis.ipynb`. To run it, install the dependencies in your virtual environment of choice

Conda:
```
conda create --n roft --file requirements.txt
```
venv:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Then run the following command to get your environment hooked up to jupyter

```
python -m ipykernel install --user --name=roft
```

Finally run jupyter notebook
```
jupyter notebook analysis.ipynb
```

## Other Details

The `/generation` folder contains the files used to generate the data for the project, sample the prompts, finetune the models, and filter the generations.

The `/data` folder contains the main dataset as well as the help guide given to students from Group B and C.

Finally, the `/processing` folder contains the code used to process the raw RoFT database dump and filter the final dataset of annotations.

## Reproduction

To reproduce the figures from the paper, simply run the `analysis.ipynb` notebook. 