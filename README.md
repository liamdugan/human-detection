# Real or Fake Text? Dataset and Analysis

This repository contains the data and code for the AAAI 2023 paper "Real or Fake Text?: Investigating Human Ability to Detect Boundaries Between Human-Written and Machine-Generated Text". In our paper we use a dataset of over 21,000 human annotations of generated text to show that humans can be trained to improve at detection and that certain genres influence generative models to make different types of errors.

To download the data, either clone the repository or use the link [here](https://www.seas.upenn.edu/~ldugan/roft.csv)!

**NEW: You can also now download RoFT through the [HuggingFace Datasets 🤗](https://huggingface.co/datasets/liamdugan/roft) Library**

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

## Citation
If you use our data or analysis code for your research, please cite us as
<pre>
@article{dugan-etal-2023-roft, 
  title="Real or Fake Text?: Investigating Human Ability to Detect Boundaries between Human-Written and Machine-Generated Text",
  author = "Dugan, Liam  and
    Ippolito, Daphne  and
    Kirubarajan, Arun  and
    Shi, Sherry  and
    Callison-Burch, Chris",
  journal="Proceedings of the AAAI Conference on Artificial Intelligence", 
  volume="37", 
  number="11", 
  year="2023", 
  month="Jun.", 
  pages="12763-12771",
  url="https://ojs.aaai.org/index.php/AAAI/article/view/26501", 
  DOI="10.1609/aaai.v37i11.26501", 
}
</pre>
