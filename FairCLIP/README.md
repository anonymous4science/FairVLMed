# FairVLMed

The dataset and code for the paper entitled *FairVLMed Dataset: Harnessing Fairness in Vision-and-Language Learning via FairCLIP*. 

# Dataset

The FairVLMed dataset can be accessed via this [link](https://drive.google.com/drive/folders/1mwgnwnk-TTzTV9UFThwE7cAryUodUpiP?usp=sharing). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The FairVLMed dataset comprises 10,000 samples from 10,000 subjects. It is divided into 7,000 training, 1,000 validation, and 2,000 test samples. Upon downloading and extracting these datasets, you will find the dataset structure as follows.

```
FairVLMed
├── data
├── gpt4_summarized_notes.csv
├── med42_summarize.csv
├── meta_all.csv
├── pmc-llama_summarize.csv
└── split_files.csv
```
The file split_files.csv details the division of data into training, validation, and testing sets. The data folder contains 10,000 NPZ files named in the format "data_xxxxxx.npz", where "xxxxxx" (e.g., 006691) is a unique numeric ID. The file meta_all.csv provides metadata (such as race, gender, ethnicity, marital status, age, and preferred language) for each NPZ file. Additionally, the files gpt4_summarized_notes.csv, med42_summarize.csv, and pmc-llama_summarize.csv contain notes summarized by GPT-4, Med42, and PMC-LLAMA, respectively.

Each NPZ file includes keys for 'fundus_slo', 'md', 'tds', 'note', 'note_ext', 'age', 'gender', 'race', 'ethnicity', 'language', 'maritalstatus', and 'glaucoma'.


# Abstract

Fairness is a critical concern in deep learning, especially in healthcare, where these models influence diagnoses and treatment decisions. Although fairness has been investigated in the vision-only domain, the fairness of medical vision-and-language (VL) models remains unexplored due to the scarcity of medical VL datasets for studying fairness. To bridge this research gap, we introduce the first fair vision-and-language medical dataset (\textit{FairVLMed}) that provides detailed demographic attributes, ground-truth labels, and clinical notes to facilitate an in-depth examination of fairness within VL foundation models. Using \textit{FairVLMed}, we conduct a comprehensive fairness analysis of two widely-used VL models (CLIP and BLIP2), pre-trained on both natural and medical domains, across four different protected attributes. Our results highlight significant biases in all VL models, with Asian, Male, Non-Hispanic, and Spanish being the preferred subgroups across the protected attributes of race, gender, ethnicity, and language respectively. In order to alleviate these biases, we propose FairCLIP, an optimal-transport based approach that achieves a favorable trade-off between performance and fairness by reducing the Sinkhorn distance between the overall sample distribution and the distributions corresponding to each demographic group. As the first VL dataset of its kind, FairVLMed holds the potential to catalyze advancements in the development of machine learning models that are both ethically aware and clinically effective.

# Requirements

To install the prerequisites, run:

```
pip install - r requirements.txt
```

# Experiments

To run the experiments for zero-shot transfer with CLIP, execute:

```
./scripts/finetune_CLIP.sh
```

To run the experiments for zero-shot transfer with FairCLIP, execute:

```
./scripts/finetune_FairCLIP.sh
```

To evaluate the models pre-trained in the above processes, execute:

```
./scripts/evaluate_CLIP.sh
```