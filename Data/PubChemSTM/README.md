# PubChemSTM Dataset

The **PubChemSTM** dataset is a large-scale collection of **77,700** SMILES–text pairs, designed for training and evaluating models on tasks such as molecular captioning, text-to-molecule generation, and multimodal learning. In many cases, this dataset is preffered for pretraining the models, followed by finetuning on a refined dataset, like ChEBI-20.

The original data was obtained using the scripts provided in the official [MoleculeSTM repository](https://github.com/THUDM/MoleculeSTM), and subsequently curated through additional preprocessing steps described below.

---

## Data Filtering & Curation

We applied further filtering to the original raw dump to enhance data quality:

- Removed molecules with **highly repetitive or boilerplate descriptions**.
- Discarded entries with **non-informative chemical descriptions**, such as those merely noting the molecule’s source (e.g., “found in a plant”).
- Ensured all remaining entries had **valid SMILES strings** and **meaningful chemical annotations**.

These refinements result in a cleaner dataset better suited for molecule–language tasks such as captioning, generation, and retrieval.

---

## Folder Structure

- [`train_data.txt`](train_data.txt):  
  Contains **77,700** filtered SMILES–text pairs, with each line formatted as:  
  `<CID>\t<SMILES>\t<description>`

- [`vocab.txt`](vocab.txt):  
  SMILES vocabulary extracted from the dataset. Useful for tokenizer training or embedding lookups.

---

## Applications

This dataset supports a wide range of tasks:

- SMILES-to-text generation
- Text-to-SMILES generation
- Cross-modal molecule–text retrieval
- Pretraining/fine-tuning multimodal models (e.g., MoleculeSTM, MolT5)

---
## Data Usage Instruction

You can directly download the text files of the data by cloning the repository. The files contain all the data samples together with their CID, SMILES string and text description.

---
## Reference

The original data collection process was based on:

> Liu, S., Nie, W., Wang, C., Lu, J., Qiao, Z., Liu, L., Tang, J., Xiao, C., & Anandkumar, A. (2023).  
> *Multi-modal molecule structure–text model for text-based retrieval and editing*.  
> *Nature Machine Intelligence*. Published online December 18, 2023.  
> [https://doi.org/10.1038/s42256-023-00759-6](https://doi.org/10.1038/s42256-023-00759-6)
