# MolTextNet Dataset

The **MolTextNet** dataset is a processed and refined version of the [ChEMBL](https://www.ebi.ac.uk/chembl/) database. It contains molecule-description pairs, designed to support tasks such as molecular captioning, text-to-molecule generation, and multimodal representation learning.

---

## Dataset Summary

- **Samples**: ~33,000
- **Format**: Text File (each row contains aunique Compund ID (CID), SMILES and corresponding description)
- **License**: [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/)

---
## Data Split:

 The dataset consists of 33,010 molecules after filtering the SMILES strings whose length is contained within 256. This dataset is further divided into train, validation and test set in the ratio of **80%/10%/10%**.

---
## Folder Structure

- [`generate_vocab.txt`](generate_vocab.txt):  
  This file contains the complete vocabulary extracted from the SMILES strings in the ChEBI20 dataset. It can be used to train a tokenizer or embedder specifically for this chemical space.

- [`mini.txt`](mini.txt):  
  A small subset of the dataset, useful for quick debugging, model prototyping, or understanding the data format.

- [`test.txt`](test.txt):  
  Contains the test split of the dataset. This is used for evaluating final model performance.

- [`train.txt`](train.txt):  
  The primary training split. It consists of the majority of samples from the dataset and is used for fitting the model.

- [`train_val_256.txt`](train_val_256.txt):  
  A reduced training + validation set with filtered samples having maximum SMILES length of 256.

- [`validation.txt`](validation.txt):  
  Contains the validation split used to monitor training performance and tune hyperparameters.

- [`validation_256.txt`](validation_256.txt):  
  A reduced validation set with filtered samples having maximum SMILES length of 256.

---
## Reference

The ChEBI20 dataset presented here was directly obtained from the official [TGM-DLM repository](https://github.com/Gong-Haisong/TGM-DLM), where it was originally curated and preprocessed by the authors. No additional modifications were made beyond their released version.

> Gong, H., Liu, Q., Wu, S., & Wang, L. (2024).  
> *Text-Guided Molecule Generation with Diffusion Language Model*.  
> In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 1, pp. 109–117).  
> [https://doi.org/10.1609/aaai.v38i1.27761](https://doi.org/10.1609/aaai.v38i1.27761)
