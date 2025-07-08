# MolTextNet Dataset

The **MolTextNet** dataset is a processed and refined version of the [ChEMBL](https://www.ebi.ac.uk/chembl/) database. It contains molecule-description pairs, designed to support tasks such as molecular captioning, text-to-molecule generation, and multimodal representation learning.

The owners used GPT-4 series models to generate coherent scientific descriptions from molecular annotations where each molecule was represented as a structured dictionary of property-value pairs, integrating structural features, physicochemical properties, bioactivity profiles, and synthesis information from ChEMBL35 and curated sources, further using the information to generate detailed textual description representative of all the above mentioned aspects.

---

## Dataset Summary

- **Samples**: ~2.47M
- **Description length**: ***Avg*** xxxx, ***Max*** xxxx
- **SMILES length**: ***Avg*** 54.58
- **Format**: CSV File (each row contains aunique Compund ID (CID), SMILES and corresponding description)

![SMILES length Distribution](https://github.com/2023csb1141/IITD-Summer-Intern/blob/main/figures/smile_lens.png)

---
## Data Processing

We calculated statistics on SMILES string lengths and then removed the samples longer than 256 characters, which was a very small portion of the data(xxx number). Further, we observed a common patten among the descriptions. A lot of descriptions had the SMILES string and CID contained within the text. We further processed the descriptions to make them useful for training molecule generation and captioning tasks, which aligns them with the kind of description presented in the other included datasets. As a part of this processing, we removed the SMILES strings and CID from the descriptions, and further tried to remove the samples wherein the decription was inconsistent with respect to the given moelcule. 

---
## Folder Structure

- [`altered_textual_desc.zip`](altered_textual_desc.zip):  
  A set of xxxx molecules in SMILES representation along with their textual descriptions.

---
## Reference

The MolTextNet dataset presented here was directly obtained from the official [repository](https://huggingface.co/datasets/liuganghuggingface/moltextnet), where it was originally curated and preprocessed by the authors.

> Zhu, Y., Liu, G., Inae, E., & Jiang, M. (2025).  
> *MolTextNet: A Two-Million Molecule-Text Dataset for Multimodal Molecular Learning*.  
> arXiv preprint arXiv:2506.00009.  
> [https://arxiv.org/abs/2506.00009](https://arxiv.org/abs/2506.00009)
