# Unified Space for Diffusion Language Models
This repository is an effort to provide a large variety of popular datasets and models pertaining to Molecular data processed using Diffuison Language Models. We present a state-of-the-art architectures for Text-to-Molecule generation and Molecule captioning. 

---
## Diffusion Language Models
1. Text-to-Molecule Generation : This model provides a DLM architecture for 1D SMILES string generation using textual description as a context. The model in its default configuration has 180M trainable parameters. It involves the use of pretrained Scibert for text encoding. The directory comes with its own set of instructions to prepare the data, train models and finetune them on different datasets.
2. Molecule-to-Text Generation : This model provides a DLM architecture for text generation using 1D SMILES strings as the context. The model canbe utilized for Molecule Captioning or Molecule Property Prediction tasks. In this architecture, there is no need of a pretrained model, but it still uses the pretrained Scibert tokenizer. The directory has the instructions to prepare the data, train the model and then finetune it.

--- 
## Data
This repository presents three poppular 1D molecular SMILES datasets, with their corresponding descriptions. Each dataset comes with its own unqiue property mostly differing on the type of descriptions present in them. The datasets find their application in various applications like Molecule Captioning, Molecule Property Preciction, Text-to-Molecule generation or Mulitmodal learning. The specific information about the datasets are present in their respective directories. Here, we present a summary of the datasets : 

| Dataset            | Num Samples | Sample                                   |
|--------------------|-------------|------------------------------------------|
| ChEBI-20           | 33,010      |                                          |
| MolTextNet         | 2.45M       |                                          |
| PubChemSTM         | 77,700      |                                          |

---
## Repository Structure
The following strcuture is provided to help in locating the location of various deliverables provided by this repository:

    .
    ├── Code                     # Contains the code files for the model architectures
    │   ├── Molecule captioning
    │   └── Molecule generation
    ├── Data                     # Contains the datasets along with their usage instructions
    │   ├── ChEBI20
    │   ├── MolTextNet
    │   ├── PubChemSTM
    │   └── README.md
    ├── figures
    └── README.md
---
