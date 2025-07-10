# Molecule Captioning with Diffusion Language Model


- The code was originally proposed for Text-Guided Molecule Generation. We have altered the pipeline to generate molecule captions using similar DLM architecture.
- We have also extended the code to finetune stored checkpoints using cross entropy loss for given number of steps. 

## Usage

### Prerequisites

1. For running the code on your own dataset, it is required to convert the data into `.txt` files, with the headers: CID`\t`SMILES`\t`descriptions. (By default, [ChEBI-20](https://github.com/2023csb1141/IITD-Summer-Intern/tree/main/Data/ChEBI20) Data is already in this format and can be used directly.)
2. Download [Scibert](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it into file `scibert`.

### Training
1. `cd improved-diffusion; cd scripts`
2. Encode text input `python process_text.py -i train_val_256; python process_text.py -i test`
3. Train model: `python train.py`

### Finetuning
We extend the given code to enable finetuning of pretrained checkpoints using cross entropy loss.
1. `cd improved-diffusion; cd scripts`
2. In `finetune.py`, set the path to the pretrained model in the `resume_checkpoint` argument, and set `lr_anneal_steps` to the maximum number of steps you intend to run (i.e. lr_anneal_steps = pretrained steps + finetuning steps).
3. Encode text input `python process_text.py -i train_val_256; python process_text.py -i test`
4. Finetune model: `python finetune.py`

### Sampling
1. `python text_sample.py; python post_sample.py`. Whenever this file is run, a new output named 'MODELOUTPUT.txt' is saved.

### Evaluation
You can evaluate all metrics except for Text2Mol by runnning `ev_reverse.py`. For Text2Mol please go to [MolT5](https://github.com/blender-nlp/MolT5) for more details.

## Requirements

- python3
- pytorch 2.0
- transformers (Be careful to follow the readme installation exactly.)

## Acknowledge
The original work is cited as follows:
```
@article{gong2024text,
  title={Text-Guided Molecule Generation with Diffusion Language Model},
  author={Gong, Haisong and Liu, Qiang and Wu, Shu and Wang, Liang},
  volume={38},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/27761},
  DOI={10.1609/aaai.v38i1.27761},
  number={1},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
  month={Mar.},
  pages={109-117}
}
```
