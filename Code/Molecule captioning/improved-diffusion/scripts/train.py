"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from mydatasets import get_dataloader, rev_dataloader, ChEBIdataset, ChEBIdataset_rev
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")

wandb.login(key="e3a69f8a6022945939112819f049b0ebd8fffebf")

def main_worker(rank,world_size, reverse = False):
    args = create_argparser(reverse).parse_args()
    set_seed(args.seed)

    if rank == 0:
        wandb.init(
            project = "DiffusionLMRegexAug",
            config = args.__dict__,
        )
        print(wandb.config)

    print('Wandb initialized')
    dist_util.setup_dist(rank = rank, world_size = world_size) 
    print("creating model and diffusion...")
    smtokenizer = regexTokenizer(max_len=256)

    if reverse:
        tokz = AutoTokenizer.from_pretrained('../../scibert')
    else:
        tokz = smtokenizer

    model = TransformerNetModel2(
        in_channels=args.model_in_channels,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=args.model_model_channels,
        dropout=args.model_dropout,
        use_checkpoint=args.use_checkpoint,
        config_name= args.config_name,
        training_mode=args.training_mode,
        vocab_size=len(tokz),
        experiment_mode= args.experiment_mode,
        logits_mode=args.logits_mode,
        hidden_size = args.model_hidden_size,
        num_attention_heads=args.model_num_attention_heads,
        num_hidden_layers = args.model_num_hidden_layers,
        reverse = reverse
    )

    if rank==0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Model total prameter number is :', pytorch_total_params)
        print('Tokenizer vocab length:',len(smtokenizer))

    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(args.diffusion_steps)],
        betas=gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps),
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=args.rescale_timesteps,
        model_arch = args.model_arch,
        training_mode = args.training_mode,
        reverse = reverse
    )

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    print('load data', '*'*50)
    if reverse:
        train_dataset = ChEBIdataset_rev(
            dir='../../datasets/SMILES/',
            smi_tokenizer=smtokenizer,
            split='train_val_256',
            replace_desc=False,
            corrupt_prob=0.,
            mask_desc=False
            # pre = pre
        )
        dataloader = rev_dataloader(train_dataset,args.batch_size,rank,world_size)
    else:
        train_dataset = ChEBIdataset(
            dir='../../datasets/SMILES/',
            smi_tokenizer=smtokenizer,
            split='train_val_256',
            replace_desc=False,
            corrupt_prob=0.,
            mask_desc=False
            # pre = pre
        )
        dataloader = get_dataloader(train_dataset,args.batch_size,rank,world_size)
    print('In total',len(train_dataset),'for training....')
    data_valid = None
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        reverse = reverse # Command to run revrse process
    ).run_loop()
    dist.destroy_process_group()


def create_argparser(reverse):
    defaults = dict()

    if reverse:
        model_in_channels = 768
    else:
        model_in_channels = 32

    text_defaults = dict(
        # attention_resolutions='16,8', 
        batch_size=64, 
        cache_mode='no', 
        checkpoint_path='../../checkpoints', 
        class_cond=False, 
        commonGen_train='diffusion_lm/common-gen/commongen_data', 
        config='ll', 
        config_name='bert-base-uncased', 
        data_dir='', 
        dataset_config_name='wikitext-2-raw-v1', 
        dataset_name='wikitext', 
        diffusion_steps=2000, 
        dropout=0.1, 
        # e2e_train='', 
        ema_rate='0.9999', 
        emb_scale_factor=1.0, 
        eval_interval=2000, 
        experiment='random', 
        experiment_mode='lm', 
        fp16_scale_growth=0.001, 
        gradient_clipping=2.4, 
        # image_size=8, 
        # in_channel=16, 
        # learn_sigma=False, 
        log_interval=20, 
        logits_mode=1, 
        lr = 0.00005,
        # lr=0.0001, # Lower the learing rate in 2024/5/5 for better convergence 
        lr_anneal_steps=200000, 
        microbatch=-1, 
        # modality='e2e-tgt', 
        model_arch='transformer', 
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None', 
        # noise_level=0.0, 
        noise_schedule='sqrt', 
        # num_channels=128, 
        # num_heads=4, 
        # num_heads_upsample=-1, 
        # num_res_blocks=2, 
        # out_channel=16, 
        # padding_mode='pad', 
        # predict_xstart=True, 
        # preprocessing_num_workers=1, 
        # rescale_learned_sigmas=True, 
        rescale_timesteps=True, 
        resume_checkpoint='', 
        roc_train='diffusion_lm/ROCstory', 
        save_interval=20000, 
        schedule_sampler='uniform', 
        seed=19991009, 
        sigma_small=False, 
        timestep_respacing='', 
        training_mode='e2e', 
        # use_bert_tokenizer='no', 
        use_checkpoint=False, 
        use_fp16=False, 
        # use_kl=False, 
        # use_scale_shift_norm=True, 
        weight_decay=0.0,
        model_in_channels = model_in_channels,
        model_model_channels = 128,
        model_dropout = 0.1,
        model_hidden_size = 768,
        model_num_attention_heads = 12,
        model_num_hidden_layers = 8
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    # os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    world_size=1
    reverse = True
    mp.spawn(main_worker,args=(world_size, reverse),nprocs=world_size,join=True)
