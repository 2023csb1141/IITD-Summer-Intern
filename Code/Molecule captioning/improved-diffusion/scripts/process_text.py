from mydatasets import get_dataloader,ChEBIdataset, ChEBIdataset_rev
import torch
import transformers
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse

rev = True # Turn it to false for Text2Mol

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True)
args = parser.parse_args()
split = args.input
smtokenizer = regexTokenizer()
if rev:
    train_dataset = ChEBIdataset_rev(
            dir='../../datasets/SMILES/',
            smi_tokenizer=smtokenizer,
            split=split,
            replace_desc=False,
            load_state=False
            # pre = pre
        )
else: 
    print('Reverse Train Dataset Called')
    train_dataset = ChEBIdataset_rev(
            dir='../../datasets/SMILES/',
            smi_tokenizer=smtokenizer,
            split=split,
            replace_desc=False,
            load_state=False
            # pre = pre
        )
print(f'Calling Tokenizer')
tokz = AutoTokenizer.from_pretrained('../../scibert')

volume = {}

if rev:
    with torch.no_grad():
        for i in range(len(train_dataset)):
            if i%190 == 0:
                print(i)
            id = train_dataset[i]['cid']
            desc = train_dataset[i]['desc']
            tok_op = tokz(
                desc,max_length=216, truncation=True,padding='max_length'
                )
            toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
            # toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
            assert(toked_desc.shape[1]==216)
            # lh = model(toked_desc.cuda()).last_hidden_state
            volume[id] = {'states':toked_desc.to('cpu')}
else:
    model = AutoModel.from_pretrained('../../scibert')
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for i in range(len(train_dataset)):
            if i%190 == 0:
                print(i)
            id = train_dataset[i]['cid']
            desc = train_dataset[i]['desc']
            tok_op = tokz(
                desc,max_length=216, truncation=True,padding='max_length'
                )
            toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
            toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
            assert(toked_desc.shape[1]==216)
            lh = model(toked_desc.cuda()).last_hidden_state
            volume[id] = {'states':lh.to('cpu'),'mask':toked_desc_attentionmask}



torch.save(volume,'../../datasets/SMILES/'+split+'_desc_states.pt')
