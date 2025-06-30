# import re

# def extract_smiles_vocab(smiles_list):
#     vocab = set()

#     # Regex to extract bracket tokens like [NH3+], [C@@H], [Cu+2]
#     bracket_pattern = re.compile(r'\[.*?]')

#     # Iterate over each SMILES string
#     for smi in smiles_list:
#         i = 0
#         while i < len(smi):
#             if smi[i] == '[':
#                 # Extract the full bracket token
#                 match = bracket_pattern.match(smi, i)
#                 if match:
#                     vocab.add(match.group())
#                     i = match.end()
#                 else:
#                     i += 1  # malformed bracket, skip
#             else:
#                 # Non-bracket characters: capture meaningful SMILES symbols
#                 if smi[i] in ('=', '#', '-', '/', '\\', '.', '(', ')'):
#                     vocab.add(smi[i])
#                 elif smi[i].isdigit():
#                     vocab.add(smi[i])
#                 elif smi[i].isalpha():
#                     # Handle multi-letter atoms like Cl, Br
#                     if i + 1 < len(smi) and smi[i:i+2] in ['Cl', 'Br']:
#                         vocab.add(smi[i:i+2])
#                         i += 2
#                         continue
#                     else:
#                         vocab.add(smi[i])
#                 i += 1

#     return sorted(vocab)

# smiles_data = []
# file_path = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/train_val_256.txt"

# with open(file_path, 'r') as file:
#     file.readline()
#     for line in file:
#         parts = line.strip().split('\t')
#         if len(parts) >= 2:
#             smiles_data.append(parts[1])
#             # print(parts[1])
#             # break

# vocab = extract_smiles_vocab(smiles_data)

# print(len(vocab))

# output_file = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/newly_generated_vocab.txt"
# with open(output_file, 'w') as f:
#     for token in vocab:
#         f.write(f"{token}\n")

    

import re

def extract_smiles_vocab(smiles_list):
    vocab = set()

    bracket_pattern = re.compile(r'\[.*?]')

    for smi in smiles_list:
        i = 0
        while i < len(smi):
            if smi[i] == '[':
                match = bracket_pattern.match(smi, i)
                if match:
                    vocab.add(match.group())
                    i = match.end()
                else:
                    i += 1  # malformed bracket
            elif smi[i] == '%':
                # Handle % followed by two digits (e.g., %10), or just a standalone %
                if i + 2 < len(smi) and smi[i+1].isdigit() and smi[i+2].isdigit():
                    vocab.add(smi[i:i+3])  # e.g., %10
                    i += 3
                else:
                    vocab.add('%')  # single '%' token
                    i += 1
            else:
                ch = smi[i]

                # Symbols and punctuation
                if ch in ('=', '#', '-', '/', '\\', '.', '(', ')'):
                    vocab.add(ch)

                # Digits (ring closures)
                elif ch.isdigit():
                    vocab.add(ch)

                # Atoms (including Cl, Br)
                elif ch.isalpha():
                    if i + 1 < len(smi) and smi[i:i+2] in ['Cl', 'Br']:
                        vocab.add(smi[i:i+2])
                        i += 2
                        continue
                    else:
                        vocab.add(ch)

                i += 1

    return sorted(vocab)

smiles_data = []
file_path = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/train_val_256.txt"

with open(file_path, 'r') as file:
    file.readline()  # skip header
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            smiles_data.append(parts[1])

vocab = extract_smiles_vocab(smiles_data)
print("Vocab size:", len(vocab))

output_file = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/newly_generated_vocab.txt"
with open(output_file, 'w') as f:
    for token in vocab:
        f.write(f"{token}\n")
