import os
import numpy
import matplotlib.pyplot as plt

# file_path = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/PUBCHEMSTM_parallel_batched.txt"

# strings=[]
# str_lens=[]
# Define your search phrase and file paths
search_phrase = "data available"
input_file = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/PUBCHEMSTM_parallel_batched.txt"
output_file = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/PUBCHEMSTM_filtered_1.txt"

dic = {}
f = open(output_file, 'r')
for line in f:
    parts = line.strip().split('\t')
    if parts[2] in dic.keys():
        dic[parts[2]].append(parts[0])
    else:
        dic[parts[2]] = [parts[0]]

l = set()
unique_count = 0
for key in dic.keys():
    l.add(len(dic[key]))
    if len(dic[key]) < 5:
        unique_count += 1

print(unique_count)
print(l)


# # Open the input file and output file
# with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#     for line in infile:
#         if search_phrase not in line:
#             outfile.write(line)


# with open(file_path, 'r') as file:
#     for line in file:
#         parts = line.strip().split('\t')
#         if len(parts) >= 2:
#             strings.append(parts[2])
#             str_lens.append(len(parts[2]))

# count_useless = 0
# useless_ind=[]

# for ind, i in enumerate(strings):
#     if ("data available" in i.lower()) :
#         # if str_lens[ind]<80:
#             count_useless+=1
#             useless_ind.append(ind)

# print("total descriptions analyzed: ",len(strings))
# print("out of them, this many are useless: ", count_useless)

# count_usefull = 0
# l = set()
# check=True
# for i in useless_ind:
#      l.add(str_lens[i])
#      if str_lens[i]>150:
#         count_usefull+=1
#         if check: 
#             print(strings[i])
#             check = False

# # print(sorted(list(l)))
# print(count_usefull)


