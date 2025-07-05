import os
import re
import pubchempy as pcp
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Manager


def fetch_smiles_batch(cid_list, max_retries=3):
    cid_to_smiles = {}
    for attempt in range(max_retries):
        try:
            compounds = pcp.get_compounds(cid_list, 'cid')
            for compound in compounds:
                cid_to_smiles[str(compound.cid)] = compound.isomeric_smiles or "NA"
            break  # success
        except Exception as e:
            print(f"[Retry {attempt+1}] Failed to fetch batch: {cid_list[:5]}... → {e}")
            time.sleep(1)

    # Fill in NA for any failed CIDs
    for cid in cid_list:
        cid_str = str(cid)
        if cid_str not in cid_to_smiles:
            cid_to_smiles[cid_str] = "NA"

    return cid_to_smiles


def process_single_file(file_path, target_file, lock):
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    i = 0
    batch = []
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            cid = line
            i += 1
            if i < len(lines):
                raw_description = lines[i].strip()
                description = re.sub(r"\s*\([A-Z]+,\s*\d{4}\)\s*$", "", raw_description)
                batch.append((cid, description))
        i += 1

        # Process every 5 entries
        if len(batch) == 10:
            write_batch_to_file(batch, target_file, lock)
            batch = []

    # Process remaining batch
    if batch:
        write_batch_to_file(batch, target_file, lock)


def write_batch_to_file(batch, target_file, lock):
    cids = [cid for cid, _ in batch]
    cid_to_smiles = fetch_smiles_batch(cids)

    lines = []
    for cid, desc in batch:
        smiles = cid_to_smiles.get(cid, "NA")
        if 10 < len(smiles) < 257:
            lines.append(f"{cid}\t{smiles}\t{desc}")

    if lines:
        with lock:
            with open(target_file, 'a', encoding='utf-8') as out:
                out.write('\n'.join(lines) + '\n')


def process_all_files(source_files, target_file_path, num_workers=4):
    if os.path.exists(target_file_path):
        os.remove(target_file_path)  # Clean previous output
    print('Processing Starting')
    with Manager() as manager:
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for path in source_files:
                futures.append(executor.submit(process_single_file, path, target_file_path, lock))

            for f in futures:
                f.result()  # wait for all to complete

# ==== CONFIG ====
text_files_dir = "/home/aditya/Summer_Intern_2025/dataset/step_01_PubChemSTM_description"
source_files = [os.path.join(text_files_dir, i) for i in sorted(os.listdir(text_files_dir))]

target_file = "/home/aditya/Summer_Intern_2025/tgm-dlm/datasets/SMILES/PUBCHEMSTM_parallel_batched.txt"

# ==== RUN ====
process_all_files(source_files, target_file, num_workers=4)