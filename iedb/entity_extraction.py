from vlm_funcs import identify_peptides_from_context, identify_hla_from_context, identify_assays_from_context, get_shortnames
from glob import glob
from argparse import ArgumentParser
import os
import pickle
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

args = ArgumentParser()
args.add_argument('--input_dir', type=str, required=True, help='Directory containing content.p')
args.add_argument('--suffix', type=str, default='')
args.add_argument('--model_name', type=str, default='./qwen2.5-vl-32b', help='Path to the VLM model')
args.add_argument('--port', type=int, default=8001, help='Port for the VLM model')

parsed_args = args.parse_args() 
input_path = parsed_args.input_dir
suffix = parsed_args.suffix
model_name = parsed_args.model_name
port = parsed_args.port

# Create client for qwen2vl inf
qwen2vl_client = OpenAI(
    base_url=f'http://localhost:{port}/v1',
    api_key='token-abc123',
    max_retries=0,
    timeout=60
)

for context_file in tqdm(glob(os.path.join(input_path, '*', 'content.p'))):
    visual_index = pd.read_pickle(context_file)
    if os.path.exists(context_file.replace('content.p', f'peptide_data{suffix}.p')) is False:
        output = identify_peptides_from_context(visual_index, qwen2vl_client, model = model_name)
        # output = get_shortnames(output,  qwen2vl_client, model_name = model_name)
        with open(context_file.replace('content.p', f'peptide_data{suffix}.p'), 'wb') as f:
            pickle.dump(output, f)
    if os.path.exists(context_file.replace('content.p', f'hla_data{suffix}.p')) is False:

        output = identify_hla_from_context(visual_index, qwen2vl_client, model= model_name)
        with open(context_file.replace('content.p', f'hla_data{suffix}.p'), 'wb') as f:
            pickle.dump(output, f)

    # output = identify_assays_from_context(visual_index, qwen2vl_client, model=model_name)
    # with open(context_file.replace('content.p', f'assay_data{suffix}.p'), 'wb') as f:
    #     pickle.dump(output, f)
