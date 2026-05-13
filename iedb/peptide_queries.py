from openai import OpenAI
from vlm_funcs import get_shortnames, create_prompt_peptide, query_qwen_w_logprobs, create_multimodal_context
import os
from glob import glob
from argparse import ArgumentParser
import pandas as pd
import pickle
from tqdm import tqdm
import yaml


args = ArgumentParser()
args.add_argument('--input_dir', type=str, required=True, help='Directory containing content.p')
args.add_argument('--suffix', type=str, default='')
args.add_argument('--model_name', type=str, default='./qwen2.5-vl-32b')
args.add_argument('--queries', type=str, default='formatted_queries.yaml')
args.add_argument('--port', type=int, default=8001)


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

print(f'Working on {input_path}.')

with open(parsed_args.queries, "r") as f:
    formatted_queries = yaml.safe_load(f)

true_peptides = pd.read_pickle(f'data/true_peptides.p')
for context_file in tqdm(glob(os.path.join(input_path, '*', 'content.p'))):
    if os.path.exists(context_file.replace('content.p', f'peptide_data{suffix}.p')) is False:
        continue
    if os.path.exists(context_file.replace('content.p', f'vlm_full_context{suffix}.p')) is True:
        continue
    sequence_data = pd.read_pickle(context_file.replace('content.p', f'peptide_data{suffix}.p'))
    pmid = int(context_file.split('/')[-2])
    if pmid not in true_peptides:
        continue
    visual_index = pd.read_pickle(context_file)
    sequence_data = {k: sequence_data[k] for k in sequence_data if k in true_peptides[pmid]}
    sequence_data = get_shortnames(sequence_data, qwen2vl_client, model_name)
    output = {}
    for peptide in sequence_data:
        shortnames = sequence_data[peptide]['Shortnames']
        context = create_multimodal_context(sequence_data[peptide], pad_to_64=False)
        output[peptide] = {'Shortnames': shortnames}
        for query in ['object_type', 'mhc_allele_name', 'organism_id']:
            try:
                prompt_input = create_prompt_peptide(peptide, shortnames, context, visual_index['Full text'], formatted_queries[query]['instruction'])
                out, logprobs = query_qwen_w_logprobs( qwen2vl_client, prompt_input, model_name)   
                output[peptide][query] = {'content': out, 'logprobs': logprobs}
            except Exception as e:
                output[peptide][query] = {'content': e, 'logprobs': None}

    with open(context_file.replace('content.p', f'vlm_full_context{suffix}.p'), 'wb') as f:
        pickle.dump(output, f)
