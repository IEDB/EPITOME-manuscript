from helper_funcs import extract_elements_basic
from joblib import Parallel, delayed
from glob import glob
from argparse import ArgumentParser
import os

args = ArgumentParser()
args.add_argument('--input_dir', type=str, required=True, help='Directory containing input PDFs')
args.add_argument('--output_dir', type=str, help='Directory to save extracted elements', default='Extracted_Elements')
args.add_argument('--jobs', type=int)
parsed_args = args.parse_args() 

output_path = parsed_args.output_dir
input_path = parsed_args.input_dir
files = glob(os.path.join(input_path, '*.pdf'))
output = Parallel(n_jobs=parsed_args.jobs)(delayed(extract_elements_basic)(f, output_path) for f in files)