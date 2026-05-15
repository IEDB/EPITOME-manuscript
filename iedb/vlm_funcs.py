from typing import List, Dict
import re
from openai import OpenAI, OpenAIError, APITimeoutError
from httpx import TimeoutException
from helper_funcs import normalize_caption, encode_image, spot_mhc, clean_hla_match,  evaluate_expression, align_objects, pad_image_to_64
from PIL import Image
import math
import pytesseract
import yaml
from collections.abc import Iterable
from typing import Dict, List, Tuple
import json
from collections import defaultdict
# from pepmatch_toolcall import search_peptide_matches
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

prompt = """You are an intelligent AI assistant that is an expert in understanding and answering questions 
            about immunology. As an immunology expert, you are given snippets from real immunology papers, 
            and you are tasked to extract specific information from the text and images provided as context."""

with open('formatted_queries.yaml', 'r') as f:
    formatted_queries = yaml.safe_load(f)

def extract_text(image_path: str, extraction_func = pytesseract.image_to_string, other_args = {}):
    """
    basic image OCR
    
    :param image_path: path to image
    :type image_path: str
    :param extraction_func: function which accepts image path and any other arguments
    :param other_args: dictionary containing other arguments to pass to extraction function
    :return: text in image
    :rtype: str
    """
    image = Image.open(image_path)
    text = extraction_func(image, **other_args)
    return text

def identify_peptides(image_path: str, qwen2vl_client: OpenAI, prompt: str = prompt, 
    query: str = formatted_queries['peptide_query']['instruction'], model: str="Qwen/Qwen2.5-VL-32B-Instruct"):
    """
    peptide identification from image using VLM
    
    :param image_path: path to image
    :type image_path: str
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param prompt: system prompt
    :type prompt: str
    :param query: user prompt
    :type query: str
    :param model: model name
    :type model: str
    :return: VLM output
    :rtype: str
    """
    peptides = query_vlm(image_path, prompt, query, qwen2vl_client, model)
    return peptides

def identify_HLA(image_path: str, qwen2vl_client: OpenAI, prompt: str = prompt,
    query = formatted_queries['hla_query']['instruction'], model: str="Qwen/Qwen2.5-VL-32B-Instruct"): 
    """
    MHC molecule name from image using VLM
    :param image_path: path to image
    :type image_path: str
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param prompt: system prompt
    :type prompt: str
    :param query: user prompt
    :type query: str
    :param model: model name
    :type model: str
    :return: VLM output
    :rtype: str
    """
    hla = query_vlm(image_path, prompt, query, qwen2vl_client, model)
    return hla

def query_vlm(image_path: str, prompt: str, query: str, qwen2vl_client: OpenAI, model: str="Qwen/Qwen2.5-VL-32B-Instruct", pad_64: bool=False):

    """
    Directly queries VLM with a single image.
    
    :param image_path: path to image
    :type image_path: str
    :param prompt: system prompt
    :type prompt: str
    :param query: user prompt
    :type query: str
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param model: model name
    :type model: str
    :param pad_64: whether to pad input image to multiple of 64
    :type pad_64: bool
    :return: vlm response
    :rtype: Any
    """

    # Construct query for image
    if image_path:
        if pad_64:
            image_path = pad_image_to_64(image_path)
        img_b64_str = encode_image(image_path)
        content = [
            {
                'type': 'text',
                'text': query
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}
            }

        ]
        input = [{"role": "system", "content": prompt},
                {'role': 'user', 
                'content': content
        }]
        try:
            response = qwen2vl_client.chat.completions.create(model=model, messages=input)
            response = response.choices[0].message.content
        except TimeoutException:
            return f"None"
        except OpenAIError as e:
            return f"None"
        except Exception as e:
            return f"None"
     
    # Construct a text only query
    else:
        content = [
            {
                'type': 'text',
                'text': query
            }
        ]
        input = [{"role": "system", "content": prompt},
                {'role': 'user', 
                'content': content
        }]

        try:
            response = qwen2vl_client.chat.completions.create(model=model, messages=input)
            response = response.choices[0].message.content
        except TimeoutException:
            return f"None"
        except OpenAIError as e:
            return f"None"
        except Exception as e:
            return f"None"
    return response

def create_multimodal_context_single(image_path: str, caption: str):
    """
    Creating multimodal content given a single image path.
    
    :param image_path: Path to image
    :type image_path: str
    :param caption: Image caption
    :type caption: str
    :return: Description
    :rtype: Any
    """
    context = []
    img_b64_str = encode_image(image_path)
    context.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}})
    formatted_caption = f'The above image has caption: {caption}'
    context.append({'type': 'text', 'text': formatted_caption})
    return context

def validate_in_list(value: str, allowed_values: List):
    """
    validation function to enforce value.
    
    :param value: VLM output value
    :type value: str
    :param allowed_values: list of permissible values
    :type allowed_values: List
    :return: returns self if validated else raises an error
    :rtype: Any
    """
    if value not in allowed_values:
        raise ValueError(f"Value {value} not in allowed values: {allowed_values}")
    return value

def identify_peptides_from_context(visual_index: Dict, qwen2vl_client: OpenAI, prompt: str =prompt,
    query: str =formatted_queries['peptide_query']['instruction'], model: str="Qwen/Qwen2.5-VL-32B-Instruct") -> Dict:

    """
    Function that accepts context file (output from Docling) and returns a dictionary with keys=putative peptide, and
    values corresponding to images and text in which this peptide appears.
    
    :param visual_index: dictionary output containing context extracted from paper
    :type visual_index: dict
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param prompt: system prompt
    :type prompt: str
    :param query: user prompt
    :type query: str
    :param model: model name to use
    :type model: str
    """

    epitope_pattern = re.compile(r"[A-Z]{7,}")
    stopwords = {'MATERIALS', 'METHODS', 'SUPPLEMENTARY', 'RESEARCH', 'SYFPEITHI', 'RESULTS', 'DISCUSSION', 'INTRODUCTION', 'REFERENCES', 'SCALEPACK', 'ARTICLE', 'ACCESS', 'REFMAC', 'REFMACS', 'PEPTIDE', 'ANTIGEN', 'TARGETS', 'TREATMENTS',
                 'STATEMENT', 'MATERIAL', 'SCIENTIFIC'}
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    sequence_data = {}
    for header_type, text in zip(visual_index['Text section Headers'], visual_index['Text sections']):
        matches = epitope_pattern.findall(text)
        for match in matches:
            if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match):
                valid_peptide = match
                sequence_data.setdefault(valid_peptide, {"Texts": [], "Tables": [], "Images": []})
                sequence_data[valid_peptide]["Texts"].append(text)


    # --- TABLE SEARCH + SAVE --- (search table elements for peptide sequences)
    for table in visual_index['Tables']:
        matched_sequences = set()
        table_image_path = table['image_path']
        captions = table['captions']
        footnotes = table['footnotes']

        for text in table['text']:
            matches = epitope_pattern.findall(text)
            matches = [match for match in matches if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match)]
            matched_sequences.update(matches)

        for match in matched_sequences:
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Tables"].append({
                "image_path": str(table_image_path),
                "captions": captions,
                "footnotes": footnotes,
                'mode':'caption_text'
            })

        matched_sequences = set()
        peptides_vlm = query_vlm(table['image_path'], prompt, query, qwen2vl_client, model)
        if peptides_vlm != 'None':
            peptides_vlm = evaluate_expression(peptides_vlm)
            if peptides_vlm is not None:
                matched_sequences.update(peptides_vlm)

        for match in matched_sequences:
            if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match):
                sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
                sequence_data[match]["Tables"].append({
                    "image_path": str(table_image_path),
                    "captions": captions,
                    "footnotes": footnotes,
                    "mode": 'VLM'
                })
                
    # --- FIGURE SEARCH + SAVE --- (search figure captions for peptide sequences)
    for fig_item in visual_index['Images']:

        captions = fig_item['captions']
        footnotes = fig_item['footnotes']
        image_path = fig_item['image_path']
        combined_text = " ".join(captions + footnotes)
        matches = epitope_pattern.findall(combined_text)
        valid_matches = []
        for match in matches:
            if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match):
                valid_matches.append(match)

        for match in valid_matches:
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Images"].append({
                "image_path": str(image_path),
                "captions": captions,
                "footnotes": footnotes,
                "mode": 'caption_text'
            })

        valid_matches = set()
        peptides_vlm = query_vlm(image_path, prompt, query, qwen2vl_client, model)

        if peptides_vlm != 'None':
            peptides_vlm = evaluate_expression(peptides_vlm)
            if peptides_vlm is not None:
                for peptide in peptides_vlm:
                    valid_matches.add(peptide)
                    
        for match in valid_matches:
            if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match):            
                sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
                sequence_data[match]["Images"].append({
                    "image_path": str(image_path),
                    "captions": captions,
                    "footnotes": footnotes,
                    "mode": 'VLM'
                })
        
        text = extract_text(image_path)
        matches = epitope_pattern.findall(text)
        for match in matches:
            # Clean non-peptide related regex matches
            if match not in stopwords and all(c in valid_amino_acids for c in match) and not all(c in "GATC" for c in match):
                sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
                sequence_data[match]["Images"].append({
                    "image_path": str(image_path),
                    "captions": captions,
                    "footnotes": footnotes,
                    "mode": 'OCR'
                })

    stitched_by_caption = visual_index['Stitched image mapping']
    for seq in sequence_data.values():
        new_images = []
        for img in seq["Images"]:
            caption_key = normalize_caption(" ".join(img.get("captions", [])))
            stitched_entry = stitched_by_caption.get(caption_key)
            if stitched_entry:
                path = stitched_entry["image_path"]
                new_images.append({
                    "image_path": path,
                    "captions": stitched_entry.get("captions", []),
                    "footnotes": stitched_entry.get("footnotes", []),
                    "mode": img['mode']
                })
            else:
                new_images.append(img)
                seen_paths.add(img["image_path"])
        seq["Images"] = new_images
    return sequence_data

def identify_hla_from_context(visual_index: Dict, qwen2vl_client: OpenAI, prompt: str = prompt,
    hla_query: str = formatted_queries['hla_query']['instruction'], model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    filter: bool = False):

    """
    Function that accepts context file (output from Docling) and returns a dictionary with keys=putative MHC molecules, and
    values corresponding to images and text in which this MHC molecule appears.
    
    :param visual_index: dictionary output containing context extracted from paper
    :type visual_index: dict
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param prompt: system prompt
    :type prompt: str
    :param query: user prompt
    :type query: str
    :param model: model name to use
    :type model: str
    """
        
    sequence_data = {}
    for header_type, text in zip(visual_index['Text section Headers'], visual_index['Text sections']):
        if header_type == 'References':
            continue
        hla_matches = spot_mhc(text)
        for match in hla_matches:
            if clean_hla_match(match, filter=filter):
                valid_hla = clean_hla_match(match)
                sequence_data.setdefault(valid_hla, {"Texts": [], "Tables": [], "Images": []})
                sequence_data[valid_hla]["Texts"].append(text)

    # --- TABLE SEARCH + SAVE --- (search table elements for peptide sequences)
    for table in visual_index['Tables']:
        matched_sequences = set()
        table_image_path = table['image_path']
        captions = table['captions']
        footnotes = table['footnotes']

        for text in table['text']:
            matches = spot_mhc(text)
            matches = set([clean_hla_match(match, filter=filter) for match in matches])
            matched_sequences.update(matches)
        for match in matched_sequences:
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Tables"].append({
                "image_path": str(table_image_path),
                "captions": captions,
                "footnotes": footnotes,
                "mode": "caption text"
            })

        hla = query_vlm(table['image_path'], prompt, hla_query, qwen2vl_client, model)
        if hla != 'None':
            hla = evaluate_expression(hla)
            if hla is not None:
                matched_sequences.update(hla)
            
        for match in matched_sequences:
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Tables"].append({
                "image_path": str(table_image_path),
                "captions": captions,
                "footnotes": footnotes,
                "mode": "VLM"
            })
            
    # --- FIGURE SEARCH + SAVE --- (search figure captions for peptide sequences)
    for fig_item in visual_index['Images']:

        captions = fig_item['captions']
        footnotes = fig_item['footnotes']
        image_path = fig_item['image_path']
        combined_text = " ".join(captions + footnotes) + ' '

        matches = spot_mhc(combined_text)
        valid_matches = []
        for match in matches:
            if clean_hla_match(match, filter=filter):
                valid_matches.append(clean_hla_match(match))

        for match in valid_matches:            
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Images"].append({
                "image_path": str(image_path),
                "captions": captions,
                "footnotes": footnotes,
                "mode": 'caption text'
            })


        valid_matches = []
        hlas = query_vlm(image_path, prompt, hla_query, qwen2vl_client, model)


        # Don't add 'None' responses from VLM (aka, peptide not found in the figure)
        if hlas != 'None':
            hlas = evaluate_expression(hlas)
            if hlas is not None:
                for hla in hlas:
                    valid_matches.append(clean_hla_match(hla))

        for match in valid_matches:            
            sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
            sequence_data[match]["Images"].append({
                "image_path": str(image_path),
                "captions": captions,
                "footnotes": footnotes,
                "mode": 'VLM'
            })

    text = extract_text(image_path)
    matches = spot_mhc(text)
    for match in matches:
        match = clean_hla_match(match, filter=filter)
        sequence_data.setdefault(match, {"Texts": [], "Tables": [], "Images": []})
        sequence_data[match]["Images"].append({
            "image_path": str(image_path),
            "captions": captions,
            "footnotes": footnotes,
            "mode": 'OCR'
        })

    # Update sequence_data to point to stitched image paths
    stitched_by_caption = visual_index['Stitched image mapping']
    for seq in sequence_data.values():
        new_images = []
        seen_paths = set()
        for img in seq["Images"]:
            caption_key = normalize_caption(" ".join(img.get("captions", [])))
            stitched_entry = stitched_by_caption.get(caption_key)
            if stitched_entry:
                path = stitched_entry["image_path"]
                if path not in seen_paths:
                    new_images.append({
                        "image_path": path,
                        "captions": stitched_entry.get("captions", []),
                        "footnotes": stitched_entry.get("footnotes", []),
                        "mode": img['mode']
                    })
                    seen_paths.add(path)
            else:
                # If no stitched version, keep original
                if img["image_path"] not in seen_paths:
                    new_images.append(img)
                    seen_paths.add(img["image_path"])
        seq["Images"] = new_images
    return sequence_data


def get_shortnames(sequence_data: Dict, qwen2vl_client: OpenAI, prompt: str = prompt, model_name: str = './qwen2.5-vl-32b') -> Dict:

    """
    retrieve short names for each peptide listed in sequence data
    
    :param sequence_data: dictionary keyed on peptide with values=peptide mentions.
    :type sequence_data: Dict
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param prompt: system-level prompt
    :type prompt: str
    :param model_name: model to use
    :type model_name: str
    """

    for peptide in sequence_data:
        shortnames = []
        texts = sequence_data[peptide]['Texts']
        if texts:
            full_text = ''
            for text in texts:
                full_text += text + ' '
            query = f"""Given the text excerpt from an immunology paper: '{full_text}'', please find and return the name
            for the corresponding epitope linear sequence: {peptide}, where the author introduces the name alongside the 
            full peptide sequence and context (like species, amino acid positions, or protein sources). If there is no 
            name found, just generate 'None'. If a name is found, please include the entire name with 
            subcharacters and prefixes. Please only generate one word in your response, either the shortname if found, 
            otherwise None."""
            shortname = query_vlm(None, prompt, query, qwen2vl_client, model_name)
            shortnames.append(shortname)

        # Check if peptide is found in a table
        tables = sequence_data[peptide]['Tables']
        if tables:
            description = ''
            for table in tables:
                image = table['image_path']
                captions = table['captions']
                if captions:
                    for caption in captions:
                        description += caption + ' '
                footnotes = table['footnotes']
                if footnotes:
                    for footnote in footnotes:
                        description += footnote + ' '

                query = f"""Given this table with description '{description}' please find and return the shortname 
                for the corresponding epitope linear sequence: {peptide}. If there is no shortname found, just generate 
                'None'. If a shortname is found, please include the entire short name with subcharacters and prefixes. 
                Please only generate one word in your response."""
                shortname = query_vlm(image, prompt, query, qwen2vl_client, model_name)
                shortnames.append(shortname)

        # Check if peptide is found in figure caption
        figures = sequence_data[peptide]['Images']
        if figures:
            for fig in figures:
                description = ''
                image = fig['image_path']
                captions = fig['captions']
                if captions:
                    for caption in captions:
                        description += caption + ' '
                footnotes = fig['footnotes']
                if footnotes:
                    for footnote in footnotes:
                        description += footnote + ' '

                query = f"""Given this figure with description '{description}' please find and return the shortname 
                for the corresponding epitope linear sequence: {peptide}. If there is no shortname found, just generate 
                'None'. If a shortname is found, please include the entire short name with subcharacters and prefixes. 
                Please only generate one word in your response."""
                shortname = query_vlm(image, prompt, query, qwen2vl_client, model_name)
                shortnames.append(shortname)

        # Add shortnames to sequence_data
        # TO DO: Reconcile shortnames
        sequence_data[peptide]["Shortnames"] = shortnames

    return sequence_data
        

def create_prompt_peptide(peptide: str, shortnames: str, multimodal_context: Dict, full_text: str, formatted_query: str,
prompt: str = prompt):
    
    """
    Creating a peptide-specific context object
    
    :param peptide: epitope linear sequence of peptide
    :type peptide: str
    :param shortnames: shortname string
    :type shortnames: str
    :param multimodal_context: context object created by create_multimodal_context
    :type multimodal_context: Dict
    :param full_text: Full text extracted from paper
    :type full_text: str
    :param formatted_query: user-level prompt
    :type formatted_query: str
    :param prompt: system-level prompt
    :type prompt: str
    """

    content = []
    content.append({'type': 'text', 'text': f"""Given the peptide linear sequence {peptide}, and corresponding shortname(s) {shortnames} 
                    (shortnames are usually how the peptide is referred to in papers), the extracted text from the paper: 
                    '{full_text}', as well as the following extracted tables and figures: """})
    content.extend(multimodal_context)
    content.append({'type': 'text', 'text': formatted_query})
    input = [{"role": "system", "content": prompt},
            {'role': 'user',
            'content': content
    }]
    return input


def get_logprobs(response) -> float:
    """
    calculate np.exp(mean(logprob)) of response obect
    
    :param response: Description
    """

    logprobs = [entry.logprob for entry in response.choices[0].logprobs.content if entry.logprob is not None]
    if not logprobs:
        raise ValueError("No usable logprobs found.")
    mean_logprob = sum(logprobs) / len(logprobs)
    confidence = math.exp(mean_logprob)
    return confidence

def query_qwen(qwen2vl_client: OpenAI, input: List[Dict], model: str="Qwen/Qwen2.5-VL-32B-Instruct", max_retries: int=3,
parsing_func=lambda x:x, extra_args: dict = {}, parsing_func_args: dict = {}):
    """
    querying qwen
    
    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param input: input passed directly to model
    :type input: List[Dict]
    :param model: Description
    :type model: str
    :param max_retries: maximum number of times to requery until accepting timeout error/incorect output format
    :type max_retries: int
    :param parsing_func: function to parse/validate output response
    :param extra_args: additional arguments for chat completion
    :type extra_args: dict
    :param parsing_func_args: additional arguments for parsing function
    :type parsing_func_args: dict
    """
    for attempt in range(1, max_retries+1):
        try:
            response = qwen2vl_client.chat.completions.create(model=model, messages=input, **extra_args)
            response = response.choices[0].message.content
            try:
                return parsing_func(response, **parsing_func_args)
            except Exception as parse_err:
                print(f"[Warning] Parsing failed (attempt {attempt}/{max_retries}): {parse_err}")
                if attempt == max_retries:
                    return f'"[PARSE ERROR after {max_retries} attempts]: {parse_err}"'+':'+response
                # else continue to next retry
                continue
        except APITimeoutError as e:
            print(f"[Warning] Qwen query failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return '"[API ERROR]: Timeout"'
            
    # FINAL FALLBACK — prevents implicit None
    return '"[ERROR]: Unhandled exception or empty response"', None

def query_qwen_w_logprobs(qwen2vl_client: OpenAI, input: List[Dict], model="Qwen/Qwen2.5-VL-32B-Instruct", max_retries=3,
parsing_func=lambda x:x, extra_args: dict = {}, parsing_func_args: dict = {}) -> Tuple:

    """
    querying qwen and returning avrage log probability of response

    :param qwen2vl_client: openai client
    :type qwen2vl_client: OpenAI
    :param input: input passed directly to model
    :type input: List[Dict]
    :param model: Description
    :type model: str
    :param max_retries: maximum number of times to requery until accepting timeout error/incorect output format
    :type max_retries: int
    :param parsing_func: function to parse/validate output response
    :param extra_args: additional arguments for chat completion
    :type extra_args: dict
    :param parsing_func_args: additional arguments for parsing function
    :type parsing_func_args: dict
    """

    for attempt in range(1, max_retries+1):
        try:
            response = qwen2vl_client.chat.completions.create(model=model, messages=input, logprobs=True, **extra_args)
            mean_logprobs = get_logprobs(response)
            response = response.choices[0].message.content
            try:
                return parsing_func(response, **parsing_func_args), mean_logprobs
            except Exception as parse_err:
                print(f"[Warning] Parsing failed (attempt {attempt}/{max_retries}): {parse_err}")
                if attempt == max_retries:
                    return f'"[PARSE ERROR after {max_retries} attempts]: {parse_err}"'+':'+response, mean_logprobs
                # else continue to next retry
                continue
        except APITimeoutError as e:
            print(f"[Warning] Qwen query failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return '"[API ERROR]: Timeout"', None
    
    # FINAL FALLBACK — prevents implicit None
    return '"[ERROR]: Unhandled exception or empty response"', None



def validate_response(json_string: str, expected_fields: List) -> dict:
    """
    validate_response: validation function to confirm that the correct fields are present
    
    :param json_string: VLM output
    :type json_string: str
    :param expected_fields: list of fields that we want the VLM to output
    :type expected_fields: List
    :return: output dictionary
    :rtype: dict
    """
    json_dictionary = json.loads(json_string)
    if isinstance(json_dictionary, Dict):
        json_dictionary = [json_dictionary]
    if not isinstance(json_dictionary, Iterable):
        raise Exception(f'Expecting a list.')
    for value in json_dictionary:
        missing_keys = set(expected_fields).difference(set(value.keys()))
        if len(missing_keys) != 0:
            keys = ', '.join(missing_keys)
            raise Exception(f'Missing {keys} in output.')
    return json_dictionary

def create_multimodal_context(peptide_sources: Dict, pad_to_64: bool = True):

    """
    create a multimodal context object for VLM query
    
    :param peptide_sources: value corresponding to a particular peptide (key) from the peptide ojbect,
    :type peptide_sources: Dict
    :param pad_to_64: whether to pad images to maximum batch size with %64 = 0.
    :type pad_to_64: bool
    """

    context = []
    tables = peptide_sources.get('Tables')
    seen_paths = set()
    for table in tables:
        image_path = table['image_path']
        if image_path in seen_paths:
            continue
        image_path  = pad_image_to_64(image_path) if pad_to_64 else image_path
        print(f"table added to images: {image_path}")
        img_b64_str = encode_image(image_path)
        caption = " ".join(table['captions']) + " ".join(table['footnotes'])

        # Add the table image to query
        context.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}})

        # Add the table caption to query
        formatted_caption = f'The above image has caption: {caption}'
        context.append({'type': 'text', 'text': formatted_caption})
        seen_paths.add(image_path)

    # Add figures and their captions to the query
    figures = peptide_sources.get('Images')
    for figure in figures:
        image_path = figure['image_path']
        if image_path in seen_paths:
            continue
        image_path  = pad_image_to_64(image_path) if pad_to_64 else image_path
        print(f"figure added to images: {image_path}")
        img_b64_str = encode_image(image_path)
        caption = figure['captions']

        # Add the table image to query
        context.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}})

        # Add the table caption to query
        formatted_caption = f'The above image has caption: {caption}'
        context.append({'type': 'text', 'text': formatted_caption})
        seen_paths.add(image_path)
    return context


def create_agent(port: int, model_name: str = 'Qwen/Qwen2.5-VL-32B-Instruct'):
    llm = ChatOpenAI(
        openai_api_key="token-abc123",
        openai_api_base=f"http://localhost:{port}/v1",
        model_name=model_name,
        temperature=0.1,
        max_tokens=256
    )
    # === Create React Agent with both tools ===
    agent = create_react_agent(
        llm,
        tools=[search_peptide_matches]
    )
    return agent


def get_source_antigen_id( peptide, proteome, agent):

    if proteome != 'None':
        print(f"Searching pepmatch with proteome: {proteome}")
        history_peptide = [
            SystemMessage(content="You are a bioinformatics assistant that helps with peptide analysis and protein identification."),
            HumanMessage(content=f"Search for matches of this peptide sequence in the {proteome} proteome: {peptide}? I want to find which protein it comes from. Return only the accession as a string and nothing else. If there are no accessions, return None. '''Example output'''\n 'P14679.3'")
        ]
        print("PEPMATCH SEARCH:")
        result_peptide = agent.invoke({"messages": history_peptide})
        source_antigen_id = result_peptide['messages'][4].content
        print("\n💬 FINAL MODEL RESPONSE:\n", source_antigen_id)
        return source_antigen_id
    
    # Explicit fallback return to avoid implicit None
    return None

