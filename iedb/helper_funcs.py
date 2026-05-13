import re
import math
import logging
import base64
import ast
from pathlib import Path
from PIL import Image
from collections import defaultdict
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
import ahocorasick
from docling.document_converter import DocumentConverter, PdfFormatOption
import pickle
from typing import List, Dict, Set, Tuple, Optional
from PIL import Image, ImageOps
import math


IMAGE_RESOLUTION_SCALE = 2.0


def dimensions(image_path: str) -> Tuple:
    """
    return image dimensions, used for resizing.
    :param image_path: path to image
    :type image_path: str
    :return: width and height
    :rtype: Tuple
    """
    image = Image.open(image_path)
    w, h = image.size
    return w, h

def resize_image_to_target(image_path: str, target_w: int, target_h: int) -> str:
    """
    resizes image to target dimensions by padding with a black border
    
    :param image_path: path to image
    :type image_path: str
    :param target_w: target width
    :type target_w: int
    :param target_h: target height
    :type target_h: int
    :return: path to resized image
    :rtype: str
    """
    image = Image.open(image_path)
    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    new_img.paste(image, (0, 0 ))
    new_img.save(image_path.replace('.png', '_resized.png'))
    return image_path.replace('.png', '_resized.png')

def is_valid_sequence(s: str, min_len: int = 7) -> bool:
    """
    returns bool filtering on non-alphabetical characters and length.    
    :param s: Description
    :type s: str
    :param min_len: Description
    :type min_len: int
    :return: Description
    :rtype: bool
    """
    return s.isupper() and s.isalpha() and s.isascii() and len(s) >= min_len

def evaluate_expression(text: str):
    """
    Wrapper for eval of VLM output string.
    Parse with ast.literal_eval but only accept string results.
    Returns None for malformed input or non-string literals.
    """
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError, TypeError):
        return None
    return value


def get_figure_number_from_caption(text) -> int | None:
    pattern = re.compile(r"figure\s+([ivxlcdm\d]+)", re.IGNORECASE)
    match = pattern.search(text.strip())
    if not match:
        return None
    fig_no_raw = match.group(1)
    try:
        return roman_to_int(fig_no_raw) if fig_no_raw.isalpha() else int(fig_no_raw)
    except ValueError:
        return None

def epitope_structure_defines(peptide: str, mhc_allele_peptide: str, mhc_alleles_paper: List, synonyms: Dict, allele_to_class: Dict):
    """
    logic as per curation manual to define Epitope structure defines field.
    
    :param peptide: linear peptide sequence e.g. GILGFVFTL
    :type peptide: str
    :param mhc_allele_peptide: peptide-associated MHC allele names, output from peptide-centric VLM query
    :type mhc_allele_peptide: str
    :param mhc_alleles_paper: MHC allele list extracted from full paper
    :type mhc_alleles_paper: List
    :param synonyms: MHC allele name synonym used for standardization of mhc_allele_peptide
    :type synonyms: Dict
    :param allele_to_class: dictionary mapping MHC molecule name to class (Class I or Class II)
    :type allele_to_class: Dict
    """
    alleles = evaluate_expression(mhc_allele_peptide)
    specific = True
    if alleles is None:
        alleles = mhc_alleles_paper
        specific = False
    alleles_corrected = [standardize_mhc_name(m, synonyms) for m in alleles]
    alleles_corrected = [p for p in alleles_corrected if p != None]
    if len(alleles_corrected) == 0:
        alleles = mhc_alleles_paper
        alleles_corrected = [standardize_mhc_name(m, synonyms) for m in alleles]
        alleles_corrected = [p for p in alleles_corrected if p != None]
        specific = False
    classes = set([allele_to_class[x] for x in alleles_corrected])
    if 'Class I' in classes:
        if len(peptide) >= 12:
            return 'Epitope Containing Region/Antigenic Site', classes, specific, alleles_corrected
        else:
            return 'Exact Epitope',  classes, specific, alleles_corrected
    else:
        if len(peptide) >= 16:
            return 'Epitope Containing Region/Antigenic Site',  classes, specific, alleles_corrected
        else:
            return 'Exact Epitope', classes, specific, alleles_corrected


def get_table_number_from_caption(text):
    # TO DO: Ensure 'Fig #' is also being caught
    #pattern = re.compile(r"^(?:fig(?:ure)?\.?\s*#?\s*|figure\s+)([ivxlcdm\d]+)", re.IGNORECASE)
    pattern = re.compile(r"table\s+([ivxlcdm\d]+)", re.IGNORECASE)
    match = pattern.search(text.strip())
    if not match:
        return None
    fig_no_raw = match.group(1)
    try:
        return roman_to_int(fig_no_raw) if fig_no_raw.isalpha() else int(fig_no_raw)
    except ValueError:
        return None



def parse_results(vlm_output: str, options: List) -> List:
    """
    Docstring for parse_results
    
    :param vlm_output: raw textual output from VLM
    :type vlm_output: str
    :param options: options that we provided to the VLM
    :type options: List
    :return: list of options that appeared in VLM output
    :rtype: List
    """
    pattern = re.compile('|'.join(options), re.IGNORECASE)
    return list(set(pattern.findall(vlm_output)))

def clean_hla_match(string: str, filter: bool = False, synonyms: dict = {}):
    """
    function to either filter MHC names or return self if filter set to false.
    
    :param string: putative MHC molecule name
    :type string: str
    :param filter: boolean instruction whether to filter for validity
    :type filter: bool
    :param synonyms: synonym dictionary where keys are valid synonyms, keys are valid vocabulary.
    :type synonyms: dict
    """
    if filter:
        string = standardize_mhc_name(string, synonyms=synonyms)
        if string is None:
            return False
        return string
    else:
        return string

def standardize(name: str):
    """
    function to perform standardization incl superscript on a putative mhc molecule name
    :param name: putative mhc molecule name
    :type name: str
    """
    superscript_map = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c', 'ᵈ': 'd', 'ᵉ': 'e',
    'ᶠ': 'f', 'ᵍ': 'g', 'ʰ': 'h', 'ⁱ': 'i', 'ʲ': 'j',
    'ᵏ': 'k', 'ˡ': 'l', 'ᵐ': 'm', 'ⁿ': 'n', 'ᵒ': 'o',
    'ᵖ': 'p', 'ʳ': 'r', 'ˢ': 's', 'ᵗ': 't', 'ᵘ': 'u',
    'ᵛ': 'v', 'ʷ': 'w', 'ˣ': 'x', 'ʸ': 'y', 'ᶻ': 'z'}
    return ''.join([superscript_map.get(char, char) for char in name.replace('-','').replace(' ', '').replace('^','').lower()])

def standardize_mhc_name(mhc_name: str, synonyms: dict):
    """
    Function to standardize MHC name to attempt to identify synonyms
    
    :param mhc_name: putative MHC name
    :type mhc_name: str
    :param synonyms: synonym dictionary where keys are valid synonyms, keys are valid vocabulary.
    :type synonyms: dict
    """
    if mhc_name in synonyms:
        return synonyms[mhc_name]
    mhc_name = standardize(mhc_name)
    synonyms_nodash = dict((standardize(p), synonyms[p]) for p in synonyms)
    if mhc_name in synonyms_nodash:
        return synonyms_nodash[mhc_name]
    for prefix in ['hla', 'h2']:
        if prefix + mhc_name in synonyms_nodash:
            return synonyms_nodash[prefix+mhc_name]
    return None

def spot_mhc_ahocorasick(text: str, automaton) -> Set:
    """
    uses automaton to identify vocabulary matches in text.
    
    :param text: text to search
    :type text: str
    :param automaton: automaton initialized with relevant vocabulary
    :return: set of words identified
    :rtype: Set
    """
    matches = set()
    for end_idx, (idx, word) in automaton.iter(text.lower()):
        matches.add(word)
    return matches

def create_ahocorasick(synonyms: dict):
    """
    Function to create automaton for Aho-Corasick function
    
    :param synonyms: dictionary containing synonyms where key=synonym, value=valid word.
    :type synonyms: dict
    """
    automaton = ahocorasick.Automaton()
    i = 0
    for word in synonyms:
        automaton.add_word(word.lower(), (i, word))
        i += 1
        automaton.add_word(word.strip('()^-').lower(), (i, word))
        i += 1
    automaton.make_automaton()
    return automaton

def stitch_images(image_paths, output_path, direction='vertical', target_multiple=112):
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    if direction == 'vertical':
        total_height = sum(heights)
        max_width = max(widths)
        stitched = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in images:
            stitched.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        total_width = sum(widths)
        max_height = max(heights)
        stitched = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            stitched.paste(img, (x_offset, 0))
            x_offset += img.width

    # Resize to nearest multiple of 112 (padding not stretching)
    w, h = stitched.size
    new_w = ((w + target_multiple - 1) // target_multiple) * target_multiple
    new_h = ((h + target_multiple - 1) // target_multiple) * target_multiple

    # Center-pad to new size
    stitched_padded = ImageOps.pad(stitched, (new_w, new_h), color=(255, 255, 255))

    stitched_padded.save(output_path)
    return str(output_path)


def roman_to_int(s):
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = prev = 0
    for char in reversed(s.upper()):
        val = roman[char]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def get_caption_number(caption, caption_type='figure'):
    if caption_type == 'table':
        pattern = re.compile(r"^\s*tables?\s+([ivxlcdm\d]+)\.?", re.IGNORECASE)
    else:
        pattern = re.compile(r"^(?:fig(?:ure)?\.?)\s*#?\s*([ivxlcdmIVXLCDM\d]+)\b", re.IGNORECASE)
    match = pattern.match(caption)
    if not match:
        return None
    no_raw = match.group().strip('.')
    return no_raw

def spot_mhc(text):
    """
    Regex pattern for identifying MHC molecule names
    
    :param text: text to search for MHC molecule names
    """
    pattern = re.compile(
        r"""
        (?<![A-Za-z0-9])(
            # ---- paired HLA class II (α/β) ----
            HLA-(
                DPA1|DQA1|DRA
            )\*\d+(?::\d+)*\s*/\s*
            (?:HLA-)?(
                DPB1|DQB1|DRB1|DRB3|DRB4|DRB5
            )\*\d+(?::\d+)*

            | # ---- canonical single HLA alleles ----
            (?:HLA[-\s]*)?
            (?:
                A|B|C|E|F|G|
                DPA1|DPB1|DQA1|DQB1|
                DRA|DRB1|DRB3|DRB4|DRB5
            )
            (?:\*\d+(?::\d+)*)?
            (?:\s+class\s+[IVX]+)?

            | # ---- compact & dotted HLA forms ----
            HLA[-\s]?(?:A|B|C)\d{4,6}
            | HLA[-\s]?(?:A|B|C)\d+\.\d+

            | # ---- HLA-Cw style ----
            (?:HLA[-\s]*)?Cw\d+(?:\.\d+)?

            | # ---- HLA serotype-style (DQ2, DR3, etc.) ----
            \(?HLA[-\s]?(?:DQ\d{0,2}|DR\d{0,2}|DP\d{0,2})\)?

            | # ---- murine alleles and haplotypes ----
            (?:^|[\s\(\[,;])                # space or punctuation before
            (?:
                H-?2[-/]?(?:K|D|L)[bdkqst]  # H-2Kb, H2-Kb, H-2Kd, etc.
              | H-?2[a-z]                   # H-2b, H-2d haplotypes
              | (?:K|D|L)[bdkqst]           # Kb, Kd, Ld, Dd
              | I[-/]?(?:A|E)[bdkstpq]?    # I-Ak, IA^b, IEk, etc.
              | I-A[bdkstpq]?              # explicit I-A / I-Ab / I-Ak
              | Qa-1[b]?                    # Qa-1 or Qa-1b
            )
            (?=[\s\)\],;:.!?])              # space or punctuation after
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    matches = [m.group(1).strip() for m in pattern.finditer(text)]
    return matches

def find_table_footnote(table_item, doc):
    """
    Finds the nearest text element below the table,
    prioritizing text that is directly below and closest vertically.
    """
    if not table_item.prov or not hasattr(table_item.prov[0], 'bbox'):
        return []
    tbl_bbox = table_item.prov[0].bbox
    tbl_page = table_item.prov[0].page_no
    tbl_center_x = (tbl_bbox.l + tbl_bbox.r) / 2
    tbl_bottom_y = tbl_bbox.b  # Bottom edge of table

    text_candidates = []

    for text_item in doc.texts:
        text = text_item.text.strip()
        if not text:  # Skip empty text
            continue

        if not text_item.prov or not hasattr(text_item.prov[0], 'bbox'):
            continue

        text_prov = text_item.prov[0]
        if text_prov.page_no != tbl_page:
            continue  # Filter only same-page text

        text_bbox = text_prov.bbox
        text_center_x = (text_bbox.l + text_bbox.r) / 2
        text_top_y = text_bbox.t  # Top edge of text

        # Only consider text that is below the table
        if text_top_y >= tbl_bottom_y:
            continue

        # Calculate vertical distance (how far below the table)
        vertical_distance = tbl_bottom_y - text_top_y
        
        # Calculate horizontal alignment (how well centered under table)
        horizontal_distance = abs(text_center_x - tbl_center_x)
        
        # Combined distance with preference for vertical proximity
        # Weight vertical distance more heavily than horizontal alignment
        combined_distance = vertical_distance + (horizontal_distance * 0.3)

        text_candidates.append((combined_distance, text))

    if text_candidates:
        text_candidates.sort(key=lambda tup: tup[0])  # Sort by combined distance
        best = text_candidates[0]
        return [{"caption": best[1]}]

    return []

def find_table_header(table_item, doc):
    """
    Finds header for given Docling 'Table', using Euclidean distance and page + overlap filtering.
    """
    if not table_item.prov or not hasattr(table_item.prov[0], 'bbox'):
        return []

    tbl_bbox = table_item.prov[0].bbox
    tbl_page = table_item.prov[0].page_no
    tbl_center_x = (tbl_bbox.l + tbl_bbox.r) / 2
    tbl_center_y = (tbl_bbox.b + tbl_bbox.t) / 2

    pattern = re.compile(r"^tables?\s+([ivxlcdm\d]+)", re.IGNORECASE)
    #pattern = re.compile(r"^figure\s+([ivxlcdm\d]+)", re.IGNORECASE)
    caption_candidates = []

    #print(f"Doc in find caption for figure: {doc.pages if doc.pages else 'No Pages'}")
    for text_item in doc.texts:
        text = text_item.text.strip()
        match = pattern.match(text)
        if not match:
            continue
        #print(f'Table caption match: {text}!!!!!!!!!\n\n')

        if not text_item.prov or not hasattr(text_item.prov[0], 'bbox'):
            continue

        caption_prov = text_item.prov[0]
        if caption_prov.page_no != tbl_page:
            continue # Filter only same-page captions

        caption_bbox = caption_prov.bbox
        cap_center_x = (caption_bbox.l + caption_bbox.r) / 2
        cap_center_y = (caption_bbox.b + caption_bbox.t) / 2

        # Compute Euclidean distance
        dx = cap_center_x - tbl_center_x
        dy = cap_center_y - tbl_center_y
        dist = math.hypot(dx, dy)

        # Use the top closest Figure caption
        try:
            tbl_no_raw = match.group(1)
            tbl_no = roman_to_int(tbl_no_raw) if tbl_no_raw.isalpha() else int(tbl_no_raw)
        except ValueError:
            tbl_no = None

        caption_candidates.append((dist, text, tbl_no))

    if caption_candidates:
        caption_candidates.sort(key=lambda tup: tup[0])  # sort by distance
        best = caption_candidates[0]
        return [{"caption": best[1], "table_number": best[2]}]

    return []

def find_caption_for_table(table_item, doc):
    """
    Finds nearby 'Table' captions anywhere near the table (not just below),
    using Euclidean distance and page + overlap filtering.
    """
    if not table_item.prov or not hasattr(table_item.prov[0], 'bbox'):
        return []

    tbl_bbox = table_item.prov[0].bbox
    tbl_page = table_item.prov[0].page_no
    tbl_center_x = (tbl_bbox.l + tbl_bbox.r) / 2
    tbl_center_y = (tbl_bbox.b + tbl_bbox.t) / 2

    pattern = re.compile(r"^tables?\s+([ivxlcdm\d]+)", re.IGNORECASE)
    #pattern = re.compile(r"^figure\s+([ivxlcdm\d]+)", re.IGNORECASE)
    caption_candidates = []

    #print(f"Doc in find caption for figure: {doc.pages if doc.pages else 'No Pages'}")
    for text_item in doc.texts:
        text = text_item.text.strip()
        match = pattern.match(text)
        if not match:
            continue
        #print(f'Table caption match: {text}!!!!!!!!!\n\n')

        if not text_item.prov or not hasattr(text_item.prov[0], 'bbox'):
            continue

        caption_prov = text_item.prov[0]
        if caption_prov.page_no != tbl_page:
            continue # Filter only same-page captions

        caption_bbox = caption_prov.bbox
        cap_center_x = (caption_bbox.l + caption_bbox.r) / 2
        cap_center_y = (caption_bbox.b + caption_bbox.t) / 2

        # Compute Euclidean distance
        dx = cap_center_x - tbl_center_x
        dy = cap_center_y - tbl_center_y
        dist = math.hypot(dx, dy)

        # Use the top closest Figure caption
        try:
            tbl_no_raw = match.group(1)
            tbl_no = roman_to_int(tbl_no_raw) if tbl_no_raw.isalpha() else int(tbl_no_raw)
        except ValueError:
            tbl_no = None

        caption_candidates.append((dist, text, tbl_no))

    if caption_candidates:
        caption_candidates.sort(key=lambda tup: tup[0])  # sort by distance
        best = caption_candidates[0]
        return [{"caption": best[1], "table_number": best[2]}]

    return []

def find_caption_for_figure(picture_item, doc):
    """
    Finds nearby 'Figure' captions anywhere near the figure (not just below),
    using Euclidean distance and page + overlap filtering.
    """
    if not picture_item.prov or not hasattr(picture_item.prov[0], 'bbox'):
        return []

    fig_bbox = picture_item.prov[0].bbox
    fig_page = picture_item.prov[0].page_no
    fig_center_x = (fig_bbox.l + fig_bbox.r) / 2
    fig_center_y = (fig_bbox.b + fig_bbox.t) / 2

    #pattern = re.compile(r"^(fig(?:ure)?)[\.:]?\s*([ivxlcdmIVXLCDM\d]+)", re.IGNORECASE)
    pattern = re.compile(r"^(?:fig(?:ure)?\.?)\s*#?\s*([ivxlcdmIVXLCDM\d]+)\b", re.IGNORECASE)
    #pattern = re.compile(r"^figure\s+([ivxlcdm\d]+)", re.IGNORECASE)
    caption_candidates = []

    #print(f"Doc in find caption for figure: {doc.pages if doc.pages else 'No Pages'}")
    for text_item in doc.texts:
        text = text_item.text.strip()
        match = pattern.match(text)
        if not match:
            continue
        #print(f'Figure caption match: {text}!!!!!!!!\n\n')

        if not text_item.prov or not hasattr(text_item.prov[0], 'bbox'):
            continue

        caption_prov = text_item.prov[0]
        if caption_prov.page_no != fig_page:
            continue # Filter only same-page captions

        caption_bbox = caption_prov.bbox

        # Only captions below and to the right are valid
        # is_below = caption_bbox.t <= fig_bbox.b
        # is_right = caption_bbox.l >= fig_bbox.r
        # if not (is_below or is_right):
        #     continue

        cap_center_x = (caption_bbox.l + caption_bbox.r) / 2
        cap_center_y = (caption_bbox.b + caption_bbox.t) / 2

        # Compute Euclidean distance
        dx = cap_center_x - fig_center_x
        dy = cap_center_y - fig_center_y
        dist = math.hypot(dx, dy)

        # Use the top closest Figure caption
        try:
            fig_no_raw = match.group(1)
            fig_no = roman_to_int(fig_no_raw) if fig_no_raw.isalpha() else int(fig_no_raw)
        except ValueError:
            fig_no = None

        caption_candidates.append((dist, text, fig_no))

    if caption_candidates:
        caption_candidates.sort(key=lambda tup: tup[0])  # sort by distance
        best = caption_candidates[0]
        #print(f'First best for figure {best[2]}: {best}')
        return [{"caption": best[1], "figure_number": best[2]}]

    return []

def normalize_caption(caption):
    return " ".join(caption.lower().strip().split())

def encode_image(image_path: str):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def valid_section_header(text):
    section_headers = set(["Summary", "Abstract", "Discussion", "Methods", "Materials and Methods", "Bibliography",
                          "References", "Introduction", "Acknowledgements", "Results", "Acknowledgments",
                          'Author Summary'])
    section_header_types = {'Summary': 'Summary', 'Abstract': 'Abstract', 'Discussion': 'Discussion',
                            'Methods': 'Materials and Methods', 'Bibliography': 'References',
                            'Acknowledgements': 'Acknowledgements', 'Materials and Methods': 'Materials and Methods',
                            'References': 'References', 'Introduction': 'Introduction', 'Results': 'Results',
                            'Author Summary': 'Summary', 'Acknowledgments': 'Acknowledgements'}
    section_header_types = {p.lower(): section_header_types[p] for p in section_header_types}
    section_headers = set([p.lower() for p in section_headers])
    if text.strip().lower() in section_headers:
        return section_header_types[text.lower()]
    return False


def extract_texts_from_refs(refs, doc):
    """Helper to resolve cref-based text references (captions, footnotes)."""
    texts = []
    for ref in refs:
        try:
            index = int(ref.cref.split("/")[-1])
            text = doc.texts[index].text.strip()
            if text:
                texts.append(text)
        except (IndexError, ValueError, AttributeError):
            continue
    return texts


def extract_elements_basic(pdf_path, output_path):
    logging.basicConfig(level=logging.INFO)

    input_doc_path = pdf_path
    pdf_prefix = pdf_path.split('/')[-1].split('.')[0]
    output_dir = Path(f"{output_path}/{pdf_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(input_doc_path)
    doc = conv_res.document
    doc_filename = conv_res.input.file.stem

    # Create data structure for holding a mapping of extracted visual elements (tables & figures) to their corresponding captions
    visual_index = {
        "Tables": [],
        "Images": []
    }

    # Create string containing the full text of the paper
    full_text = ''
    text_sections = []
    text_section_types = []
    all_headers =[]
    header = ''
    for i, text_item in enumerate(doc.texts):
        if text_item.label.name == 'SECTION_HEADER':
            all_headers.append(text_item.text)
            if i == 0:
                section_type = "Title and Authors"
            else:
                section_type = valid_section_header(text_item.text)
            if section_type:
                header = section_type
        full_text += text_item.text + ' '
        text = text_item.text.strip()
        text_sections.append(text)
        text_section_types.append(header)

    # --- TABLE SEARCH + SAVE --- (search table elements for peptide sequences)
    for i, table_item in enumerate(doc.tables):
        if hasattr(table_item, "data") and table_item.data and hasattr(table_item.data, "grid"):
            table_image_path = output_dir / f"{doc_filename}-table-{i + 1}.png"

            with table_image_path.open("wb") as fp:
                table_item.get_image(doc).save(fp, "PNG")

            table_text = []
            for row in table_item.data.grid:
                for cell in row:
                    if cell.text:
                        table_text.append(cell.text.strip())

            # TO DO: Search captions for epitopes
            captions = extract_texts_from_refs(table_item.captions, doc)
            footnotes = extract_texts_from_refs(table_item.footnotes, doc)
            fallback_header = find_table_header(table_item, doc)

            if fallback_header:
                header_text = fallback_header[0]["caption"]
                captions = [header_text]  # make sure this is a list of str

            fallback_footnote = find_table_footnote(table_item, doc)
            #print(f"Fallback footnote for {table_image_path}: {fallback_footnote}")
            if fallback_footnote:
                footer_text = fallback_footnote[0]["caption"]
                # tbl_number = fallback_footnote[0]["table_number"]
                #print(f"\n🧠 Fallback Caption Found for Table #{tbl_number}: {caption_text}\n")
                footnotes = [footer_text]  # make sure this is a list of str

            #print(f"Matched sequences before entering in master visual index: {matched_sequences}")
            # Add to master visual_index
            visual_index["Tables"].append({
                "image_path": str(table_image_path),
                "captions": captions,
                "footnotes": footnotes,
                'text': table_text,
            })

    # --- FIGURE SEARCH + SAVE --- (search figure captions for peptide sequences)
    for i, fig_item in enumerate(doc.pictures):

        image_path = output_dir / f"{doc_filename}-picture-{i + 1}.png"
        captions = extract_texts_from_refs(fig_item.captions, doc)
        #print(f"Caption before fallback for Picture {i+1}: {captions}")
        #print(f"Immediate caption from figure {i}: {captions}")
        footnotes = extract_texts_from_refs(fig_item.footnotes, doc)
        
        #if not captions:
        fallback_caption = find_caption_for_figure(fig_item, doc)
        if fallback_caption:
            caption_text = fallback_caption[0]["caption"]
            captions = [caption_text]  # make sure this is a list of str

        # Save image before running VLM inference
        with image_path.open("wb") as fp:
            fig_item.get_image(doc).save(fp, "PNG")

        #print(f"Valid matches from {image_path}: {list(set(valid_matches))}")
        # Add to master visual_index
        visual_index["Images"].append({
            "image_path": str(image_path),
            "captions": captions,
            "footnotes": footnotes,
        })


    # Group image entries by exact caption text (assumes that all extracted images with the same caption are actually all in one figure)
    grouped_images = defaultdict(list)
    for entry in visual_index["Images"]:
        # Use a tuple of all captions joined (you can normalize spacing if needed)
        caption_key = " ".join(entry.get("captions", [])).strip()
        if caption_key:
            grouped_images[caption_key].append(entry)

        # If image does not have a caption found with it, use its path as caption_key
        else:
            grouped_images[entry['image_path']].append(entry)

    # Replace groups of images with stitched versions
    deduped_images = []

    for caption, entries in grouped_images.items():
        if len(entries) == 1:
            deduped_images.append(entries[0])
            continue

        # Prepare paths
        image_paths = [e['image_path'] for e in entries]
        # print(f"Image paths from grouped images: {image_paths}")
        fig_no = get_figure_number_from_caption(caption)
        base_image_name = Path(image_paths[0]).stem.split("-picture")[0]
        new_image_path = output_dir / f"{base_image_name}-stitched-{fig_no}.png"

        # Stitch and save
        stitched_path = stitch_images(image_paths, new_image_path, direction="vertical")

        # Create new unified entry
        new_entry = {
            "image_path": stitched_path,
            "captions": entries[0].get("captions", []),
            "footnotes": []
        }

        deduped_images.append(new_entry)

    # Replace the original list with the deduped/stitched one
    visual_index["Images"] = deduped_images
    #print(f"Visual index after removing duplicates and stitch together images: {visual_index}")

    def normalize_caption(caption):
        return " ".join(caption.lower().strip().split())

    # Build a mapping of normalized caption → stitched image entry
    stitched_by_caption = {
        normalize_caption(" ".join(entry.get("captions", []))): entry
        for entry in visual_index["Images"]
    }

    visual_index['Full text'] = full_text
    visual_index['Text sections'] = text_sections
    visual_index['Text section Headers'] = text_section_types
    visual_index['Stitched image mapping'] = stitched_by_caption
    with open(f'{output_dir}/content.p', 'wb') as file_object:
        pickle.dump(visual_index, file_object)
    return visual_index, full_text, all_headers

def pad_image_to_64(image_path: str):
    image = Image.open(image_path)
    w, h = image.size
    if (w % 64 == 0) and (h % 64 == 0):
        return image_path
    print(f'Resizing image: {image_path}')
    new_w = math.ceil(w / 64) * 64
    new_h = math.ceil(h / 64) * 64
    new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    new_img.paste(image, (0, 0))
    new_img.save(image_path.replace('.png', '_aligned.png'))
    return image_path.replace('.png', '_aligned.png')


def align_objects(assays_object: Dict, epitope_object: Dict, mhc_object: Dict):
    """
    Function to align separate assay data structure (where key=image_path, value=assay(s))and
    epitope and MHC data structures.
    
    :param assays_object: dictionary where key=image_path, value=assay(s)
    :type assays_object: Dict
    :param epitope_object: dictionary where key=peptide, value=peptide data structure including image path where applicable.
    :type epitope_object: Dict
    :param mhc_object: dictionary where key=MHC molecule name, value=peptide data structure including image path where applicable.
    :type mhc_object: Dict
    """
    visual_index = defaultdict(lambda: {'mhc': [], 'epitope':[], 'assay':[]})
    for mhc_allele in mhc_object:
        for image_type in ['Tables', 'Images']:
            for element in mhc_object[mhc_allele].get(image_type, {}):
                visual_index[element['image_path']]['mhc'].append(mhc_allele)
    for epitope in epitope_object:
        for image_type in ['Tables', 'Images']:
            for element in epitope_object[epitope].get(image_type, {}):
                visual_index[element['image_path']]['epitope'].append(epitope)
    for image_path in assays_object:
        visual_index[image_path]['assay'] += assays_object[image_path]
    for image_path in visual_index:
        for object_type in ['mhc', 'epitope']:
            visual_index[image_path][object_type] = list(set(visual_index[image_path][object_type]))
    return visual_index