import argparse
import json
import logging
import os
import re

from tqdm import tqdm

from datasets import Dataset

from tex.process_tex import extract, get_tables

logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

total_papers = 0
DPO, KTO, IPO, SIMPO, GPO, SLiC, IRPO, OPPO, RPO, ORPO = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

PREFERENCE_ALGOS = {
    "DPO": ["Direct Preference Optimization", DPO],
    "KTO": ["Kahneman-Tversky Optimization", KTO],
    "IPO": ["Implicit Preference Optimization", IPO],
    "SimPO": ["Simple Preference Optimization", SIMPO],
    "GPO": ["Generalized Preference Optimization", GPO],
    "SLiC-HF": ["Sequence Likelihood Calibration with Human Feedback", SLiC],
    "IRPO": ["Iterative Reasoning Preference Optimization", IRPO],
    "OPPO": ["Offline Preference-guided Policy Optimization", OPPO],
    "RPO": ["Relative Preference Optimization", RPO],
    "ORPO": ["Odds Ratio Preference Optimization", ORPO]
}

REGEX_PATTERNS = {}

for abbr, full_name_tuple in PREFERENCE_ALGOS.items():
    full_name = full_name_tuple[0]  # safely get from tuple

    patterns = re.compile(
    rf"""
    (?!.*\\cite[^{{}}]*{re.escape(abbr)}[^{{}}]*}})  # Strict citation blocker
    (?<!\w)(?<!-)                                   # No preceding word char/hyphen
    (?:
        \(\s*{re.escape(abbr)}\s*\)                 # (IPO)
        | \b{re.escape(abbr)}\b                     # IPO as word
        | \b{re.escape(full_name)}\b                # Full name as words
    )
    (?!\w)(?!-)                                     # No following word char/hyphen
    """,
    re.IGNORECASE | re.VERBOSE
)


    REGEX_PATTERNS[abbr] = patterns

def detect_algorithms(latex_code):
    """
    Returns a list of detected preference optimization algorithms used in a LaTeX paper.
    """
    matches = set()
    for abbr, pattern in REGEX_PATTERNS.items():
        # Search for the pattern in the text
        match = pattern.search(latex_code)
        if match:
            matches.add(abbr)
            # Log the match details
            logging.info(
                f"Found '{abbr}' match at position {match.start()}-{match.end()}: "
                f"'{match.group()}' in context: '{get_context(latex_code, match.start(), match.end())}'"
            )
    
    return sorted(matches)


def extract_full_paper_and_table_src(paper_src_path):
    extracted_data = extract(paper_src_path)
    if extracted_data is not None:
        paper_id = paper_src_path.split('/')[-1].strip()
        extracted_data['paper_id'] = paper_id
    return extracted_data

def get_context(text, start_pos, end_pos, context_chars=30):
    """Get surrounding context for a match"""
    context_start = max(0, start_pos - context_chars)
    context_end = min(len(text), end_pos + context_chars)
    return text[context_start:context_end].replace('\n', ' ')


def main(ml_domain_path, output_path):
    global total_papers, PREFERENCE_ALGOS
    logging.info("Extracting from arxiv src")
    domain_papers = ['arxiv_src_2023', 'arxiv_src_2024']
    domain_papers_dir = []

    for year_folder in domain_papers:
        year_path = os.path.join(ml_domain_path, year_folder)
        for paper in os.listdir(year_path):
            paper_path = os.path.join(year_path, paper)
            domain_papers_dir.append(paper_path)
    domain_papers_dir.sort()

    extracted_arxiv_srcs = []
    for paper_dir in tqdm(domain_papers_dir):
        result = extract_full_paper_and_table_src(paper_dir)
        if result and all(value is not None for value in result.values()):
            algos = detect_algorithms(result['full_paper_latex_code'])
            if algos:
                for algo in algos:
                    PREFERENCE_ALGOS[algo][1] += 1
                    logging.info(f"{algo}")
                extracted_arxiv_srcs.append(result)
                total_papers += 1
                logging.info(f"{result['paper_id']}")

    logging.info("Original length of papers: {}".format(len(domain_papers_dir)))
    logging.info("Extracted length of papers from arxiv src: {}".format(len(extracted_arxiv_srcs)))
    
    n = 0
    for i in extracted_arxiv_srcs:
        n += len(i['tables_list'])
    
    logging.info("Valid Table Latex from arxiv src: {}".format(n))

    huggingface_dict = {}
    huggingface_dict['paper_id'] = [extracted_arxiv_src['paper_id'] for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['tables_list'] = [json.dumps(extracted_arxiv_src['tables_list']) for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['tables_index'] = [json.dumps(extracted_arxiv_src['tables_index']) for extracted_arxiv_src in extracted_arxiv_srcs]
    huggingface_dict['full_paper_latex_code'] = [extracted_arxiv_src['full_paper_latex_code'] for extracted_arxiv_src in extracted_arxiv_srcs]
    
    for key in huggingface_dict:
        logging.info(f"{key} has {len(huggingface_dict[key])} entries")

    logging.info( f"{len(huggingface_dict['paper_id']) == len(huggingface_dict['tables_list']) == len(huggingface_dict['tables_index']) ==len(huggingface_dict['full_paper_latex_code'])}")

    dataset = Dataset.from_dict(huggingface_dict)
    logging.info(f"{len(os.listdir(output_path))}")
    dataset.save_to_disk(output_path)
    logging.info(f"{len(os.listdir(output_path))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_ml_path', type=str)
    parser.add_argument('--ml_table_ds', type=str)
    args = parser.parse_args()

    main(args.extracted_ml_path, args.ml_table_ds)
    logging.info("Finished extracting tables from arxiv sources")
    for algo, count in PREFERENCE_ALGOS.items():
        logging.info(f"{algo}={count[1]}")
    logging.info(f"Total={total_papers}")