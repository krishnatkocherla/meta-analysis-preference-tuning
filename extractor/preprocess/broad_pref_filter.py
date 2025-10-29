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

# Expanded keyword list for preference tuning in LLMs
KEYWORDS = [
    # RLHF / human feedback methods
    "RLHF", "RLAIF", "RL from human feedback", "RL from AI feedback",
    "reinforcement learning with human feedback",
    "reinforcement learning from human preferences",
    "reinforcement learning from AI preferences",
    "human preference reinforcement learning",
    "learning from human feedback", "learning from preferences",
    "Advantage Weighted Reinforcement Learning from Human Feedback",
    "AWRHF",

    # Direct/Implicit Preference Optimization variants
    "PPO", "Proximal Preference Optimization",
    "DPO", "Direct Preference Optimization",
    "IPO", "Implicit Preference Optimization",
    "f-DPO", "f-divergence Direct Preference Optimization",
    "KTO", "Kahneman-Tversky Optimization",
    "ORPO", "Odds Ratio Preference Optimization",
    "SimPO", "Simple Preference Optimization",
    "RRHF", "Rank Responses with Human Feedback",
    "CPL", "Contrastive Preference Learning",
    "RPO", "Relative Preference Optimization",
    "OPPO", "Offline Preference-guided Policy Optimization",
    "IRPO", "Iterative Reasoning Preference Optimization",
    "SLiC", "SLiC-HF", "Sequence Likelihood Calibration with Human Feedback",
    "GPO", "Generalized Preference Optimization",
    "GRPO", "Group Relative Policy Optimization",
    "Preference-aware fine-tuning", "preference-conditioned fine-tuning",

    # Reward modeling / alignment / human evaluation
    "reward modeling", "reward model", "reward function learning",
    "preference model", "preference predictor", "preference score model",
    "human alignment", "AI alignment", "value alignment",
    "alignment tuning", "safety tuning", "instruction preference",
    "human evaluation", "feedback-based training", "feedback optimization",
    "choice modeling", "comparison-based learning",

    # Instruction tuning / prompt-based preference
    "instruction fine-tuning", "instruction-tuned", "prompt preference",
    "prompt-based preference", "instruction optimization", "prompt optimization",
    "human-in-the-loop tuning", "interactive preference tuning",

    # General preference optimization terms
    "preference optimization", "preference tuning", "preference-based fine-tuning",
    "preference-based RL", "preference learning", "preference-aware training",
    "ranking-based fine-tuning", "preference signal", "preference supervision",
    "reward shaping", "policy shaping", "feedback shaping", "guided fine-tuning"
]

# Build regex patterns for each keyword (looser matching)
REGEX_PATTERNS = {}
for kw in KEYWORDS:
    patterns = re.compile(
        rf"""
        (?!.*\\cite[^{{}}]*{re.escape(kw)}[^{{}}]*}})   # avoid citations
        (?<!\w)(?<!-)                                    # no bad prefix
        {re.escape(kw)}                                  # the keyword
        (?:s)?                                           # allow plural 's'
        (?!\w)(?!-)                                      # no bad suffix
        """,
        re.IGNORECASE | re.VERBOSE
    )
    REGEX_PATTERNS[kw] = patterns


def get_context(text, start_pos, end_pos, context_chars=30):
    """Get surrounding context for a match"""
    context_start = max(0, start_pos - context_chars)
    context_end = min(len(text), end_pos + context_chars)
    return text[context_start:context_end].replace('\n', ' ')


def detect_keywords(latex_code):
    """
    Returns a list of detected preference-related keywords in the paper.
    """
    matches = set()
    for kw, pattern in REGEX_PATTERNS.items():
        match = pattern.search(latex_code)
        if match:
            matches.add(kw)
            logging.info(
                f"Found '{kw}' in context: "
                f"'{get_context(latex_code, match.start(), match.end())}'"
            )
    return sorted(matches)


def extract_full_paper_and_table_src(paper_src_path):
    extracted_data = extract(paper_src_path)
    if extracted_data is not None:
        paper_id = paper_src_path.split('/')[-1].strip()
        extracted_data['paper_id'] = paper_id
    return extracted_data


def main(ml_domain_path, output_path):
    global total_papers
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
            kws = detect_keywords(result['full_paper_latex_code'])
            if kws:  # keep if at least one keyword hit
                extracted_arxiv_srcs.append(result)
                total_papers += 1
                logging.info(f"{result['paper_id']} -> {kws}")

    logging.info("Original length of papers: {}".format(len(domain_papers_dir)))
    logging.info("Extracted length of papers from arxiv src: {}".format(len(extracted_arxiv_srcs)))
    
    n = sum(len(i['tables_list']) for i in extracted_arxiv_srcs)
    logging.info("Valid Table Latex from arxiv src: {}".format(n))

    huggingface_dict = {
        'paper_id': [src['paper_id'] for src in extracted_arxiv_srcs],
        'tables_list': [json.dumps(src['tables_list']) for src in extracted_arxiv_srcs],
        'tables_index': [json.dumps(src['tables_index']) for src in extracted_arxiv_srcs],
        'full_paper_latex_code': [src['full_paper_latex_code'] for src in extracted_arxiv_srcs],
    }

    for key in huggingface_dict:
        logging.info(f"{key} has {len(huggingface_dict[key])} entries")

    dataset = Dataset.from_dict(huggingface_dict)
    dataset.save_to_disk(output_path)
    logging.info("Saved dataset to disk")
    logging.info(f"Total kept papers={total_papers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_ml_path', type=str)
    parser.add_argument('--ml_table_ds', type=str)
    args = parser.parse_args()

    main(args.extracted_ml_path, args.ml_table_ds)
    logging.info("Finished extracting tables from arxiv sources")
