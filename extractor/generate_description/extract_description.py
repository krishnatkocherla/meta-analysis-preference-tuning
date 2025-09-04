import os
import logging
import argparse
import re

from copy import deepcopy
from datasets import load_from_disk

import torch
import transformers
from datasets import Dataset, load_from_disk
from datasets import concatenate_datasets, load_dataset
from extractor.preprocess.tex.process_tex import preprocess_tex_src


from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from transformers import LogitsProcessor

# https://muellerzr.github.io/til/end_thinking.html
class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float('-inf')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1

        if (
            scores is None
            or scores.numel() == 0
            or scores.dim() < 2
            or scores.shape[0] == 0
            or scores.shape[1] == 0
        ):
            return scores  # skip processing if logits are empty or improperly shaped

        if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
            scores[:] = self.neg_inf
            if self.nl_token < scores.shape[-1]:
                scores[0][self.nl_token] = 0
            if self.think_end_token < scores.shape[-1]:
                scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            ratio = self.tokens_generated / self.max_thinking_tokens

            if ratio > 0.95:
                if self.nl_token < scores.shape[-1] and self.think_end_token < scores.shape[-1]:
                    boost = 1 + ratio
                    scores[0][self.nl_token] = scores[0][self.think_end_token] * boost
                    scores[0][self.think_end_token] = scores[0][self.think_end_token] * boost

            if self.tokens_generated >= (self.max_thinking_tokens - 1):
                scores[:] = self.neg_inf
                if self.tokens_generated == self.max_thinking_tokens - 1:
                    if self.nl_token < scores.shape[-1]:
                        scores[0][self.nl_token] = 0
                else:
                    if self.think_end_token < scores.shape[-1]:
                        scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores


logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

def truncate_length(prompt, tokenizer):
    temp = tokenizer.encode(prompt)
    if len(temp) > 32768:
        temp = temp[:32768]
        truncated_prompt = tokenizer.decode(temp)
    else:
        truncated_prompt = prompt
    return truncated_prompt

def extract_output(qwen_output):
    # Extract text after </think>
    if qwen_output.count("<think>")  == 1 and qwen_output.count("</think>") == 0:
        return "<FAILED>"
    after_think = qwen_output.split("</think>", 1)[-1].strip()
    return after_think


def clean_arxiv_tex(tex_content):  
    tex_content = re.sub(r'\\appendix.*?\\end{document}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'\\begin{figure}.*?\\end{figure}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'%.*?\n', '', tex_content)   # Remove comments
    tex_content = re.sub(r'\s+', ' ', tex_content)   # Remove extra whitespace
    tex_content = re.sub(r'\\begin{thebibliography}.*?\\end{thebibliography}', '', tex_content, flags=re.DOTALL)  # Remove bibliography
    tex_content = re.sub(r'\\documentclass.*?\\begin{document}', '\\begin{document}', tex_content, flags=re.DOTALL)  # Remove packages
    tex_content = re.sub(r'\\(usepackage|documentclass|title|author|date|maketitle).*?\n', '', tex_content)  # Remove certain commands  
    tex_content = re.sub(r'\\cite[t|p]?{.*?}', '', tex_content)  # Remove citations
    tex_content = re.sub(r'\\begin{equation}.*?\\end{equation}', '', tex_content, flags=re.DOTALL) # Remove equations
    tex_content = re.sub(r'\\begin{align}.*?\\end{align}', '', tex_content, flags=re.DOTALL)  # Remove equations
    tex_content = re.sub(r'\\begin{table}.*?\\end{table}', '', tex_content, flags=re.DOTALL)

    return tex_content  


def main(args):

    logging.info(args)
        
    dataset = load_from_disk(args.hf_ds_path)
    
    dataset_names = []
    for i in dataset:
        dataset_name = i['context_augmented_table_results_extracted']['dataset']
        dataset_names.append(dataset_name)

    subsets = []
    for i in dataset:
        subset = i['context_augmented_table_results_extracted']['subset']
        subsets.append(subset)

    assert len(dataset_names) == len(subsets)

    dataset_subset_concats = []
    for i in range(len(dataset_names)):
        if subsets[i] == "xx":
            dataset_subset_concats.append(dataset_names[i])
        else:
            dataset_subset_concats.append(dataset_names[i] + "<|SEP|>" + subsets[i])

    dataset_descriptions = dataset['dataset_description']
    paper_ids = dataset['paper_id']
    citation_tags = dataset['dataset_link']

    dataset_info = list(zip(dataset_subset_concats, dataset_descriptions, paper_ids, citation_tags))
    
    target_dataset_info = []
    for dataset_subset, dataset_description, paper_id, citation_tag in dataset_info:
        if dataset_description == "<FAILED>":
            target_dataset_info.append((dataset_subset, dataset_description, paper_id, citation_tag))

    target_dataset_info = list(set(target_dataset_info))

    logging.info("Loading pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pipeline = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype="float16",
        tensor_parallel_size=4,  # Adjust for multi-GPU
        trust_remote_code=True,
        download_dir=args.cache_dir
    )
    
    processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=25000)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32000, logits_processors=[processor])


    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    # prompt_list = []
    results = []
    source_list = []
    for i, (dataset_subset, dataset_description, paper_id, citation_tag) in enumerate(target_dataset_info):

        orig_source_path = args.orig_source_path
        citation_source_path = args.citation_source_path

        if len(citation_tag) > 0:
            source_path = os.path.join(citation_source_path, citation_tag)
            source_list.append(citation_tag)
        else:
            # Determine the year folder based on first two digits of the paper_id
            year_prefix = paper_id[:2]
            if year_prefix == "23":
                year_folder = "arxiv_src_2023"
            elif year_prefix == "24":
                year_folder = "arxiv_src_2024"
            else:
                year_folder = ""  # fallback: no year folder, just use orig_source_path

            if year_folder:
                source_path = os.path.join(orig_source_path, year_folder, paper_id)
            else:
                source_path = os.path.join(orig_source_path, paper_id)

            source_list.append(paper_id)

        
        tex_content = preprocess_tex_src(source_path)
        tex_content = clean_arxiv_tex(tex_content)

        prompt = deepcopy(prompt_template)
        
        if "<|SEP|>" not in dataset_subset:
            dataset_name = dataset_subset
            subset = "xx"
        else:
            dataset_name, subset = dataset_subset.split("<|SEP|>")

        prompt = prompt.replace("{{dataset}}", dataset_name)
        prompt = prompt.replace("{{subset}}", subset)
        prompt = prompt.replace("{{text_source}}", tex_content)
        message = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = truncate_length(text, tokenizer)
        outputs = pipeline.generate([text], sampling_params)
        for output in outputs:
            results.append(extract_output(output.outputs[0].text))
        # prompt_list.append(prompt)

    # prompt_list = truncate_length(prompt_list)

    assert len(results) == len(source_list)

    unique_dataset_subset_to_description = {}
    for i in range(len(target_dataset_info)):
        unique_dataset_subset_to_description[target_dataset_info[i][0]] = (results[i], source_list[i])

    dataset_descriptions_augmented = deepcopy(dataset['dataset_description'])
    dataset_description_source = deepcopy(dataset['description_source'])
    
    m = 0
    n = 0

    for i, instance in enumerate(dataset_subset_concats):
        if instance in unique_dataset_subset_to_description.keys():
            dataset_description = unique_dataset_subset_to_description[instance][0]
            dataset_descriptions_augmented[i] = dataset_description
            dataset_description_source[i] = unique_dataset_subset_to_description[instance][1]
            if '<FAILED>' in dataset_description:
                m += 1
            else:
                n += 1
            
    logging.info(f"Number of failed dataset extractions: {m} out of {m + n}")

    dataset = dataset.remove_columns('dataset_description')
    dataset = dataset.remove_columns('description_source')
    dataset = dataset.add_column('dataset_description', dataset_descriptions_augmented)
    dataset = dataset.add_column('description_source', dataset_description_source)
    
    logging.info(f"saving {len(dataset)} dataset descriptions")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Extraction Generation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default='./extractor/generate_description/prompt/extract_description.txt')
    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    parser.add_argument('--orig_source_path', type=str)
    parser.add_argument('--citation_source_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()
    
    logging.info(f"Start Extracting Dataset Descriptions")

    main(args)
