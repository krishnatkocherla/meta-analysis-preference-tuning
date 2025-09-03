import argparse
import json
import logging
import os
import re

from copy import deepcopy
from tqdm import tqdm

import torch
import transformers
from datasets import Dataset, load_from_disk
from datasets import concatenate_datasets, load_dataset


from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from transformers import LogitsProcessor
from tokenprocessor import ThinkingTokenBudgetProcessor


logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
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

def parse_extracted_results(response: str):
    list_of_dicts = []
    for line in response.strip().splitlines():
        try:
            temp = json.loads(line)
            # Ensure all values are strings
            for key in temp:
                temp[key] = str(temp[key])
            list_of_dicts.append(temp)
        except json.JSONDecodeError:
            continue
    return list_of_dicts

  

def clean_arxiv_tex_for_contextual_augmentation(tex_content):  
    tex_content = re.sub(r'\\appendix.*?\\end{document}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'\\begin{figure}.*?\\end{figure}', '', tex_content, flags=re.DOTALL)  
    tex_content = re.sub(r'%.*?\n', '', tex_content)   # Remove comments
    tex_content = re.sub(r'\s+', ' ', tex_content)   # Remove extra whitespace
    tex_content = re.sub(r'\\begin{thebibliography}.*?\\end{thebibliography}', '', tex_content, flags=re.DOTALL)  # Remove bibliography
    tex_content = re.sub(r'\\documentclass.*?\\begin{document}', '\\begin{document}', tex_content, flags=re.DOTALL)  # Remove packages
    tex_content = re.sub(r'\\(usepackage|documentclass|title|author|date|maketitle).*?\n', '', tex_content)  # Remove certain commands  
    tex_content = re.sub(r'\\begin{equation}.*?\\end{equation}', '', tex_content, flags=re.DOTALL) # Remove equations
    tex_content = re.sub(r'\\begin{align}.*?\\end{align}', '', tex_content, flags=re.DOTALL)  # Remove equations
    
    return tex_content


def main(args):
    logging.info("Loading dataset...")

    shard_dirs = sorted([
        os.path.join(args.hf_ds_path, d)
        for d in os.listdir(args.hf_ds_path)
        if os.path.isdir(os.path.join(args.hf_ds_path, d)) and d.startswith("shard_")
    ])

    dataset = concatenate_datasets([load_from_disk(shard) for shard in shard_dirs])

    logging.info(f"Processing shard idx {args.shard_idx+1} out of {args.total_shards} shards")
    logging.info(f"Filtered outputs will be saved in {args.hf_ds_output_path}")

    logging.info("Sharding dataset...")
    dataset = dataset.shard(num_shards=args.total_shards, index=args.shard_idx)
    logging.info(f"Shard size: {len(dataset)}")
    
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
    logging.info("Pipeline loaded, beginning extraction")

    full_tex_contents = []

    for i in dataset:
        full_tex_contents.append(clean_arxiv_tex_for_contextual_augmentation(i['full_tex']))

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    parsed_results = []
    for i, instance in enumerate(tqdm(dataset)):
        prompt = deepcopy(prompt_template)
        prompt = prompt.replace("{{records}}", instance['table_results_extracted'])
        prompt = prompt.replace("{{table_code}}", instance['table_source'])
        prompt = prompt.replace("{{text}}", full_tex_contents[i])
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = truncate_length(text, tokenizer)
        outputs = pipeline.generate([text], sampling_params)
        for output in outputs:
            parsed_results.append(parse_extracted_results(extract_output(output.outputs[0].text)))
            logging.info(extract_output(output.outputs[0].text))
            logging.info(parse_extracted_results(extract_output(output.outputs[0].text)))
            logging.info(instance['paper_id'])

    dataset = dataset.add_column("context_augmented_table_results_extracted", parsed_results)    

    logging.info(f"flattening dataset")

    parsed_results = dataset['context_augmented_table_results_extracted']
    flattened_dataset = {}
    flattened_results = []
    for column in dataset.column_names:
        if column != 'context_augmented_table_results_extracted':
            flattened_dataset[column] = []
    
    for idx, result_list in enumerate(parsed_results):
        num_results = len(result_list)
        flattened_results.extend(result_list)
        
        for column in dataset.column_names:
            if column != 'context_augmented_table_results_extracted':
                flattened_dataset[column].extend([dataset[column][idx]] * num_results)
    
    flattened_dataset['context_augmented_table_results_extracted'] = flattened_results
    dataset = dataset.from_dict(flattened_dataset)

    logging.info(f"saving {len(dataset)} context augemented records")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Context Augmentation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/context_augment.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()
    
    logging.info(f"Start Augmenting Database with Context")

    main(args)
    