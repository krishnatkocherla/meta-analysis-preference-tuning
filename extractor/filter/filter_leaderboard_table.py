import argparse
import json
import logging
import os

from copy import deepcopy
from tqdm import tqdm

import torch
import transformers
from datasets import Dataset, load_from_disk


from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def download_and_cache_model(model_name_or_path: str, cache_dir: str):
    logging.info(f"Pre-downloading model and tokenizer: {model_name_or_path} to cache: {cache_dir}")
    AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)


logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
)

MESSAGES_TEMPLATE = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "<|PROMPT|>"},
    ]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

def llm_process(prompt: str, llm: LLM, max_new_tokens: int) -> str:
    print(f"Prompt sample:\n{prompt[:500]}")
    
    if len(prompt) > 10000:
        prompt = prompt[:10000]

    full_prompt = (
        "You are an AI assistant.\n"
        f"User: {prompt}\n"
        "Assistant:"
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    output = llm.generate([full_prompt], sampling_params)
    response = output[0].outputs[0].text.strip()
    return response



def filter_table(dataset: Dataset, pipeline, template_prompt: str) -> Dataset:
    fildered_dataset = []

    for instance in tqdm(dataset):
        tables_list = json.loads(instance['tables_list'])
        tables_index = json.loads(instance['tables_index'])
        
        filtered_index = []
        for i, table in enumerate(tables_list):
            table_src = table[0]
            prompt = deepcopy(template_prompt)
            prompt = prompt.replace("{Table LaTeX}", table_src)
            output = llm_process(prompt, pipeline, max_new_tokens=32)

            if 'true' in output.lower():
                filtered_index.append(i)
        
        for idx in filtered_index:
            filtered_instance = {
                'paper_id': instance['paper_id'],
                'full_tex': instance['full_paper_latex_code'],
                'table_source': tables_list[idx][0],
                'table_index': tables_index[idx],
            }
            fildered_dataset.append(filtered_instance)            

    logging.info(f"Filtered Dataset Length with Tables of Interest: {len(fildered_dataset)}")
    unique_paper_ids = set(item['paper_id'] for item in fildered_dataset)
    logging.info(f"Number of unique papers with preference tuning tables: {len(unique_paper_ids)}")

    keys = fildered_dataset[0].keys()
    huggingface_dict = {}
    for k in keys:
        huggingface_dict[k] = [fildered_domain_instance[k] for fildered_domain_instance in fildered_dataset]
    filtered_dataset = Dataset.from_dict(huggingface_dict)
    
    return filtered_dataset


def main(ml_domain_table_ds: str, output_path: str, model_name_or_path: str, prompt_path: str, 
         cache_dir: str, shard_idx: int, total_shards: int):

    logging.info(f"Classifying whether the table contains the leaderboard table")
    
    # download_and_cache_model(model_name_or_path, cache_dir)

    logging.info("Loading dataset...")
    dataset = load_from_disk(ml_domain_table_ds)

    logging.info(f"Processing shard idx {shard_idx+1} out of {total_shards} shards")
    logging.info(f"Filtered outputs will be saved in {output_path}")

    logging.info("Sharding dataset...")
    dataset = dataset.shard(num_shards=total_shards, index=shard_idx)
    logging.info(f"Shard size: {len(dataset)}")
    
    logging.info("Loading pipeline...")
    pipeline = LLM(
    model=model_name_or_path,
    tokenizer=model_name_or_path,
    dtype="float16",
    tensor_parallel_size=4,  # Adjust for multi-GPU
    trust_remote_code=True,
    download_dir=cache_dir,
    max_model_len=8192,
    )

    logging.info("Pipeline loaded, beginning filtering")

    with open(prompt_path, 'r') as f:
        template_prompt = f.read().strip()
    logging.info("Beginning table filtering...")

    filtered_dataset = filter_table(dataset, pipeline, template_prompt)    
    filtered_dataset.save_to_disk(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ml_table_ds', type=str)
    parser.add_argument('--ml_leaderboard_table_ds', type=str)
    parser.add_argument('--prompt_path', type=str, default="/srv/nlprx-lab/share6/kkocherla3/arxiv_src/pipeline/src/extractor/filter/prompt/filter_preference.txt")
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)
    args = parser.parse_args()
    
    main(args.ml_table_ds, args.ml_leaderboard_table_ds, args.model_name_or_path, args.prompt_path,
         args.cache_dir, args.shard_idx, args.total_shards)
    logging.info("Finished filtering leaderboard tables")