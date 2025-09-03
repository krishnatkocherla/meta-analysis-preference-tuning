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

def extract_output(qwen_output):
    # Extract text after </think>
    if qwen_output.count("<think>")  == 1 and qwen_output.count("</think>") == 0:
        return "<FAILED>"
    after_think = qwen_output.split("</think>", 1)[-1].strip()
    return after_think


def filter_metric(dataset, prompt_template, pipeline, sampling_params, tokenizer):

    filtered_dataset = []
    adjusted_values = []
    standardized_metric_names = []

    results = []
    logging.info("Beginning filtering")
    for i, instance in enumerate(tqdm(dataset)):
        record = instance['context_augmented_table_results_extracted']
        metric = record['metric']
        value = record['value']
        pref_tuning_method = record['pref_tuning_method']

        if metric is None or value is None or pref_tuning_method == "xx":
            metric = "fluency"
            value = "0"

        prompt = deepcopy(prompt_template)
        prompt = prompt.replace("{{METRIC_NAME}}", metric)
        prompt = prompt.replace("{{METRIC_VALUE}}", value)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        outputs = pipeline.generate([text], sampling_params)
        for output in outputs:
            results.append((extract_output(output.outputs[0].text)))
            logging.info(results[-1])
            logging.info(instance['paper_id'])
        

    assert len(results) == len(dataset)

    for i, output in enumerate(results):
        
        output = output.replace('\n', '')
        output = output.replace('```json', '')
        output = output.replace('```', '')

        if '<FAILED>' in output:
            continue

        try:
            output = output.replace("'", '"')
            output = json.loads(output)
        except Exception:
            logging.info("parsing error, skipping...")
            continue

        try:
            output['Metric_Value'] = float(output['Metric_Value'])
        except Exception:
            logging.info("format error, skipping...")
            continue

        adjusted_values.append(output['Metric_Value'])
        standardized_metric_names.append(output['Metric_Name'])
        filtered_dataset.append(dataset[i])

    logging.info(f"Filtered Dataset Length of Records with Valid Metrics: {len(filtered_dataset)}")

    keys = filtered_dataset[0].keys()
    huggingface_dict = {}
    for k in keys:
        huggingface_dict[k] = [fildered_domain_instance[k] for fildered_domain_instance in filtered_dataset]
    filtered_dataset = Dataset.from_dict(huggingface_dict)
    filtered_dataset = filtered_dataset.add_column("adjusted_metric_value", adjusted_values)
    filtered_dataset = filtered_dataset.add_column("standardized_metric", standardized_metric_names)
    
    return filtered_dataset


def main(args):

    logging.info(f"Classifying whether the record contains the valid metrics")

    shard_dirs = sorted([
        os.path.join(args.hf_ds_path, d)
        for d in os.listdir(args.hf_ds_path)
        if os.path.isdir(os.path.join(args.hf_ds_path, d)) and d.startswith("shard_")
    ])

    dataset = concatenate_datasets([load_from_disk(shard) for shard in shard_dirs])

    logging.info(f"Current Dataset Length: {len(dataset)}")
    logging.info(f"Processing shard idx {args.shard_idx+1} out of {args.total_shards} shards")
    logging.info(f"Filtered outputs will be saved in {args.hf_ds_output_path}")

    logging.info("Sharding dataset...")
    dataset = dataset.shard(num_shards=args.total_shards, index=args.shard_idx)
    logging.info(f"Shard size: {len(dataset)}")

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

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
    logging.info("Pipeline loaded")


    filtered_dataset = filter_metric(dataset, prompt_template, pipeline, sampling_params, tokenizer)    
    filtered_dataset.save_to_disk(args.hf_ds_output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/filter_metric.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()
    
    logging.info(f"Start Filtering Metrics")

    main(args)
    logging.info("Finished Filtering")
