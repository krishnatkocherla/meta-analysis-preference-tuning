import os
import logging
import argparse

from copy import deepcopy
from tqdm import tqdm
from datasets import load_from_disk

import torch
import transformers
from datasets import Dataset, load_from_disk
from datasets import concatenate_datasets, load_dataset


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


def main(args):
    # shard_dirs = sorted([
    #     os.path.join(args.hf_ds_path, d)
    #     for d in os.listdir(args.hf_ds_path)
    #     if os.path.isdir(os.path.join(args.hf_ds_path, d)) and d.startswith("shard_")
    # ])

    # # Load and concatenate all shards
    # dataset = concatenate_datasets([load_from_disk(shard) for shard in shard_dirs])
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

    dataset_subset_concat = []
    for i in range(len(dataset_names)):
        if subsets[i] == "xx":
            dataset_subset_concat.append(dataset_names[i])
        else:
            dataset_subset_concat.append(dataset_names[i] + "<|SEP|>" + subsets[i])
    
    unique_datsets = list(set(dataset_subset_concat))
    
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
    for i, instance in enumerate(unique_datsets):
        prompt = deepcopy(prompt_template)

        if "<|SEP|>" not in instance:
            prompt = prompt.replace("{{dataset}}", instance)
            prompt = prompt.replace("{{subset}}", "xx")
        else:
            dataset_name, subset = instance.split("<|SEP|>")
            prompt = prompt.replace("{{dataset}}", dataset_name)
            prompt = prompt.replace("{{subset}}", subset)
        message = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        text = truncate_length(text, tokenizer)
        outputs = pipeline.generate([text], sampling_params)
        for output in outputs:
            results.append(extract_output(output.outputs[0].text))
    
    unique_dataset_to_description = {}
    for i in range(len(unique_datsets)):
        unique_dataset_to_description[unique_datsets[i]] = results[i]

    dataset_descriptions = []
    description_source = []
    m = 0
    for dataset_subset in dataset_subset_concat:
        description = unique_dataset_to_description[dataset_subset]
        dataset_descriptions.append(description)
        if '<FAILED>' in description:
            m += 1
            description_source.append('<FAILED>')
        else:
            description_source.append('GPT4o')
    
    logging.info(f"Number of failed dataset descriptions: {m} out of {len(dataset_descriptions)}")

    dataset = dataset.add_column("dataset_description", dataset_descriptions)    
    dataset = dataset.add_column("description_source", description_source)
    
    logging.info(f"Saving {len(dataset)} dataset tables' descriptions")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Generation Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default='./extractor/generate_description/prompt/generate_description.txt')
    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()
    
    logging.info(f"Start Generating Datset Descriptions")

    main(args)
