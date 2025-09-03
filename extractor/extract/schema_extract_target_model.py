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

def extract_output(qwen_output):
    # Extract text after </think>
    if qwen_output.count("<think>")  == 1 and qwen_output.count("</think>") == 0:
        return "<FAILED>"
    after_think = qwen_output.split("</think>", 1)[-1].strip()
    return after_think



def main(args):
    logging.info("Loading dataset...")

    shard_dirs = sorted([
        os.path.join(args.hf_ds_path, d)
        for d in os.listdir(args.hf_ds_path)
        if os.path.isdir(os.path.join(args.hf_ds_path, d)) and d.startswith("shard_")
    ])

    # Load and concatenate all shards
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

    logging.info("Pipeline loaded, beginning extraction")

    with open(args.prompt_path, 'r') as f:
        template_prompt = f.read().strip()
    results = []
    promptlist = []
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32000, logits_processors=[processor])
    for instance in tqdm(dataset):
        prompt = deepcopy(template_prompt)
        prompt = prompt.replace("{{table_code}}", instance['table_source'])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        outputs = pipeline.generate([text], sampling_params)
        for output in outputs:
            results.append(extract_output(output.outputs[0].text))
            logging.info(extract_output(output.outputs[0].text))
            logging.info(instance['paper_id'])

    n = 0
    idx = []
    for i, result in enumerate(results):
        if "<FAILED>" in result:
            n += 1
            idx.append(i)
    logging.info(f"{n} Failed out of {len(results)} Tables")

    dataset = dataset.add_column("table_results_extracted", results)    
    dataset = dataset.select([i for i in range(len(dataset)) if i not in idx])

    logging.info(f"saving {len(dataset)} Tables")
    dataset.save_to_disk(args.hf_ds_output_path)

    logging.info(f"Extraction Finished")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt_path', type=str, default='./extractor/extract/prompt/schema_extract_target_model.txt')

    parser.add_argument('--hf_ds_path', type=str)
    parser.add_argument('--hf_ds_output_path', type=str)
    
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--shard_idx', type=int, default=0)
    parser.add_argument('--total_shards', type=int, default=1)

    args = parser.parse_args()
    
    logging.info(f"Start Extracting Tables of Target Model Experiments")

    main(args)
