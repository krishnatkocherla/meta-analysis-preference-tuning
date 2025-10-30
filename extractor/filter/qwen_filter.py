import os
from qwen_agent.agents import Assistant

# ---------------------------------------
# LLM Configuration
# ---------------------------------------
llm_cfg = {
    # Use your running vLLM model
    'model': 'Qwen/Qwen3-32B',
    'model_server': 'http://localhost:8000/v1',  # Your vLLM server base URL
}

# ---------------------------------------
# System Instruction
# ---------------------------------------
system_instruction = """Filter each research paper based on the following criteria:
1. The paper must be about multimodal learning or cross-modal representation.
2. It should involve neural architectures (e.g., transformers, CNNs, or LSTMs).
3. The study must include empirical experiments, not purely theoretical analysis.
4. The dataset used must be open-source or publicly available.
5. If uncertain, mark as "Unclear".
"""

# ---------------------------------------
# Input PDFs
# ---------------------------------------
pdf_dir = "/nethome/kkocherla3/flash/downloaded_pdfs"  # safer: absolute path
if not os.path.exists(pdf_dir):
    raise FileNotFoundError(f"Directory not found: {pdf_dir}")

files = [
    os.path.join(pdf_dir, f)
    for f in os.listdir(pdf_dir)
    if f.lower().endswith(".pdf")
]

if not files:
    raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

print(f"Found {len(files)} PDF(s) to process.\n")

# ---------------------------------------
# Query Setup
# ---------------------------------------
query = "Classify whether this paper meets the criteria. Respond with 'Yes', 'No', or 'Unclear' and a short justification."

# ---------------------------------------
# Run Assistant
# ---------------------------------------
for file_path in files:
    print(f"Processing: {file_path}")

    # Create Assistant instance for each file
    bot = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        function_list=[],
        files=[file_path],
    )

    # Run query
    messages = [{'role': 'user', 'content': query}]
    try:
        for response in bot.run(messages=messages):
            print(f"\n--- Output for {os.path.basename(file_path)} ---")
            print(response)
            print("-----------------------------------\n")
    except Exception as e:
        print(f"Error processing {file_path}: {e}\n")