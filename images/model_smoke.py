"""
Smoke test: run google/gemma-4-E2B-it on the first 4 entries of Wikitext-2.
"""

import textwrap
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
NUM_ENTRIES = 4
MAX_NEW_TOKENS = 30


def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError:
        print("WARNING: CUDA device unavailable, falling back to CPU.")
        return "cpu"


def main() -> None:
    device = _pick_device()
    print(f"Loading tokenizer and model: {MODEL_ID}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    print("Loading Wikitext-2 (test split) …")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    non_empty = [row["text"] for row in dataset if row["text"].strip()]
    entries = non_empty[:NUM_ENTRIES]

    for i, text in enumerate(entries, 1):
        prompt = text[:200]  # truncate long passages to keep prompts manageable
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)

        print(f"\n--- Entry {i} ---")
        print("Prompt  :", textwrap.shorten(prompt, width=80))
        print("Continuation:", textwrap.shorten(generated, width=80))

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
