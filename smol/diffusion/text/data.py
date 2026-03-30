# scripts/inspect_wikitext2.py

from datasets import load_dataset


def main(split="train", num_examples=10):
    # Load raw WikiText-2 (character-friendly)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    print(f"Loaded split='{split}' with {len(dataset)} examples\n")

    count = 0
    for i, example in enumerate(dataset):
        text = example["text"]

        # Skip empty lines (WikiText has many)
        if not text.strip():
            continue

        print("=" * 80)
        print(f"Example {i}")
        print("-" * 80)
        print(text)

        count += 1
        if count >= num_examples:
            break


if __name__ == "__main__":
    main()