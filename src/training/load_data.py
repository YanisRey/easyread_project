from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="../../data/training_data/images", split="train")

print(dataset[0])
