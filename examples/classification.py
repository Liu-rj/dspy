import random
from typing import Literal
import dspy
from dspy.datasets import DataLoader
from datasets import load_dataset

# Load the Banking77 dataset.
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

# Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
]
random.Random(0).shuffle(trainset)

dspy.configure(lm=dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'))

# Define the DSPy module for classification. It will use the hint at training time, if available.
signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThoughtWithHint(signature)

# Optimize via BootstrapFinetune.
# optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
optimizer = dspy.MIPROv2(metric=(lambda x, y, trace=None: x.label == y.label), auto="light", num_threads=24)
optimized_classifier = optimizer.compile(classify, trainset=trainset)

# Save the optimized classifier.
optimized_classifier.save("checkpoints/classifier.json")

res = classify(text="What does a pending cash withdrawal mean?")
print(res.label)
print(res.reasoning)

res = optimized_classifier(text="What does a pending cash withdrawal mean?")
print(res.label)
print(res.reasoning)
