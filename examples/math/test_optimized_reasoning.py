import dspy
import time
from dspy.datasets import MATH

llama_lm = dspy.LM("bedrock/meta.llama3-1-8b-instruct-v1:0")
claud_lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
dspy.configure(lm=llama_lm)

dataset = MATH(subset='algebra')
print(len(dataset.train), len(dataset.dev))

example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)

module = dspy.ChainOfThought("question -> answer")
print(module(question=example.question))
print(dspy.inspect_history())

THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

res = evaluate(module)
print(f"Score before optimization: {res:.2f}")

module.load("optimized_math_reasoning.json")

res = evaluate(module)
print(f"Score after optimization: {res:.2f}")
print(dspy.inspect_history())
