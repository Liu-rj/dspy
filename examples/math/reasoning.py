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

THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

res = evaluate(module)
print(f"Score before optimization: {res:.2f}")

kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=claud_lm), prompt_model=llama_lm)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

tic = time.time()
kwargs = dict(requires_permission_to_run=False, max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module, time_tuple = optimizer.compile(module, trainset=dataset.train, **kwargs)
compile_time = time.time() - tic

print(f"Bootstrap demo time: {time_tuple[0]:.2f}s")
print(f"Propose instruction time: {time_tuple[1]:.2f}s")
print(f"Bayesian optimization time: {time_tuple[2]:.2f}s")
print(f"Total compile time: {compile_time:.2f}s")

optimized_module.save("optimized_math_reasoning.json")

res = evaluate(optimized_module)
print(f"Score after optimization: {res:.2f}")
# print(dspy.inspect_history())
