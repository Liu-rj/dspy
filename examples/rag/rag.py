import dspy
import ujson
import random
import time
from dspy.evaluate import SemanticF1
from dspy.utils import download
from sentence_transformers import SentenceTransformer

lm = dspy.LM("bedrock/meta.llama3-1-8b-instruct-v1:0")
# lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
dspy.configure(lm=lm)

qa = dspy.Predict("question: str -> response: str")
response = qa(question="what are high memory and low memory on linux?")
print(response.response)

# dspy.inspect_history(n=1)

# cot = dspy.ChainOfThought("question -> response")
# response = cot(question="should curly braces appear on their own line?")
# print(response.response)

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download(
    "https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl"
)

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

# Inspect one datapoint.
print(data[0])

data = [dspy.Example(**d).with_inputs("question") for d in data]

# Let's pick an `example` here from the data.
example = data[2]
print(example)

# Split the data into train, dev, and test sets.
random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]
print(len(trainset), len(devset), len(testset))


# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
# pred = cot(**example.inputs())

# Compute the metric score for the prediction.
# score = metric(example, pred)

# print(f"Question: \t {example.question}\n")
# print(f"Gold Response: \t {example.response}\n")
# print(f"Predicted Response: \t {pred.response}\n")
# print(f"Semantic F1 Score: {score:.2f}")

# print(dspy.inspect_history(n=1))

# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(
    devset=devset, metric=metric, num_threads=12, display_progress=True, display_table=2
)

# Evaluate the Chain-of-Thought program.
# print(evaluate(cot))

# Download the full RAG-QA Arena "Tech" corpus.
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)["text"][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embedder = dspy.Embedder(model.encode, batch_size=32)
search = dspy.retrievers.Embeddings(
    embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve
)


class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)


rag = RAG()
rag(question="what are high memory and low memory on linux?")

dspy.inspect_history()

score = evaluate(RAG())
print(f"Semantic F1 Score Before Optimization: {score:.2f}")

tp = dspy.MIPROv2(
    metric=metric, auto="medium", num_threads=12
)  # use fewer threads if your rate limit is small

tic = time.time()
optimized_rag, time_tuple = tp.compile(
    RAG(),
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    requires_permission_to_run=False,
)
compile_time = time.time() - tic
print(f"Bootstrap demo time: {time_tuple[0]:.2f}s")
print(f"Propose instruction time: {time_tuple[1]:.2f}s")
print(f"Bayesian optimization time: {time_tuple[2]:.2f}s")
print(f"Total compile time: {compile_time:.2f}s")

# save the optimized RAG
optimized_rag.save("optimized_rag.json")

baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print("#" * 80)
print("Baseline response:")
print(baseline.response)
print("#" * 80)

pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print("#" * 80)
print("Optimized response:")
print(pred.response)
print("#" * 80)

score_optimized = evaluate(optimized_rag)
print(f"Semantic F1 Score After Optimization: {score_optimized:.2f}")

# cost in USD, as calculated by LiteLLM for certain providers
cost = sum([x["cost"] for x in lm.history if x["cost"] is not None])

# load the optimized RAG
loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")

print(loaded_rag(question="cmd+tab does not work on hidden or minimized windows"))

print(f"Bootstrap demo time: {time_tuple[0]:.2f}s")
print(f"Propose instruction time: {time_tuple[1]:.2f}s")
print(f"Bayesian optimization time: {time_tuple[2]:.2f}s")
print(f"Total compile time: {compile_time:.2f}s")
print(f"Semantic F1 Score Before Optimization: {score:.2f}")
print(f"Semantic F1 Score After Optimization: {score_optimized:.2f}")
