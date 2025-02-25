import dspy
import random
import requests
from dspy.datasets import DataLoader

lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
dspy.configure(lm=lm)

kwargs = dict(
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",)
)
hover = DataLoader().from_huggingface(
    dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs
)

hpqa_ids = set()
hover = [
    dspy.Example(
        claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))
    ).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3
    and x["hpqa_id"] not in hpqa_ids
    and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]

# print example
example = trainset[0]
print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

DOCS = {}


def search_wikipedia(query: str) -> str:
    """Search wikipedia by query and return the titles and snippets of the search results."""
    search_url = f"https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}

    # Make the request to Wikipedia API
    response = requests.get(search_url, params=params)

    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve search results"

    # Parse the JSON response
    data = response.json()
    search_results = data.get("query", {}).get("search", [])

    # Extract titles and snippets of the search results
    results = []
    for result in search_results:
        title = result.get("title")
        snippet = result.get("snippet")
        results.append(f"Title: {title}\nSnippet: {snippet}\n")

    return "\n".join(results)


def get_wikipedia_page_content(title: str) -> str:
    """Get the content of a Wikipedia page by title."""
    page_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }

    # Make the request to Wikipedia API
    response = requests.get(page_url, params=params)

    # Check if the request was successful
    if response.status_code != 200:
        return "Failed to retrieve page content"

    # Parse the JSON response
    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    # Extract the content of the page
    for page_id, page in pages.items():
        content = page.get("extract", "No content found")
        return content

    return "No content found"


instructions = (
    "Find all Wikipedia titles relevant to verifying (or refuting) the claim."
)
signature = dspy.Signature("claim -> titles: list[str]", instructions)
react = dspy.ReAct(
    signature, tools=[search_wikipedia, get_wikipedia_page_content], max_iters=20
)

# res = react(claim=example.claim)
res = react(claim="The actor who played captain hook in the live production with Taylor Louderman won the academy award for his role in The Deer Hunter.")
print(res.titles)
# print(dspy.inspect_history(n=2))
exit()


def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0

    # If we're just doing inference, just measure the recall.
    return recall


def safe_react(claim: str):
    try:
        return react(claim=claim)
    except Exception as e:
        return dspy.Prediction(titles=[])


evaluate = dspy.Evaluate(
    devset=devset,
    metric=top5_recall,
    num_threads=16,
    display_progress=True,
    display_table=5,
)

score = evaluate(safe_react)
print(f"Top-5 Recall Before Optimization: {score:.2f}")

optimized_react = dspy.ReAct(
    "claim -> titles: list[str]",
    tools=[search_wikipedia, lookup_wikipedia],
    max_iters=20,
)
optimized_react.load("optimized_react.json")

res = optimized_react(claim=example.claim)
print(res.titles)
print(dspy.inspect_history(n=2))

score_optimized = evaluate(optimized_react)
print(f"Top-5 Recall After Optimization: {score_optimized:.2f}")
