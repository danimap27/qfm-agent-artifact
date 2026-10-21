"""Query arXiv for candidate references and print verifiable metadata."""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

QUERIES = [
    "Faithfulness Tests for Natural Language Explanations",
    "Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals",
    "Do Models Explain Themselves Counterfactual Simulatability of Natural Language Explanations",
    "Diagnosing AI Explanation Methods with Folk Concepts of Behavior",
    "e-SNLI Natural Language Inference Explanations",
    "Teach Me to Explain A Review of Datasets for Explainable NLP",
    "Explainable Artificial Intelligence XAI Concepts taxonomies opportunities and challenges toward responsible AI",
    "Explanation in artificial intelligence Insights from the social sciences",
    "The Mythos of Model Interpretability",
    "Why Should I Trust You Explaining the Predictions of Any Classifier",
    "A Unified Approach to Interpreting Model Predictions",
    "Quantum computing with Qiskit",
    "Explainable quantum machine learning",
]

NS = {"a": "http://www.w3.org/2005/Atom"}


def search(title: str) -> list[dict]:
    query = f'ti:"{title}"'
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": query, "max_results": 3})
    with urllib.request.urlopen(url, timeout=30) as response:
        root = ET.fromstring(response.read())
    results = []
    for entry in root.findall("a:entry", NS):
        results.append(
            {
                "title": " ".join(entry.find("a:title", NS).text.split()),
                "authors": [a.find("a:name", NS).text for a in entry.findall("a:author", NS)],
                "published": entry.find("a:published", NS).text[:10],
                "id": entry.find("a:id", NS).text.rsplit("/", 1)[-1],
            }
        )
    return results


for query in QUERIES:
    print(f"\n## {query}")
    try:
        for item in search(query):
            print(f"  {item['id']}  {item['published']}  {item['title'][:95]}")
            print(f"     {', '.join(item['authors'][:4])}{' et al.' if len(item['authors']) > 4 else ''}")
    except Exception as error:  # noqa: BLE001
        print(f"  ERROR {error}", file=sys.stderr)
