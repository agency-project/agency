"""Deterministic local tasks; gold data remains in the host evaluator."""

from __future__ import annotations

import json
import random

CORPUS = {
    "atlas-2024": {
        "text": "Atlas owner: Platform. Retention: 30 days. Version 2024; superseded.",
        "version": 2024,
    },
    "atlas-2026": {
        "text": "Atlas owner: Data. Retention: 90 days. Current version 2026. Escalations use the owner roster.",
        "version": 2026,
    },
    "roster": {
        "text": "Data escalation: Mira. Platform escalation: Leon. Security escalation: Sam.",
        "version": 2026,
    },
    "beacon": {
        "text": "Beacon owner: Security. Retention: 7 days. Region: eu-west.",
        "version": 2026,
    },
    "regions": {
        "text": "Atlas region: us-east. eu-west failover: eu-central. us-east failover: us-west.",
        "version": 2026,
    },
    "distractor": {
        "text": "Atlas marketing launched in 2023. Beacon is also a historic lighthouse name.",
        "version": 2023,
    },
}

RAG = [
    (
        "atlas-current",
        "Who currently owns Atlas and how long is its retention?",
        {"owner": "Data", "retention_days": 90},
        ["atlas-2026"],
    ),
    (
        "atlas-escalation",
        "Who handles current Atlas escalations?",
        {"contact": "Mira"},
        ["atlas-2026", "roster"],
    ),
    (
        "beacon-escalation",
        "Who handles Beacon escalations and what is its retention?",
        {"contact": "Sam", "retention_days": 7},
        ["beacon", "roster"],
    ),
    ("atlas-failover", "Where does Atlas fail over?", {"region": "us-west"}, ["regions"]),
    (
        "beacon-failover",
        "Where does Beacon fail over?",
        {"region": "eu-central"},
        ["beacon", "regions"],
    ),
]

# Each C program reads one bounded integer and prints a defined integer result.
# Domains prevent overflow, out-of-bounds access, invalid shifts and UB.
C_CASES = {
    "parsing": (
        'int f(int x) { char s[24]; snprintf(s, sizeof(s), "%d", x); return (int)strtol(s, NULL, 10); }',
        -9999,
        9999,
        "decimal parsing",
    ),
    "buffers": (
        "int f(int x) { unsigned char b[8] = {0}; int n=x%8; for(int i=0;i<n;i++) b[i]=(unsigned char)(i+1); int s=0; for(int i=0;i<8;i++) s+=b[i]; return s; }",
        0,
        255,
        "bounded buffer fill",
    ),
    "state": (
        "struct S { int value; int count; }; int f(int x) { struct S s={0,0}; for(int i=0;i<x;i++){ s.value+=i; s.count++; } return s.value+s.count; }",
        0,
        100,
        "struct accumulator state",
    ),
    "errors": (
        "int f(int x) { if(x<0) return -1; if(x>100) return -2; return x*2; }",
        -100,
        200,
        "error codes",
    ),
    "bitfields": (
        "int f(int x) { unsigned int u=(unsigned int)x; return (int)((u & 15u) + ((u >> 4u) & 15u)); }",
        0,
        65535,
        "unsigned field extraction",
    ),
}


def local_tasks(suite):
    if suite == "rag":
        return [
            {
                "id": name,
                "stratum": "stale" if name.startswith("atlas") else "multihop",
                "question": question,
                "gold": gold,
                "answer_fields": {
                    key: "integer" if type(value) is int else "string"
                    for key, value in gold.items()
                },
                "evidence": evidence,
                "corpus": CORPUS,
            }
            for name, question, gold, evidence in RAG
        ]
    if suite == "migration":
        tasks = []
        for name, (function, low, high, contract) in C_CASES.items():
            source = "#include <stdio.h>\n#include <stdlib.h>\n" + function
            source += '\nint main(void) { int x; if(scanf("%d", &x)!=1) return 2; printf("%d\\n",f(x)); return 0; }\n'
            tasks.append({"id": name, "stratum": contract, "source": source, "domain": [low, high]})
        return tasks
    if suite == "tandem":
        return [
            {
                "id": f"ledger-{i}",
                "stratum": "independent-ledgers",
                "left": [i, i + 2, i + 4],
                "right": [i + 1, i + 3, i + 5],
            }
            for i in range(5)
        ]
    raise ValueError(suite)


class CorpusTools:
    def __init__(self, corpus):
        self.corpus = corpus

    def retrieve(self, arguments):
        query = arguments["query"].lower().split()
        scored = []
        for doc_id, document in self.corpus.items():
            score = sum(word in document["text"].lower() for word in query)
            if score:
                scored.append((score, doc_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return {"document_ids": [doc_id for _, doc_id in scored[: arguments.get("limit", 5)]]}

    def read(self, arguments):
        doc_id = arguments["document_id"]
        return {"document_id": doc_id, **self.corpus[doc_id]}

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": "Search local documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": "Read a local document.",
                    "parameters": {
                        "type": "object",
                        "properties": {"document_id": {"type": "string"}},
                        "required": ["document_id"],
                    },
                },
            },
        ]


def rag_prompt(task):
    fields = task.get("answer_fields") or {
        key: "integer" if type(value) is int else "string" for key, value in task["gold"].items()
    }
    return (
        task["question"]
        + "\nAnswer fields and types: "
        + json.dumps(fields)
        + "\nUse exactly these field names and canonical values (for a person, the name only). "
        "Submit answer as an object and evidence as a document ID array."
    )


def evaluate_rag(task, text):
    try:
        answer = json.loads(text)
    except (TypeError, ValueError):
        return {
            "success": False,
            "answer_correct": False,
            "evidence_supported": False,
            "failure": "task",
        }
    if not isinstance(answer, dict):
        return {
            "success": False,
            "answer_correct": False,
            "evidence_supported": False,
            "failure": "task",
        }
    actual = answer.get("answer")
    correct = isinstance(actual, dict) and actual == task["gold"]
    correct = correct and all(
        type(actual.get(key)) is type(value) for key, value in task["gold"].items()
    )
    evidence = answer.get("evidence", [])
    supported = isinstance(evidence, list) and all(isinstance(doc, str) for doc in evidence)
    supported = supported and set(task["evidence"]) <= set(evidence)
    supported = supported and all(doc in task["corpus"] for doc in evidence)
    return {
        "success": correct and supported,
        "answer_correct": correct,
        "evidence_supported": supported,
        "failure": None if correct and supported else "task",
    }


def differential_inputs(task, seed, count=100):
    low, high = task["domain"]
    randomizer = random.Random(seed)
    return [low, high, max(low, min(high, 0))] + [
        randomizer.randint(low, high) for _ in range(count)
    ]


def evaluate_tandem(task, text):
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        result = None
    expected = {
        "left": sum(task["left"]),
        "right": sum(task["right"]),
        "total": sum(task["left"]) + sum(task["right"]),
    }
    valid = (
        isinstance(result, dict)
        and result == expected
        and all(type(value) is int for value in result.values())
    )
    return {"success": valid, "failure": None if valid else "task"}
