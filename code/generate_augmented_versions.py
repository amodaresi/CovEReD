from datasets import load_dataset
import sys, pickle
import os

from itertools import product
import json
from utils.contr.src.contriever import Contriever
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
from copy import deepcopy
from random import sample, shuffle

SPLIT = "train"
DATA_FILEPATH = "[FILEPATH]/"
AUGMENTED_OUTPUT_FP = f"[FILEPATH]/cf_{SPLIT}/"
CACHE_DIR = "[CACHE_DIR]"
N_CPUs = 12
MAX_ALTERNATIVES = 3
NO_of_SETs = 10
THR = 0.7

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ds = load_dataset("json", data_files=os.path.join(DATA_FILEPATH, f"{SPLIT}_entFix.json"), keep_in_memory=True)["train"]
print(ds)

# identifying relations of each vertex
relation_maps = []
for i in range(len(ds)):
    relation_map = {v: set() for v in range(len(ds[i]["vertexSet"]))}
    for relation in ds[i]["labels"]:
        relation_map[relation['h']].add(('h', relation['r']))
        relation_map[relation['t']].add(('t', relation['r']))
    relation_maps.append(relation_map)

# adding extra metadata for each vertex
all_vertices = []
CONTEXT_WINDOW = 16
for i in range(len(ds)):
    vertexSet = ds[i]["vertexSet"]
    all_tokens = [s for sent in ds[i]["sents"] for s in sent]
    for vertex_id, vertex in enumerate(vertexSet):

        newVertex = {
            "aliases": set([s["name"] for s in vertex]),
            "contexts": set([" ".join(all_tokens[max(s["global_pos"][0]-CONTEXT_WINDOW,0):min(s["global_pos"][0]+s["pos"][1]-s["pos"][0]+CONTEXT_WINDOW, len(all_tokens))]) for s in vertex]),
            "idx": len(all_vertices),
            "doc_id": i,
            "vertex_id": vertex_id,
            "type": set([s["type"] for s in vertex]),
            "relation_map": relation_maps[i][vertex_id]
        }
        all_vertices.append(newVertex)

contriever = Contriever.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR).to(device)
contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco", cache_dir=CACHE_DIR) 

def get_representation(phrases, batch_size=48, pbar=False):
    steps = int(np.ceil(len(phrases) * 1.0 / batch_size))
    all_embs = []
    for i in range(steps):
        inputs = contriever_tokenizer(phrases[i*batch_size:min((i+1)*batch_size, len(phrases))], padding=True, truncation=True, return_tensors="pt").to(device)
        all_embs.append(contriever(**inputs).cpu().detach().numpy())
    return np.concatenate(all_embs, axis=0) if len(all_embs) > 0 else []

# Get all reps of aliases
all_aliases = [alias for vertex in all_vertices for alias in vertex["aliases"]]
all_reps = get_representation(all_aliases)
alias_to_rep = {k: all_reps[i] for i, k in enumerate(all_aliases)}

# Get all reps of contexts
all_contexts = [context for vertex in all_vertices for context in vertex["contexts"]]
all_reps = get_representation(all_contexts)
context_to_rep = {k: all_reps[i] for i, k in enumerate(all_contexts)}


def sim_function_batched(x, y):
        x = np.array(x)
        y = np.array(y)
        return (x.dot(y.T)) / np.sqrt((x * x).sum(axis=-1).reshape((-1,1)) * (y * y).sum(axis=-1).reshape((1,-1)))


def get_alternatives(candidate, SELFSIM=0.2, CONTEXTSIM=0.4, SELFSIM_UPPER=0.8):
    alternatives = []
    for vertex in all_vertices:
        if vertex["doc_id"] == candidate["doc_id"]:
            continue
        if len(vertex["aliases"].intersection(candidate["aliases"])) > 0:
            continue
        if len(vertex["type"].intersection(candidate["type"])) == 0:
            continue
        rel_sim_count = len(candidate["relation_map"].intersection(vertex["relation_map"]))
        if rel_sim_count == 0:
            continue
        sim = sim_function_batched([alias_to_rep[alias] for alias in vertex["aliases"]], [alias_to_rep[alias] for alias in candidate["aliases"]]).max()
        if sim < SELFSIM or SELFSIM_UPPER < sim:
            continue
        vertex["sim"] = sim
        sim = sim_function_batched([context_to_rep[context] for context in vertex["contexts"]], [context_to_rep[context] for context in candidate["contexts"]]).max()
        if sim < CONTEXTSIM:
            continue
        vertex["rel_sim_count"] = rel_sim_count
        vertex["context_sim"] = sim
        alternatives.append(vertex)
    return alternatives

def apply_replacement(example, vertex_id, replacementSet):
    aliases = [v["name"] for v in example["vertexSet"][vertex_id]]
    aliases_reps = [alias_to_rep[alias] for alias in aliases]

    # Check if its similar to other aliases
    other_aliases = []
    for v_id, vertex in enumerate(example["vertexSet"]):
        if v_id == vertex_id:
            continue
        other_aliases.extend([v["name"] for v in vertex])

    replacementSet = list(replacementSet)
    sim = sim_function_batched([alias_to_rep[alias] for alias in other_aliases], [alias_to_rep[alias] for alias in replacementSet]).max()
    if sim > 0.5:
        return None

    # Find matching aliases
    replacementSet_reps = [alias_to_rep[alias] for alias in replacementSet]
    sim = sim_function_batched(aliases_reps, replacementSet_reps)

    new_example = deepcopy(example)
    for a_id, alias in enumerate(new_example["vertexSet"][vertex_id]):
        replacement = replacementSet[np.argmax(sim[a_id])]
        replacement_tokenized = replacement.split()
        shift = len(replacement_tokenized) - (alias["pos"][1] - alias["pos"][0])
        shift_point = alias["global_pos"][0]
        alias["name"] = replacement
        new_example["vertexSet"][vertex_id][a_id] = alias
        new_example["sents"][alias["sent_id"]][alias["pos"][0]:alias["pos"][1]] = replacement_tokenized

        for v_id, _ in enumerate(new_example["vertexSet"]):
            for a_id_2, _ in enumerate(new_example["vertexSet"][v_id]):
                if shift_point < new_example["vertexSet"][v_id][a_id_2]["global_pos"][0]:
                    new_example["vertexSet"][v_id][a_id_2]["global_pos"][0] += shift
                    new_example["vertexSet"][v_id][a_id_2]["global_pos"][1] += shift
                    assert new_example["vertexSet"][v_id][a_id_2]["global_pos"][0] >= 0
                    if new_example["vertexSet"][v_id][a_id_2]["sent_id"] == alias["sent_id"]:
                        new_example["vertexSet"][v_id][a_id_2]["pos"][0] += shift
                        new_example["vertexSet"][v_id][a_id_2]["pos"][1] += shift
                        assert new_example["vertexSet"][v_id][a_id_2]["pos"][0] >= 0
                elif a_id == a_id_2 and v_id == vertex_id:
                    new_example["vertexSet"][v_id][a_id_2]["pos"][1] += shift

    return new_example


def get_topK_args(x, K=10):
    return np.flip(np.argsort(x))[:K + min(len(x) - K - np.sum(x == 0), 0)]


def generate_augmented(doc_id, THR=THR):
    example = ds[doc_id]
    relation_mask = np.zeros((len(example["vertexSet"]), len(example["labels"])))
    for rel_idx, relation in enumerate(example["labels"]):
        relation_mask[relation["h"]][rel_idx] = 1
        relation_mask[relation["t"]][rel_idx] = 1

    vertexSet = [v for v in all_vertices if v["doc_id"] == doc_id]
    alternative_per_vertex = {v: [] for v in range(len(vertexSet))}
    for vertex_id in alternative_per_vertex.keys():
        if np.sum(relation_mask[vertex_id]) == 0:
            continue
        alternatives = sorted(get_alternatives(vertexSet[vertex_id]), key=lambda x: (x["rel_sim_count"], x["sim"], x["context_sim"]), reverse=True)[:15]
        for alternative in alternatives:
            if alternative["aliases"] not in alternative_per_vertex[vertex_id]:
                alternative_per_vertex[vertex_id].append(alternative["aliases"])

    for k in alternative_per_vertex.keys():
        alternatives = alternative_per_vertex[k]

        filtered_sets = []
        # Loop over each set in the original list
        for i, set_a in enumerate(alternatives):
            # Check if there exists another set of which set_a is a subset
            if not any(set_a < set_b for j, set_b in enumerate(alternatives) if i != j):
                filtered_sets.append(set_a)

        alternative_per_vertex[k] = filtered_sets

    altered_examples = {}
    number_of_relations_edited = {}
    searched_branches = []

    relation_count = relation_mask.sum(axis=-1)
    maxAlternatives = MAX_ALTERNATIVES

    for vertex_id in get_topK_args(relation_count):
        searched_branches.append({vertex_id})
        for alternative in alternative_per_vertex[vertex_id]:
            new_example = apply_replacement(example, vertex_id, alternative)
            if new_example is not None:
                if (vertex_id,) not in altered_examples:
                    altered_examples[(vertex_id,)] = [new_example]
                    number_of_relations_edited[(vertex_id,)] = relation_count[vertex_id] / relation_count.sum()
                elif len(altered_examples[(vertex_id,)]) < maxAlternatives:
                    altered_examples[(vertex_id,)].append(new_example)

    for j in range(len(searched_branches) - 1):
        children_altered_examples = {}
        for parent_id in altered_examples.keys(): 
            for vertex_id in get_topK_args(relation_count):
                child_id = parent_id + (vertex_id,)
                if vertex_id in parent_id:
                    continue
                if set(child_id) in searched_branches:
                    continue
                else:
                    searched_branches.append(set(child_id))
                for parent_example in altered_examples[parent_id]:
                    for alternative in alternative_per_vertex[vertex_id]:
                        new_example = apply_replacement(parent_example, vertex_id, alternative)
                        if new_example is not None:
                            if child_id not in children_altered_examples:
                                children_altered_examples[child_id] = [new_example]
                                number_of_relations_edited[child_id] = number_of_relations_edited[parent_id] + relation_count[vertex_id] / relation_count.sum()
                            else:
                                children_altered_examples[child_id].append(new_example)

        for k, v in children_altered_examples.items():
            shuffle(v)
            altered_examples[k] = v[:maxAlternatives]

    output_examples = []
    for k, v in altered_examples.items():
        if number_of_relations_edited[k] > THR:
            output_examples.extend(v)

    return output_examples


output_examples = p_map(generate_augmented, [i for i in range(len(ds))], num_cpus=N_CPUs)
# output_examples = [generate_augmented(i) for i in range(871, len(ds))]
for i in tqdm(range(len(output_examples))):
    shuffle(output_examples[i])

for i in tqdm(range(NO_of_SETs)):
    bucket = []
    for j in range(len(ds)):
        if i < len(output_examples[j]):
            bucket.append(output_examples[j][i])
            bucket[-1]["original_doc_id"] = j
    
    print(len(bucket))
    with open(os.path.join(AUGMENTED_OUTPUT_FP,f"{SPLIT}_sample_bucket_{i+1}.json"), "w") as f:
        json.dump(bucket, f)