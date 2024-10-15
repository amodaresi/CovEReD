# CovEReD &mdash; Counterfactual via Entity Replacement for DocRE
This repository contains the code and data associated with our paper, "[Consistent Document-Level Relation Extraction via Counterfactuals](https://www.arxiv.org/abs/2407.06699)", to be presented in the Findings of [EMNLP 2024](https://2024.emnlp.org/).

## Abstract
> Many datasets have been developed to train and evaluate document-level relation extraction (RE) models. Most of these are constructed using real-world data. It has been shown that RE models trained on real-world data suffer from factual biases. To evaluate and address this issue, we present **CovEReD**, **a counterfactual data generation approach for document-level relation extraction datasets using entity replacement**. We first demonstrate that models trained on factual data exhibit inconsistent behavior: while they accurately extract triples from factual data, they fail to extract the same triples after counterfactual modification. This inconsistency suggests that models trained on factual data rely on spurious signals such as specific entities and external knowledge&mdash;rather than on the input context&mdash;to extract triples. We show that by generating document-level counterfactual data with **CovEReD** and training models on them, consistency is maintained with minimal impact on RE performance. We release our **CovEReD** pipeline as well as **Re-DocRED-CF**, a dataset of counterfactual RE documents, to assist in evaluating and addressing inconsistency in document-level RE.

## Re-DocRED-CF
For training and evaluation, we have run CovEReD five times on all Re-DocRED splits. All five sets of train/dev/test dataset files are available through HuggingFace Datasets 🤗.
To select a specific variation (e.g. `var-01`):
```python
dataset = load_dataset("amodaresi/Re-DocRED-CF", "var-01")
```
#### Output:
```python
DatasetDict({
    train: Dataset({
        features: ['title', 'labels', 'original_doc_id', 'vertexSet', 'sents'],
        num_rows: 2870
    })
    dev: Dataset({
        features: ['title', 'labels', 'original_doc_id', 'vertexSet', 'sents'],
        num_rows: 466
    })
    test: Dataset({
        features: ['title', 'labels', 'original_doc_id', 'vertexSet', 'sents'],
        num_rows: 453
    })
    train_mix: Dataset({
        features: ['title', 'labels', 'original_doc_id', 'vertexSet', 'sents'],
        num_rows: 5923
    })
})
```
The `train_mix` is the original training set combined with its counterfactual variation counterpart.
We have also included four additional training set variations (var-[06, 07, 08, 09]), though they were not used in the evaluations presented in our paper.

The properties `title`, `labels`, `vertexSet`, and `sents` are structured similarly to those in the original DocRED & Re-DocRED datasets:

- `title`: Document title.
- `labels`: List of relations. Each entry indicates the relation between a head and a tail entity, with some entries also specifying evidence sentences.
- `vertexSet`: List of entity vertex sets. Each entry represents a vertex specifying all mentions of an entity by their position in the document, along with their type.
- `sents`: Tokenized sentences.

In examples that are counterfactually generated, the title includes a variation number. For example: `AirAsia Zest ### 1`.
The `original_doc_id` denotes the index of the example in the original seed dataset, i.e., Re-DocRED.

## Counterfactual Generation
To generate counterfactuals, you will first need to download the seed DocRE dataset.

For instance, for Re-DocRED:

`git clone https://github.com/tonytan48/Re-DocRED.git`

After downloading the dataset, use the `code/fix_overlapping_entities.ipynb` notebook to apply the entity mention cleanup process and store the output dataset.

Using the output dataset, run `code/generate_augmented_versions.py` to generate counterfactual sets.
For dev/test sets, use the `code/generate_augmented_versions_val.py` file, which also utilizes the training set to gather a larger entity replacement pool.
In the aforementioned files, you can edit the following hyperparameters:

- `N_CPUs`: The number of cores used for multiprocessing
- `MAX_ALTERNATIVES`: Maximum number of alternatives to sample from for each node
- `THR`: Affected relations threshold; an augmented document should have more than `THR` of its relations affected by the replacements
- `T_SELFSIM` and `T_SELFSIM_UPPER`: Minimum and maximum entity mention similarity threshold
- `T_CONTEXTSIM`: Minimum context similarity threshold

*The `code/utils` folder includes files to load contriever, which are retrieved from:<br>`https://github.com/facebookresearch/contriever`*

## Cite
If you use the dataset, **CovEReD** pipeline, or code from this repository, please cite the paper:
```bibtex
@inproceedings{modarressi-covered-2024,
    title="Consistent Document-Level Relation Extraction via Counterfactuals", 
    author="Ali Modarressi and Abdullatif Köksal and Hinrich Schütze",
    year="2024",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    address = "Miami, United States",
    publisher = "Association for Computational Linguistics",
}
```
