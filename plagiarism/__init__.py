"""
Plagiarism-detection module (independent of the paper).

This package contains the plagiarism-oriented pair model, its training/tuning
loops and entry points, weight-export utilities and standalone inference models.

It reuses two shared building blocks from the main package instead of
duplicating them:
    * the autoregressive decoder  -> ``src.model.decoder.TransformDecoder``
    * the data pipeline           -> ``src.dataset`` (tokenizer, augmentations,
                                      DomainNet loaders)

Because of that, all entry points are meant to be run from the repository root
so that both ``src`` and ``plagiarism`` are importable, e.g.::

    python -m plagiarism.run_train_plagiarism --config plagiarism/configs/train_config_plagiarism.yaml --data_path data
"""
