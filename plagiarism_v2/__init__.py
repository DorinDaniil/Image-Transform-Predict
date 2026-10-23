"""
plagiarism_v2 -- second-generation augmentation vocabulary for the plagiarism model.

What changes compared to ``plagiarism``:
    * the transform vocabulary gets 3 extra tokens on top of the 13 used by the
      currently deployed model:
          13  jpeg_artefacts   (re-encoding with low JPEG quality)
          14  text_overlay     (realistic multilingual text in fixed layouts)
          15  fragment_edit    (ONE category for any other local change; the
                                kind is sampled from a fixed list: stickers,
                                blur / mosaic / box censoring, copy-move,
                                inpaint fill, recolor)
    * a new ``ImageTransformerV2`` that produces those augmentations.

What is intentionally reused *unchanged* from the existing code:
    * the model            -> ``plagiarism.effnet_plagiarism.ImageTransformPlagiarismPredictor``
    * the training loops   -> ``plagiarism.train_plagiarism.train_model`` and
                              ``plagiarism.tuning_plagiarism.train_model``
    * the datasets/loaders -> ``src.dataset.get_domainnet_dataloaders`` and
                              ``plagiarism.run_tune_plagiarism.CleanNegativePairsDataset``
    * the 300x300 preprocessing.

Only the token vocabulary (``vocab_size``) grows, which enlarges the decoder's
token embedding table and output projection by 3 rows; everything else in the
architecture is identical. The v2 model is trained from scratch.

Run from the repository root::

    python -m plagiarism_v2.run_train_plagiarism_v2 --config plagiarism_v2/configs/train_config_plagiarism_v2.yaml --data_path data
    python -m plagiarism_v2.preview_augmentations --images assets --out plagiarism_v2/preview
"""
