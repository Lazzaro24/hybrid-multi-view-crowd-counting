"""Building blocks of the proposed-model trainer (training_scripts/train_proposed_multiscene.py).

config      every hyperparameter / run setting, and apply_args() for the command line
cli         the argument parser
model       ProposedModel, the necks, the sparse-expert variants, checkpoint helpers
targets     ground-truth maps, the oracle dense-zone mask, BEV geometry, peak extraction
losses      loss functions and the per-patch loss of each curriculum stage
validation  one validation pass: per-stage metrics, report and checkpoint-selection metric
data        scene registries, train loaders, the fixed validation set
training    optimiser, EMA, BatchNorm handling and checkpoint saving
reporting   console tee, timing logs and the per-scene validation tables
"""
