 miniGAP

 Educational Genomic Interaction Modeling Prototype

miniGAP is an early Python prototype for exploring a basic genomic machine-learning workflow: genomic binning, feature generation, neural-network training, and heatmap visualization.

The current version uses ChIP-seq peak-signal data to construct features for nearby genomic bins and display model-generated pairwise scores.

 Important scientific note

This repository is currently an educational prototype. It does **not** yet make experimentally validated chromatin-interaction predictions.

The original correlation-based labeling approach is being replaced. The next version will use experimentally measured Hi-C contact data as ground-truth labels and matched epigenomic features from the same cell type and genome assembly.

Until that upgrade and independent evaluation are complete, the heatmap should be interpreted only as a demonstration of the modeling workflow—not as a biological interaction map.

 Current workflow

1. Load a BED-like ChIP-seq peak-signal file.
2. Divide genomic positions into fixed-size bins.
3. Calculate simple per-bin signal features.
4. Generate features for nearby bin pairs:
   - Signal in bin A
   - Signal in bin B
   - Absolute signal difference
   - Genomic distance in bins
5. Train a small PyTorch neural network.
6. Display a heatmap of pairwise model scores.

 Planned scientific version

The planned MiniGAP scientific proof of concept will use:

- Human K562 cells
- GRCh38 genome assembly
- Real Hi-C contact measurements as model targets
- Matched epigenomic features, beginning with CTCF and H3K27ac
- A chromosome 22 subset at 10 kb resolution for initial training and evaluation
- Held-out genomic regions or chromosomes for validation

The project will report appropriate evaluation metrics and will distinguish predictive performance from exploratory visualization.

 Requirements

- Python 3
- pandas
- numpy
- PyTorch
- matplotlib

 Current usage

The present script expects a local BED-like input file. Update the input path in `miniGAP.py` to the location of your own file, then run:

```bash
python miniGAP.py
```
 Status

Active development. The repository is being transitioned from a demonstration prototype to a reproducible, scientifically grounded genomic interaction-modeling project.
