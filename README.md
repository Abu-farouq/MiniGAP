 miniGAP

 Educational Genomic Interaction Modeling Prototype

miniGAP is an early Python prototype for exploring a basic genomic machine-learning workflow: genomic binning, feature generation, neural-network training, and heatmap visualization.

The current version uses ChIP-seq peak-signal data to construct features for nearby genomic bins and display model-generated pairwise scores.

 Important scientific note

This repository is currently an educational prototype. It does not yet make experimentally validated chromatin-interaction predictions.

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

Features
Genome binning at 5,000 bp resolution
Automatic feature extraction:
Signal A
Signal B
Absolute difference
Genomic distance (in bins)
Correlation-based binary labels
Tiny neural network (4 and 16 and 8 and 1)
Heatmap visualization of predicted interactions for the first 50 bins

Project Structure
miniGAP.py   # Main script
Requirements
All dependencies used in this project are directly visible in the code:
pandas
numpy
torch (PyTorch)
matplotlib

How It Works (Step by Step)
1. Load BED file
The script loads a BED-like file containing genomic peaks and signal values.
The expected file path (as used in the script) is:
/storage/emulated/0/ENCFF252PLM.bed.txt
2. Bin the genome
Each genomic position is assigned to a 5,000 bp bin.
3. Compute features
For each bin, the mean and variance of the signal are calculated.
4. Generate training samples
Only bins within a distance of 10 are used.
Labels are assigned based on correlation > 0.5.
5. Train a small neural network
The model predicts interaction likelihood between two bins.
6. Generate interaction matrix
Predictions for the first 50 bins are visualized as a heatmap.

Usage
Run the script:
python miniGAP.py
Output includes:
Training logs (loss per epoch)
Interaction heatmap visualization
Printed confirmation when complete

Example Output
Dataset size information
Training loss for each epoch
Heatmap of predicted interaction scores

Notes
The label creation uses a simple correlation threshold (0.5).
If the correlation cannot be confirmed or falls below threshold, the label is 0.
All steps and calculations are visible in the code for transparency.
No external assumptions are used beyond what is explicitly in the script.

