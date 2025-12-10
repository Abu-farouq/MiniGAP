miniGAP: A Minimal Genome Interaction Prediction Demo
miniGAP is a lightweight Python project that demonstrates how to generate simple genomic interaction features from ChIP-seq peak signal data and train a tiny neural network to predict bin-to-bin interaction likelihoods.
The workflow includes:
Loading genomic BED data
Binning the genome into 5 kb windows
Computing mean/variance of ChIP-seq signal per bin
Generating training samples using correlation-based labels
Training a small PyTorch neural network
Producing and visualizing a 2D interaction score matrix
This project is intentionally minimal and optimized for running on a mobile Python environment (e.g., PyDroid).

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

How It Works (Step-by-Step)
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

