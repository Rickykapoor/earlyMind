# EarlyMind - Infant ID Risk Detection

Multimodal infant intellectual disability risk detection using EEG, MRI, and HPO features.

## Model Card

### Model Description
EarlyMind is a multimodal deep learning system for assessing developmental risk in infants using:
- **EEG**: Electroencephalogram data (19 channels × 7680 timesteps)
- **MRI**: Magnetic Resonance Imaging slices (3 × 64 × 64)
- **HPO**: Human Phenotype Ontology features (5284 dimensions)

### Supported Modalities
The system can process any combination of available modalities. Results improve with more data sources.

### Output
- **Risk Probability**: Probability of intellectual disability (0-1)
- **DQ Estimate**: Developmental Quotient estimate (0-100)
- **DQ Label**: Classification (Normal, Mild, Moderate, Severe, Profound ID)
- **Confidence**: Prediction confidence level

### Limitations
- Designed for research use only
- Not a clinical diagnostic tool
- Requires trained clinical interpretation
- Performance may vary across populations

## Usage

1. Upload data files (EEG .npy/.csv, MRI .npy, HPO .npy/.csv)
2. Adjust symptom severity scores (0-1)
3. Click Predict to get risk assessment

## Technical Details

- **Architecture**: Fusion Transformer with cross-attention
- **Framework**: PyTorch
- **Device**: CPU (GPU support available)

## License

Research Use Only
