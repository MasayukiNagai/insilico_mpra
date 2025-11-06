# In silico MPRA
This repository provides tools to generate dense in‑silico MPRA libraries with predicted activities using [MPRALegNet](https://github.com/autosome-ru/human_legnet). Given one or more 200‑bp DNA sequences, the package applies random mutagenesis at a user‑specified rate, scores all generated variants with MPRALegNet, and saves results.

Key features
- Create dense perturbation libraries from TSV or FASTA inputs (sequences must be 200 bp).
- Control mutation rate and total number of variants per input sequence.
- Output per‑sequence HDF5 result files and PNG summaries (activity distributions).
- Simple CLI workflow: train a model, provide sequences, and generate libraries.

## Installation
```
git clone git@github.com:MasayukiNagai/insilico_mpra.git
pip install .
```

## Usage
1. Train MPRALegNet
2. Pick sequence(s) of interest. Each sequence has to be 200-bp.
3. Run


### 1. Training
If you don't have weights/checkpoints yet, you first need to train the oralce models.

```
# Option A: Use arguments
python train.py --data_path /path/to/data.h5 --model_dir ./models/my_model

# Option B: Use config files
python train.py --config config.json
```

### 2. Sequences of interest
You can supply sequences either as a TSV/CSV or as a FASTA. All sequences MUST be 200 bp long (MPRALegNet requirement).

**TSV format**
- `sequence` (required): DNA sequence used for dense perturbations (200 bp).
- `header` (optional): used to name the output file `{header}.h5`. If missing, the output file will be `mpra{index}.h5` where `index` is the row order (starting at 0).
- Example (tab-separated):
  ```
  sequence	header
  ATG...200bp...TGA	my_promoter_1
  GCT...200bp...AAC my_promoter_2
  ```

**FASTA format**
- Header: everything after the leading `>` is used as the output filename `{header}.h5`.
- Sequence: lines after a header are concatenated to form the full sequence (must be 200 bp).
- Example:
  ```
  >my_promoter_1
  ATG...200bp...TGA
  >my_promoter_2
  GCT...200bp...AAC
  ```

### 3. Generate in-silico MPRA libraries
```
python generate_library.py \
  --input /path/to/input.tsv \
  --weights_dir /path/to/weights_directory \
  --outdir /path/to/output_directory
  --mut_rate 0.15 \
  --num 30000
```

The outputs will be saved as follows:
```
outdir
├── header1.h5
├── header1.png  # histogram that shows distributions of activities
├── header2.h5
├── header2.png
...
```

### 4. Load the in-silico MPRA libraries
```
import h5py
import numpy as np

path = "output/yourfile.h5"

with h5py.File(path, "r") as f:
    print(list(f.keys()))  # ['x', 'y']

    x = f["x"][:]  # NumPy array of mutated one-hot sequences
    y = f["y"][:]  # NumPy array of predicted expression values

print(x.shape, y.shape)
```

## TODOs
- Data processing for the training data
- Output sequences in ACTG instead of numpy arrays?
- Loosen the dependencies (the current implementation strictly follows the original implementation)

## Authors
* Masayuki Nagai
* Steven Yu

## License
MIT License -- see LICENSE file for details
