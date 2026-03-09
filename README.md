# AID Transfer Learning

Reproducible transfer learning experiments on the
[AID (Aerial Image Dataset)](https://captain-whu.github.io/DiRS/) benchmark.

---

## Project Structure

```
aid-transfer-learning/
├── configs/
│   └── baseline.yaml          # Baseline experiment configuration
├── src/
│   ├── datasets/
│   │   ├── aid_dataset.py     # AIDDataset class + transform factories
│   │   ├── dataloader.py      # get_dataloaders() factory
│   │   └── split_utils.py     # Stratified splitting + summary table
│   └── utils/
│       ├── config.py          # YAML config loader → Config dataclass
│       └── seed.py            # set_seed() for full reproducibility
├── scripts/
│   ├── prepare_dataset.py             # Verify data, generate splits
│   └── create_visualization_subset.py # Build fixed 30×30 subset CSV
├── outputs/                   # Generated artefacts (CSVs, checkpoints …)
├── notebooks/                 # Exploratory / analysis notebooks
├── data/
│   └── AID/                   # ← place the AID dataset here
│       ├── airport/
│       ├── bareland/
│       └── ...  (30 classes)
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / navigate to project
cd aid-transfer-learning

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the AID dataset under data/AID/
#    Each class should be a sub-directory containing image files.
```

---

## Infrastructure Scripts

### 1 · Verify dataset and generate splits

```bash
python scripts/prepare_dataset.py --config configs/baseline.yaml
```

This script will:

1. Read all configuration values from `configs/baseline.yaml`.
2. Verify the dataset path exists and contains class sub-directories.
3. Perform a **stratified** train / val / test split (default 70 / 15 / 15).
4. Print a per-class breakdown table to stdout.
5. Save three manifest CSVs to `outputs/`:
   - `outputs/train_split.csv`
   - `outputs/val_split.csv`
   - `outputs/test_split.csv`

Each CSV contains columns: `path`, `label_idx`, `class_name`.

---

### 2 · Create the visualisation subset

```bash
python scripts/create_visualization_subset.py --config configs/baseline.yaml
```

This script will:

1. Discover the full dataset.
2. Reproducibly sample **30 images from each of the 30 classes** using the
   seed specified in the config.
3. Save the sampled paths to:
   - `outputs/visualization_subset.csv`

This fixed subset is used for all PCA / t-SNE comparisons so that feature
representations across different models are always evaluated on identical images.

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--images-per-class N` | `30` | Images to sample per class |
| `--num-classes N` | `30` | Number of classes to include |

---

## Configuration

All experiment parameters live in a single YAML file.
The default is `configs/baseline.yaml`:

```yaml
dataset_path: data/AID     # Root of the AID directory
image_size:   224          # Resize target (height = width)
batch_size:   32
num_workers:  4
train_split:  0.70
val_split:    0.15
test_split:   0.15
seed:         42
```

To create a new experiment configuration, copy `baseline.yaml`, modify the
values you need, and pass the new file with `--config path/to/your.yaml`.

---

## Reproducibility

Every random operation (Python `random`, NumPy, PyTorch CPU and CUDA) is
seeded through `src/utils/seed.py::set_seed(seed)`.  All scripts call this
function before any data-dependent operation, so results are bit-for-bit
identical for a given seed.
