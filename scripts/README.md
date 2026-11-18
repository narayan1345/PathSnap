# Scripts

Run the dataset inventory from the repository root:

```powershell
python -m pip install -e .
python scripts/inspect_dataset.py --data-root data/raw
```

The report is written to `results/dataset_inventory.json`. The script performs
read-only trip/file discovery; it does not train models or calculate model scores.
