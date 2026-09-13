# Training: Dataset500 and the 5-fold plain U-Net ensemble

## Files

- `Dataset500_manifest.json`: the 8762 image + label paths on
  `FLARE-MedFM/PancancerCTSeg` and the nnU-Net `dataset.json`.
- `reproduce_dataset500.py`: downloads the manifest files and assembles
  `nnUNet_raw/Dataset500_FLAREPanCancer`.
- `download_validation.py`: downloads the validation subtree (public 50 +
  labels, hidden 100, 172 healthy).
- `preprocess_dataset500.sh`: fingerprint + preprocessing with the shipped plan.
- `nnUNetPlans_tr.json`: the plan used for the submission. Target spacing
  2.0 x 1.5 x 1.5 mm (z, y, x), set manually; the median spacing of the
  training set is 2.5 x 0.78 x 0.78 mm and the automatic plan would keep
  ~0.78 mm in-plane. Patch 96 x 160 x 160, batch 2, resampling with
  `resample_torch_fornnunet`. Do not run `nnUNetv2_plan_experiment`: it would
  produce a different plan.
- `nnUNetTrainer_Epoch5000_Lr1e3.py`: trainer with 5000 epochs and initial
  learning rate 1e-3; everything else is stock nnU-Net.
- `build_manifest.py`: one-off script that produced the manifest from our
  internal layout (server-specific paths, kept for provenance).

## Dataset500

The full labelled set (17 575 CT scans) is dominated by whole-body and chest
collections (about 5700 scans each), while the rarest sources have 52-154
scans. Dataset500 is a per-source capped subset: large sources are capped,
rare ones are kept in full; 28 source datasets, 11 anatomical groups, 8762 scans.

| Group | Sources | Scans |
|---|---|---:|
| Whole-body | DeepLesion, MSWAL, LongitudinalCTLesion, autoPET-CT | 2764 |
| Pancreas | PanTS, PANORAMA, PanTrack, MSD | 1797 |
| Liver | TotalSegmentator-lesions, MSD, MSD-HepaticVessel, WAW-TACE, HCC-TACE, Colorectal-Metastases | 1631 |
| Chest | LUNA25, LIDC-IDRI, LNDb, NSCLC-Radiomics, NSCLC-Radiogenomics, MSD-LungTumor | 1251 |
| Kidney | KiTS23 | 488 |
| Lymph nodes | Mediastinal-SEG, CT | 300 |
| Esophagus | AbdomenAtlas | 154 |
| Colon | MSD | 126 |
| Head and neck | SegRap25 | 120 |
| Endometrium | AbdomenAtlas | 79 |
| Adrenal glands | ACC-Ki67-Seg | 52 |

With identical training, the full set gives ~69.4 % DSC on hidden validation
versus 75.38 % for Dataset500.

## Steps

```bash
pip install huggingface_hub nnunetv2==2.8.1
export HF_TOKEN=...            # access to FLARE-MedFM/PancancerCTSeg

python reproduce_dataset500.py --root /data/flare26      # ~350 GB, symlinks into nnUNet_raw
python download_validation.py  --root /data/flare26      # ~25 GB, optional subsets: --subset public|hidden|healthy
./preprocess_dataset500.sh /data/flare26                 # fingerprint + preprocess with nnUNetPlans_tr

# install the trainer once per environment
cp nnUNetTrainer_Epoch5000_Lr1e3.py \
   $(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/training_length/

export nnUNet_raw=/data/flare26/nnUNet_raw
export nnUNet_preprocessed=/data/flare26/nnUNet_preprocessed
export nnUNet_results=/data/flare26/nnUNet_results
for f in 0 1 2 3 4; do
  nnUNetv2_train 500 3d_fullres $f -tr nnUNetTrainer_Epoch5000_Lr1e3 -p nnUNetPlans_tr --npz
done
```

Each fold takes about 72 h on one H100 80 GB. The five `checkpoint_final.pth`
files go into `../docker/model/fold_{0..4}/` together with `plans.json` and
`dataset.json` from the results folder.
