# FLARE 2026 Task 1 submission image.
# Base: cu118 build (eval-host driver 510.39.01 = CUDA 11.6; cu12x would need
# driver >= 525). torch is NOT reinstalled so pip cannot pull a CUDA 12 build.
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# nnunetv2 pinned to the version the model was trained with.
RUN pip install --no-cache-dir --no-deps nnunetv2==2.8.1 && \
    pip install --no-cache-dir \
        "acvl-utils>=0.2.6,<0.3" \
        "dynamic-network-architectures>=0.4.4,<0.5" \
        "batchgenerators>=0.25.1" "batchgeneratorsv2>=0.3.2" \
        blosc2 einops nibabel SimpleITK scikit-image scikit-learn scipy \
        pandas tifffile imagecodecs yacs tqdm graphviz matplotlib seaborn && \
    rm -rf /root/.cache/pip

# Stand-in trainer class so nnU-Net can resolve the checkpoints (see custom_trainer.py).
COPY custom_trainer.py /tmp/nnUNetTrainer_Epoch5000_Lr1e3.py
RUN python3 -c "import nnunetv2, os, shutil; \
dst = os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer', 'variants', 'nnUNetTrainer_Epoch5000_Lr1e3.py'); \
shutil.copy('/tmp/nnUNetTrainer_Epoch5000_Lr1e3.py', dst); print('installed custom trainer ->', dst)"

WORKDIR /workspace
COPY predict.py /workspace/predict.py
COPY predict.sh /workspace/predict.sh
COPY maha_gate.json /workspace/maha_gate.json
COPY model /workspace/model
RUN mkdir -p /workspace/nnunet_dummy /workspace/inputs /workspace/outputs && \
    chmod +x /workspace/predict.sh && \
    python3 -c "import nnunetv2, torch; print('nnunetv2', nnunetv2.__file__); \
print('torch', torch.__version__, torch.version.cuda)" && \
    python3 -c "from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f; \
c = f('nnUNetTrainer_Epoch5000_Lr1e3'); print('custom trainer resolves OK ->', c)"

# The organizers run: docker run ... /bin/bash -c "sh predict.sh"
CMD ["sh", "/workspace/predict.sh"]
