# =============================================================================
# alzheimer_fl/models/test_model.py
# ----------------------------------
# Verification script: proves tensor alignment through the full forward pass.
# Updated for 4-class classification with 9 clinical features.
#
# Usage:  python models/test_model.py
# =============================================================================

import torch
from multimodal_model import (
    MultimodalAlzheimersModel,
    CNNBranch,
    MLPBranch,
    FusionClassifier,
)


def test_tensor_alignment() -> None:
    """
    End-to-end shape verification.

    Creates dummy data and traces shapes through every layer to ensure
    the tensor dimensions align perfectly during concatenation.
    Updated for 4-class, 9-feature configuration.
    """
    print("=" * 60)
    print("  TENSOR ALIGNMENT VERIFICATION (4-class, 9-feature)")
    print("=" * 60)

    # ── Configuration ────────────────────────────────────────────────────
    BATCH_SIZE: int = 4
    NUM_CLINICAL_FEATURES: int = 9   # MMSE, CDR, Age, EDUC, nWBV, eTIV, ASF, SES, Gender
    NUM_CLASSES: int = 4              # MildDemented, ModerateDemented, NonDemented, VeryMildDemented
    MRI_HEIGHT: int = 224
    MRI_WIDTH: int = 224
    MRI_CHANNELS: int = 1  # Grayscale

    # ── Create dummy inputs ──────────────────────────────────────────────
    mri_input: torch.Tensor = torch.randn(
        BATCH_SIZE, MRI_CHANNELS, MRI_HEIGHT, MRI_WIDTH
    )
    clinical_input: torch.Tensor = torch.randn(
        BATCH_SIZE, NUM_CLINICAL_FEATURES
    )

    print(f"\n[INPUT]")
    print(f"  MRI shape:      {tuple(mri_input.shape)}")
    print(f"  Clinical shape: {tuple(clinical_input.shape)}")
    print(f"  Num classes:    {NUM_CLASSES}")

    # ── Test CNN Branch independently ────────────────────────────────────
    cnn = CNNBranch(pretrained=False)  # No pretrained for quick test
    cnn.eval()
    with torch.no_grad():
        z_cnn = cnn(mri_input)
    print(f"\n[CNN BRANCH]")
    print(f"  Output shape:   {tuple(z_cnn.shape)}")
    assert z_cnn.shape == (BATCH_SIZE, 512), f"Expected (B, 512), got {z_cnn.shape}"
    print(f"  [OK] Matches expected (B, 512)")

    # ── Test MLP Branch independently ────────────────────────────────────
    mlp = MLPBranch(input_dim=NUM_CLINICAL_FEATURES)
    mlp.eval()
    with torch.no_grad():
        z_mlp = mlp(clinical_input)
    print(f"\n[MLP BRANCH]")
    print(f"  Output shape:   {tuple(z_mlp.shape)}")
    assert z_mlp.shape == (BATCH_SIZE, 128), f"Expected (B, 128), got {z_mlp.shape}"
    print(f"  [OK] Matches expected (B, 128)")

    # ── Test Fusion Classifier independently ─────────────────────────────
    fusion = FusionClassifier(num_classes=NUM_CLASSES)
    fusion.eval()
    with torch.no_grad():
        logits = fusion(z_cnn, z_mlp)
    print(f"\n[FUSION CLASSIFIER]")
    print(f"  Fused input:    (B, {512 + 128}) = (B, 640)")
    print(f"  Output shape:   {tuple(logits.shape)}")
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected (B, {NUM_CLASSES}), got {logits.shape}"
    print(f"  [OK] Matches expected (B, {NUM_CLASSES})")

    # ── Test Full Model (end-to-end) ─────────────────────────────────────
    model = MultimodalAlzheimersModel(
        num_clinical_features=NUM_CLINICAL_FEATURES,
        num_classes=NUM_CLASSES,
        pretrained_cnn=False,
    )
    model.eval()
    with torch.no_grad():
        full_logits = model(mri_input, clinical_input)
    print(f"\n[FULL MODEL -- End-to-End]")
    print(f"  Output shape:   {tuple(full_logits.shape)}")
    assert full_logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected (B, {NUM_CLASSES}), got {full_logits.shape}"
    )
    print(f"  [OK] Matches expected (B, {NUM_CLASSES})")

    # ── Verify predicted classes ─────────────────────────────────────────
    preds: torch.Tensor = full_logits.argmax(dim=1)
    print(f"\n[PREDICTIONS]")
    for i in range(BATCH_SIZE):
        label = model.CLASS_LABELS[preds[i].item()]
        print(f"  Sample {i}: class {preds[i].item()} -> {label}")

    # ── Grad-CAM target layer check ──────────────────────────────────────
    target_layer = model.get_cnn_target_layer()
    print(f"\n[GRAD-CAM TARGET]")
    print(f"  Layer:  {target_layer.__class__.__name__}")
    print(f"  [OK] Ready for pytorch-grad-cam integration")

    # ── Parameter count summary ──────────────────────────────────────────
    print(f"\n[MODEL SUMMARY]")
    print(model)

    total_trainable = model.count_parameters(trainable_only=True)
    total_all = model.count_parameters(trainable_only=False)
    print(f"\n  Trainable params:  {total_trainable:>10,}")
    print(f"  Total params:      {total_all:>10,}")

    print(f"\n{'=' * 60}")
    print("  ALL ASSERTIONS PASSED -- TENSORS ALIGN PERFECTLY [OK]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_tensor_alignment()
