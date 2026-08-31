# =============================================================================
# alzheimer_fl/models/multimodal_model.py
# ----------------------------------------
# Multimodal Alzheimer's Disease Prediction Model
#
# Architecture (per the 2025 IEEE paper):
#   ┌─────────────────┐   ┌─────────────────┐
#   │  2D MRI Slice    │   │ Clinical Tabular │
#   │  (B, 1, 224, 224)│   │  (B, D_tab)      │
#   └───────┬─────────┘   └───────┬─────────┘
#           │                      │
#     ┌─────▼──────┐        ┌─────▼──────┐
#     │  CNN Branch │        │  MLP Branch │
#     │ (ResNet-18) │        │ (3-layer)   │
#     └─────┬──────┘        └─────┬──────┘
#           │ ℝ^{B×512}           │ ℝ^{B×128}
#           └──────┬──────────────┘
#                  │ Concatenate → ℝ^{B×640}
#            ┌─────▼──────┐
#            │   Fusion    │
#            │ Classifier  │
#            └─────┬──────┘
#                  │ ℝ^{B×3}
#                  ▼
#           {CN, MCI, AD}
#
# Tensor Alignment Contract:
#   - CNN branch outputs:  (batch_size, 512)
#   - MLP branch outputs:  (batch_size, 128)
#   - Fusion input:        (batch_size, 512 + 128 = 640)
#   - Final output:        (batch_size, 3)  — logits (not softmax)
#
# Author:  Kasyap (Final-Year Academic Project)
# License: Academic / Research Use Only
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN BRANCH — Feature extractor for structural MRI slices
# ─────────────────────────────────────────────────────────────────────────────
class CNNBranch(nn.Module):
    """
    Convolutional feature extractor built on a modified ResNet-18.

    Why ResNet-18?
    ─────────────
    • ResNet-18 is lightweight enough for federated settings where each
      client may have limited GPU memory, yet deep enough to capture
      spatial hierarchies in brain MRI (sulci, gyri, ventricle shape).
    • Residual connections solve the vanishing gradient problem:
          h(x) = F(x) + x
      where F(x) is the residual mapping learned by the conv block.

    Modifications from vanilla ResNet-18:
    ─────────────────────────────────────
    1. First conv layer changed:  3-channel RGB  →  1-channel grayscale
       (MRI slices are single-channel intensity images).
    2. Final fc layer removed and replaced with an Identity, so the
       forward pass returns a 512-dim feature vector (the avgpool output).
    3. Optional dropout before the feature output for regularization.

    Input:  (B, 1, 224, 224) — batch of grayscale MRI slices
    Output: (B, 512)         — compact spatial feature embedding
    """

    # Class-level constant: the dimensionality of the output embedding.
    # This is fixed by ResNet-18's architecture (512 filters in layer4).
    OUTPUT_DIM: int = 512

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_early_layers: bool = False,
    ) -> None:
        """
        Args:
            pretrained:          If True, load ImageNet-pretrained weights.
                                 Transfer learning significantly boosts
                                 convergence when MRI training data is small.
            dropout_rate:        Probability of zeroing an element before
                                 the output.  Dropout acts as an approximate
                                 Bayesian regularizer:
                                     p(y|x) ≈ (1/T) Σ_t f(x; θ_t)
                                 where each θ_t is a different dropout mask.
            freeze_early_layers: If True, freeze conv1 through layer2.
                                 Useful when fine-tuning with very small
                                 client datasets in FL to avoid overfitting
                                 on low-level texture features.
        """
        super().__init__()

        # ── Step 1: Load base ResNet-18 ──────────────────────────────────
        # torchvision ≥ 0.13 uses `weights=` instead of `pretrained=`.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet: models.ResNet = models.resnet18(weights=weights)

        # ── Step 2: Adapt first conv for 1-channel input ─────────────────
        # Original conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # We keep kernel_size, stride, padding identical so spatial dims
        # remain (B, 64, 112, 112) after this layer.
        #
        # Weight initialization strategy for the new 1-channel conv:
        #   If pretrained, we average the 3-channel weights along the
        #   input-channel axis → (64, 1, 7, 7).  This preserves the
        #   learned Gabor-like edge detectors from ImageNet.
        original_conv1_weight: torch.Tensor = resnet.conv1.weight.data  # (64, 3, 7, 7)

        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,  # BatchNorm immediately follows; bias is redundant
        )

        if pretrained:
            # Mean across the RGB dimension → (64, 1, 7, 7)
            # Mathematically: w_new[i, 0, h, w] = (1/3) Σ_c w_old[i, c, h, w]
            resnet.conv1.weight.data = original_conv1_weight.mean(
                dim=1, keepdim=True
            )

        # ── Step 3: Remove the original classification head ──────────────
        # ResNet-18's fc layer is: Linear(512, 1000) for ImageNet classes.
        # We replace it with Identity so that self.backbone(x) returns
        # the 512-dim feature vector from the AdaptiveAvgPool2d.
        resnet.fc = nn.Identity()

        self.backbone: nn.Module = resnet

        # ── Step 4: Optional early-layer freezing for FL stability ───────
        if freeze_early_layers:
            # Freeze: conv1, bn1, relu, maxpool, layer1, layer2
            # Keep trainable: layer3, layer4 (high-level semantic features)
            frozen_modules = [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
            ]
            for module in frozen_modules:
                for param in module.parameters():
                    param.requires_grad = False

        # ── Step 5: Dropout for regularization ───────────────────────────
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN branch.

        Args:
            mri: Input tensor of shape (B, 1, 224, 224).
                 Values should be normalized (e.g., zero-mean, unit-var).

        Returns:
            Feature vector of shape (B, 512).

        Mathematical pipeline:
            x₀ = mri                                     # (B, 1, 224, 224)
            x₁ = ReLU(BN(Conv7×7(x₀)))                  # (B, 64, 112, 112)
            x₂ = MaxPool(x₁)                             # (B, 64, 56, 56)
            x₃ = ResBlock_layer1(x₂)                     # (B, 64, 56, 56)
            x₄ = ResBlock_layer2(x₃)                     # (B, 128, 28, 28)
            x₅ = ResBlock_layer3(x₄)                     # (B, 256, 14, 14)
            x₆ = ResBlock_layer4(x₅)                     # (B, 512, 7, 7)
            x₇ = AdaptiveAvgPool(x₆)                     # (B, 512, 1, 1)
            x₈ = Flatten(x₇)                             # (B, 512)
            out = Dropout(x₈)                             # (B, 512)
        """
        features: torch.Tensor = self.backbone(mri)   # (B, 512)
        features = self.dropout(features)              # (B, 512)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLP BRANCH — Feature extractor for clinical / psychological scores
# ─────────────────────────────────────────────────────────────────────────────
class MLPBranch(nn.Module):
    """
    Multi-Layer Perceptron for tabular clinical data.

    Clinical features from OASIS-3 typically include:
        - MMSE (Mini-Mental State Examination) score
        - CDR  (Clinical Dementia Rating)
        - Age, Education, SES (Socioeconomic Status)
        - eTIV (Estimated Total Intracranial Volume)
        - nWBV (Normalized Whole-Brain Volume)
        - ASF  (Atlas Scaling Factor)
        - Gender (encoded)

    Architecture:
        Input(D_tab) → [Linear → BN → ReLU → Dropout] × 3 → Output(128)

    Each hidden layer applies the transformation:
        h_l = Dropout(ReLU(BN(W_l · h_{l-1} + b_l)))

    where:
        W_l ∈ ℝ^{d_l × d_{l-1}}  — weight matrix
        b_l ∈ ℝ^{d_l}            — bias vector
        BN normalizes:  ĥ = (h - μ_B) / √(σ²_B + ε)

    Input:  (B, D_tab) — batch of clinical feature vectors
    Output: (B, 128)   — compact clinical feature embedding
    """

    # Class-level constant: dimensionality of the output embedding.
    OUTPUT_DIM: int = 128

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        dropout_rate: float = 0.4,
    ) -> None:
        """
        Args:
            input_dim:    Number of clinical features (D_tab).
                          For OASIS-3, this is typically 8–12.
            hidden_dims:  Tuple specifying the width of each hidden layer.
                          Default (256, 256, 128) provides a gradual
                          compression bottleneck: D_tab → 256 → 256 → 128.
            dropout_rate: Applied after each ReLU activation.  Higher than
                          the CNN branch because tabular data is more prone
                          to overfitting on small federated partitions.
        """
        super().__init__()

        # Dynamically build the MLP stack from the hidden_dims tuple.
        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for i, h_dim in enumerate(hidden_dims):
            # ── Linear transformation: h = Wx + b ───────────────────────
            layers.append(nn.Linear(prev_dim, h_dim))

            # ── Batch Normalization ──────────────────────────────────────
            # Stabilizes training by reducing internal covariate shift.
            # In FL, BN statistics are kept local to each client to
            # avoid leaking distributional information about patient data.
            layers.append(nn.BatchNorm1d(h_dim))

            # ── ReLU activation: σ(x) = max(0, x) ───────────────────────
            # Chosen over sigmoid/tanh because:
            #   1. No vanishing gradient for positive activations
            #   2. Computationally cheaper (no exponential)
            layers.append(nn.ReLU(inplace=True))

            # ── Dropout regularization ───────────────────────────────────
            layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = h_dim

        # Package all layers into a Sequential container.
        self.mlp: nn.Sequential = nn.Sequential(*layers)

        # Sanity check: final hidden dim must equal OUTPUT_DIM.
        assert hidden_dims[-1] == self.OUTPUT_DIM, (
            f"Last hidden_dim ({hidden_dims[-1]}) must equal "
            f"MLPBranch.OUTPUT_DIM ({self.OUTPUT_DIM})"
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP branch.

        Args:
            clinical: Input tensor of shape (B, D_tab).
                      Features should be standardized (z-score) before
                      being passed in:  x' = (x - μ) / σ

        Returns:
            Feature vector of shape (B, 128).

        Mathematical pipeline:
            h₀ = clinical                                # (B, D_tab)
            h₁ = Dropout(ReLU(BN(W₁·h₀ + b₁)))         # (B, 256)
            h₂ = Dropout(ReLU(BN(W₂·h₁ + b₂)))         # (B, 256)
            h₃ = Dropout(ReLU(BN(W₃·h₂ + b₃)))         # (B, 128)
            out = h₃                                     # (B, 128)
        """
        return self.mlp(clinical)  # (B, 128)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUSION CLASSIFIER — Late fusion + 3-class prediction head
# ─────────────────────────────────────────────────────────────────────────────
class FusionClassifier(nn.Module):
    """
    Fusion and classification head.

    Strategy: **Late Fusion via Concatenation**
    ─────────────────────────────────────────
    The CNN and MLP branches produce independent embeddings.  We
    concatenate them along the feature dimension:

        z = [z_cnn ‖ z_mlp] ∈ ℝ^{B × (512 + 128)} = ℝ^{B × 640}

    This concatenated vector is then passed through two fully-connected
    layers to produce the final 3-class logits.

    Why concatenation over attention-based fusion?
    ──────────────────────────────────────────────
    1. Simplicity & reproducibility for academic review.
    2. With only ~640 fused features, an attention mechanism adds
       complexity without significant performance gain.
    3. Concatenation preserves all information from both modalities
       without lossy compression.

    Input:  cnn_features  (B, 512)
            mlp_features  (B, 128)
    Output: logits        (B, 3)   — raw scores for {CN, MCI, AD}
    """

    def __init__(
        self,
        cnn_dim: int = CNNBranch.OUTPUT_DIM,     # 512
        mlp_dim: int = MLPBranch.OUTPUT_DIM,     # 128
        num_classes: int = 4,
        fusion_hidden: int = 256,
        dropout_rate: float = 0.5,
    ) -> None:
        """
        Args:
            cnn_dim:       Dimensionality of CNN branch output.
            mlp_dim:       Dimensionality of MLP branch output.
            num_classes:   Number of diagnostic classes.
                           0 = Mild Dementia
                           1 = Moderate Dementia
                           2 = Non Demented
                           3 = Very Mild Dementia
            fusion_hidden: Width of the hidden fusion layer.
            dropout_rate:  Dropout after the fusion hidden layer.
                           Set higher (0.5) because this is the final
                           decision-making layer and we want to maximize
                           generalization to unseen client distributions.
        """
        super().__init__()

        fused_dim: int = cnn_dim + mlp_dim  # 512 + 128 = 640

        self.classifier: nn.Sequential = nn.Sequential(
            # ── Fusion hidden layer ──────────────────────────────────────
            # Reduces 640 → 256, learning cross-modal interactions:
            #     h_fused = ReLU(BN(W_f · z + b_f))
            nn.Linear(fused_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # ── Output layer ─────────────────────────────────────────────
            # Projects 256 → 3 (raw logits).
            # We do NOT apply softmax here because:
            #   1. nn.CrossEntropyLoss internally applies log_softmax
            #   2. For inference, argmax(logits) = argmax(softmax(logits))
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        mlp_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse features and classify.

        Args:
            cnn_features: (B, 512) from CNNBranch
            mlp_features: (B, 128) from MLPBranch

        Returns:
            logits: (B, 3) raw class scores

        Mathematical pipeline:
            z = [z_cnn ‖ z_mlp]                          # (B, 640)
            h = Dropout(ReLU(BN(W_f · z + b_f)))         # (B, 256)
            logits = W_o · h + b_o                        # (B, 3)
        """
        # Concatenate along feature dimension (dim=1).
        # Shape check: both tensors must share the batch dimension.
        assert cnn_features.size(0) == mlp_features.size(0), (
            f"Batch size mismatch: CNN={cnn_features.size(0)}, "
            f"MLP={mlp_features.size(0)}"
        )

        fused: torch.Tensor = torch.cat(
            [cnn_features, mlp_features], dim=1
        )  # (B, 640)

        logits: torch.Tensor = self.classifier(fused)  # (B, 3)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTIMODAL ALZHEIMERS MODEL — Top-level wrapper
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalAlzheimersModel(nn.Module):
    """
    Top-level multimodal model for federated Alzheimer's prediction.

    This class composes the three sub-modules (CNNBranch, MLPBranch,
    FusionClassifier) into a single nn.Module that can be:
      1. Serialized / deserialized for Flower FL communication.
      2. Wrapped with Grad-CAM (targeting the CNN branch's layer4).
      3. Analyzed with SHAP (by isolating the MLP branch path).

    Complete forward pass:
        (mri, clinical) → CNNBranch(mri) → z_cnn
                          MLPBranch(clinical) → z_mlp
                          FusionClassifier(z_cnn, z_mlp) → logits

    Usage:
        model = MultimodalAlzheimersModel(num_clinical_features=10)
        mri   = torch.randn(8, 1, 224, 224)   # batch of 8 MRI slices
        tab   = torch.randn(8, 10)             # batch of 8 clinical vectors
        logits = model(mri, tab)               # (8, 3)
        preds  = logits.argmax(dim=1)          # (8,) class indices
    """

    # Diagnostic class labels for human-readable outputs.
    # Order matches torchvision.datasets.ImageFolder alphabetical sorting:
    #   MildDemented=0, ModerateDemented=1, NonDemented=2, VeryMildDemented=3
    CLASS_LABELS: Tuple[str, ...] = (
        "Mild Dementia",
        "Moderate Dementia",
        "Non Demented",
        "Very Mild Dementia",
    )

    def __init__(
        self,
        num_clinical_features: int,
        pretrained_cnn: bool = True,
        cnn_dropout: float = 0.3,
        mlp_dropout: float = 0.4,
        mlp_hidden_dims: Tuple[int, ...] = (256, 256, 128),
        fusion_hidden: int = 256,
        fusion_dropout: float = 0.5,
        num_classes: int = 4,
        freeze_early_cnn: bool = False,
    ) -> None:
        """
        Args:
            num_clinical_features: Number of tabular clinical features (D_tab).
            pretrained_cnn:        Use ImageNet-pretrained ResNet-18 weights.
            cnn_dropout:           Dropout rate for CNN branch.
            mlp_dropout:           Dropout rate for MLP branch.
            mlp_hidden_dims:       Widths for the MLP hidden layers.
            fusion_hidden:         Width of the fusion hidden layer.
            fusion_dropout:        Dropout rate for the fusion classifier.
            num_classes:           Number of diagnostic output classes.
            freeze_early_cnn:      Freeze early ResNet layers (for FL stability).
        """
        super().__init__()

        # ── Instantiate sub-modules ──────────────────────────────────────
        self.cnn_branch: CNNBranch = CNNBranch(
            pretrained=pretrained_cnn,
            dropout_rate=cnn_dropout,
            freeze_early_layers=freeze_early_cnn,
        )

        self.mlp_branch: MLPBranch = MLPBranch(
            input_dim=num_clinical_features,
            hidden_dims=mlp_hidden_dims,
            dropout_rate=mlp_dropout,
        )

        self.fusion: FusionClassifier = FusionClassifier(
            cnn_dim=CNNBranch.OUTPUT_DIM,    # 512
            mlp_dim=MLPBranch.OUTPUT_DIM,    # 128
            num_classes=num_classes,
            fusion_hidden=fusion_hidden,
            dropout_rate=fusion_dropout,
        )

        # Store config for serialization / logging in FL.
        self.num_clinical_features: int = num_clinical_features
        self.num_classes: int = num_classes

    def forward(
        self,
        mri: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full multimodal forward pass.

        Args:
            mri:      (B, 1, 224, 224) — grayscale MRI slices.
            clinical: (B, D_tab)       — standardized clinical features.

        Returns:
            logits:   (B, 3)           — raw class scores (no softmax).

        End-to-end mathematical summary:
        ─────────────────────────────────
            z_cnn   = Dropout(ResNet18_features(mri))           ∈ ℝ^{B×512}
            z_mlp   = MLP_3layer(clinical)                      ∈ ℝ^{B×128}
            z_fused = [z_cnn ‖ z_mlp]                           ∈ ℝ^{B×640}
            h       = Dropout(ReLU(BN(W_f · z_fused + b_f)))   ∈ ℝ^{B×256}
            logits  = W_o · h + b_o                             ∈ ℝ^{B×3}
        """
        # ── Branch 1: Spatial features from MRI ─────────────────────────
        z_cnn: torch.Tensor = self.cnn_branch(mri)        # (B, 512)

        # ── Branch 2: Clinical features from tabular data ────────────────
        z_mlp: torch.Tensor = self.mlp_branch(clinical)   # (B, 128)

        # ── Fusion + Classification ──────────────────────────────────────
        logits: torch.Tensor = self.fusion(z_cnn, z_mlp)  # (B, 3)

        return logits

    def get_cnn_target_layer(self) -> nn.Module:
        """
        Return the target layer for Grad-CAM visualization.

        Grad-CAM computes the gradient of the class score y^c with
        respect to the feature maps A^k of a target convolutional layer:

            α_k^c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}     (importance weight)

            L^c_{Grad-CAM} = ReLU(Σ_k  α_k^c · A^k)      (heatmap)

        We target ResNet-18's `layer4` — the deepest convolutional block —
        because it captures the most semantically meaningful features
        (e.g., hippocampal atrophy, ventricular enlargement).

        Returns:
            nn.Module: The layer4 block of the CNN backbone.
        """
        return self.cnn_branch.backbone.layer4

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Useful for logging in FL experiments to verify that all clients
        share the same architecture (parameter count must be identical
        across clients for FedAvg weight averaging to work).

        The FedAvg aggregation formula:
            w_{t+1} = Σ_k (n_k / n) · w_k^{t+1}

        requires every client model w_k to have the same parameter
        shape and count.

        Args:
            trainable_only: If True, count only parameters with
                            requires_grad=True.

        Returns:
            Total number of (trainable) parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """Pretty-print with parameter counts per branch."""
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters())
        mlp_params = sum(p.numel() for p in self.mlp_branch.parameters())
        fus_params = sum(p.numel() for p in self.fusion.parameters())
        total = cnn_params + mlp_params + fus_params

        return (
            f"MultimodalAlzheimersModel(\n"
            f"  CNN Branch (ResNet-18):   {cnn_params:>10,} params\n"
            f"  MLP Branch (3-layer):     {mlp_params:>10,} params\n"
            f"  Fusion Classifier:        {fus_params:>10,} params\n"
            f"  {'-' * 42}\n"
            f"  Total:                    {total:>10,} params\n"
            f"  Classes: {self.CLASS_LABELS}\n"
            f"  Clinical features: {self.num_clinical_features}\n"
            f")"
        )
