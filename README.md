# Multimodal agent-enhanced diffusion Transformer for cultural relic image restoration

## Overview

This project implements a latent diffusion Transformer for two-dimensional cultural relic image restoration. The network combines a damaged image, a damage mask, a textual description, a structural sketch, and a material reference to reconstruct missing image regions. Cross-modal alignment and attribute-specific supervision support the recovery of shape, texture, and color appearance. Four functional agents coordinate condition encoding, candidate generation, quality assessment, and iterative parameter control.

The project includes data preparation, model training, image restoration, quantitative evaluation, baseline comparison, ablation experiments, and computational benchmarking.

## 1. Model architecture

### Multimodal condition encoding

| Condition | Encoder | Feature representation |
| --- | --- | --- |
| Text description | CLIP ViT-L/14 text encoder | 768-dimensional token features |
| Damaged image and mask | Variational autoencoder (VAE) | Four-channel latent features with an additional mask channel for conditioning |
| Structural sketch | ResNet-50 | 2048-dimensional spatial features before projection |
| Material reference | DINOv2 ViT-B/14 | 768-dimensional visual token features |

The condition encoders use frozen pretrained weights in the full configuration. Each modality is projected to 512 dimensions and subsequently mapped to the 1024-dimensional Transformer space. Modality identifiers and sequence position encodings distinguish the condition tokens. Noisy latent patches, a diffusion timestep token, and multimodal condition tokens form a joint sequence processed by multi-head self-attention.

The Transformer uses long skip connections between shallow and deep layers. A 3 × 3 convolution fuses the corresponding spatial feature maps. Three independent noise prediction heads share the Transformer backbone and provide shape, texture, and color predictions.

### Gaussian diffusion and training objective

The forward diffusion process follows a linear Gaussian noise schedule:

$$
\alpha_t=1-\beta_t,\qquad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s
$$

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
$$

Here, $x_0$ is the reference image latent, $x_t$ is the noisy latent at timestep $t$, and $\beta_t$ is the noise variance. Reverse sampling uses deterministic denoising diffusion implicit model (DDIM) updates.

The combined noise prediction is a weighted sum of the three branch predictions:

$$
\varepsilon_\theta=\sum_{i\in\{s,r,c\}}w_i\varepsilon_\theta^i,
\qquad w_i\geq0,\quad \sum_iw_i=1.
$$

The training objective combines noise prediction, cross-modal alignment, and branch supervision:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{simple}}+\eta_a\mathcal{L}_{\mathrm{align}}+\eta_b\mathcal{L}_{\mathrm{branch}}.
$$

Cross-modal alignment uses normalized pooled modality features, a shared feature center, and a contrastive objective. Samples with the same object identifier are excluded from the negative pairs. Attribute-specific supervision operates on decoded clean-latent estimates:

$$
\begin{aligned}
\Phi_s(I)&=\nabla(G_\sigma*Y(I)),\\
\Phi_r(I)&=Y(I)-G_\sigma*Y(I),\\
\Phi_c(I)&=G_\sigma*I.
\end{aligned}
$$

Here, $Y$ denotes luminance, $G_\sigma$ denotes Gaussian smoothing, and $\nabla$ denotes the spatial gradient. Each branch minimizes its corresponding masked attribute reconstruction error. The output image preserves observed pixels outside the damage mask:

$$
\hat I=M\odot D(\hat x_0)+(1-M)\odot I_d.
$$

### Agent-guided restoration

| Agent | Function |
| --- | --- |
| Condition agent | Encodes and organizes available multimodal conditions. |
| Generation agent | Generates candidate images through conditioned DDIM sampling. |
| Quality agent | Evaluates structural, texture, color, and text consistency using available input conditions. |
| Control agent | Adjusts branch fusion weights and condition guidance strength for subsequent candidates. |

Condition guidance retains the damaged image and mask in the base condition and adds auxiliary information through the guidance coefficient $\gamma$:

$$
\varepsilon^g=\varepsilon_\theta(x_t,t,c_b)+\gamma\left[\varepsilon_\theta(x_t,t,c)-\varepsilon_\theta(x_t,t,c_b)\right].
$$

For attribute score $q_i^{(k)}$ and threshold $\delta$, the controller applies the following updates:

$$
e_i^{(k)}=\max(0,\delta-q_i^{(k)}),\qquad
w_i^{(k+1)}=\frac{w_i^{(k)}\exp(\rho_w e_i^{(k)})}{\sum_j w_j^{(k)}\exp(\rho_w e_j^{(k)})}.
$$

$$
\gamma^{(k+1)}=\operatorname{clip}\left(\gamma^{(k)}+\rho_\gamma(\delta-q_{\mathrm{sem}}^{(k)}),\gamma_{\min},\gamma_{\max}\right).
$$

The quality agent averages available condition scores and retains the highest-scoring candidate. Reference target pixels are excluded from generation, feedback, and candidate selection. Fixed-round comparisons use the same noise seed sequence and sampling budget for feedback and fixed-parameter sampling.

## 2. Configuration

`configs/full.json` defines the primary model and experiment settings.

| Architecture parameter | Value |
| --- | --- |
| Image resolution | 512 × 512 |
| Latent dimensions | 4 × 64 × 64 |
| Latent patch size | 8 × 8 |
| Transformer layers | 12 |
| Attention heads | 16 |
| Hidden dimension | 1024 |
| Attention head dimension | 64 |
| Attention scaling denominator | 8 |
| Condition projection dimension | 512 |
| Timestep embedding dimension | 256 |
| Feed-forward expansion ratio | 4 |
| Dropout | 0.1 |
| Noise prediction branches | Shape, texture, color |

| Training parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 8 |
| Maximum epochs | 500 |
| Early stopping patience | 20 |
| Learning rate schedule | Cosine annealing |
| Gradient clipping norm | 1.0 |
| Alignment loss weight | 0.1 |
| Branch supervision loss weight | 0.2 |
| Contrastive loss weight | 0.1 |
| Contrastive temperature | 0.07 |
| Auxiliary condition dropout probability | 0.1 |
| Branch supervision weights (shape, texture, color) | 0.40, 0.35, 0.25 |
| Branch supervision timestep range | 0–499 (zero-based indexing) |
| Gaussian filter standard deviation | 2.0 |
| Training seed | 42 |

| Diffusion and inference parameter | Value |
| --- | --- |
| Forward diffusion timesteps | 1000 |
| Noise schedule | Linear |
| Initial noise variance | 1 × 10⁻⁴ |
| Final noise variance | 2 × 10⁻² |
| Reverse sampling method | DDIM |
| Sampling steps per round | 50 |
| Initial fusion weights (shape, texture, color) | 0.40, 0.35, 0.25 |
| Initial guidance strength | 5.0 |
| Guidance bounds | 1.0–8.0 |
| Attribute quality threshold | 0.90 |
| Branch weight update step | 0.5 |
| Guidance update step | 1.0 |
| Standard generation rounds | 3 |
| Fixed-round evaluation range | 1–5 |
| Evaluation seeds | 42, 123, 2025 |
| Inference batch size | 1 |
| Device and precision | CUDA, FP16 |
| Benchmark warm-up runs | 20 |
| Benchmark timing repetitions | 100 |

### Pretrained components

| Component | Configured checkpoint or weight identifier |
| --- | --- |
| CLIP | `openai/clip-vit-large-patch14` |
| VAE and Stable Diffusion 2.1 base | `sd2-community/stable-diffusion-2-1-base` |
| ResNet-50 | `IMAGENET1K_V2` |
| DINOv2 | `facebook/dinov2-base` |

The SD 2.1 and ControlNet training settings use a learning rate of 1 × 10⁻⁵ and a maximum of 100 epochs.

`configs/local.json` provides a separate compact CPU configuration for checking the training, inference, and evaluation interfaces with the bundled examples. Each run saves its own resolved configuration and hardware information.

## 3. Dataset and preprocessing

Dataset repository: <https://github.com/C3R8U/Heritage-Dataset>

The data pipeline reads image paths and object-level metadata from a CSV manifest. Supported metadata fields are `object_id`, `image`, `category`, `period`, `material`, `reference_status`, and `source`. The generated manifest adds the `split` field.

- Training, validation, and test partitions target a 70:15:15 object-level split, stratified by category.
- Images are converted to RGB and normalized to [0, 1].
- The damage mask covers 5%–50% of the image and uses random strokes and irregular patch regions; a mask value of 1 denotes the restoration region.
- Training augmentation includes random cropping, rotation within ±15°, saturation changes within ±20°, brightness changes within ±15°, contrast changes within ±20°, and Gaussian noise with standard deviation 0.05–0.30.
- Validation and test masks remain fixed for each object identifier and configuration seed.
- Text descriptions are assembled from category, material, and period metadata; structural sketches are extracted from observed image regions using Canny edge detection.
- Material references are selected from distinct training objects with matching category and material, with matching period prioritized.

The full-profile dataset loader accepts records with `complete` or `paired_complete` reference status for paired image evaluation. The catalog and referenced image files must be placed at the paths specified by the data manifest.

## 4. Installation and execution

Run all commands from the project root.

### Full configuration

Install the dependencies in a CUDA-enabled Python environment and make the configured pretrained weights accessible:

```bash
python -m pip install -r requirements.txt
```

Populate `data/catalog.csv` using the columns defined in `data/catalog_schema.csv`, then prepare the object-level partitions:

```bash
python -m heritage.prepare_manifest --catalog data/catalog.csv --out data/full_manifest.csv --seed 42
```

Launch training, evaluation, pretrained baseline runs, benchmarking, and report generation:

```bash
bash scripts/run_full.sh
```

### Local interface check

```bash
python -m pip install -r requirements-local.txt
bash scripts/run_local.sh
```

### Individual commands

Train and evaluate a selected configuration:

```bash
python -m heritage.runner --config configs/full.json --out runs/full
```

Run SD 2.1 and ControlNet training and evaluation after the main run creates validation scales:

```bash
python -m heritage.pretrained_baselines --config configs/full.json --out runs/pretrained --scales runs/full/validation_scales.json
```

Restore an example image using a saved run:

```bash
python -m heritage.infer --run runs/local --id SX-03410 --seed 42 --rounds 3 --out runs/example
```

Evaluate computational resources and generate reports:

```bash
python -m heritage.benchmark --run runs/full --variant proposed --rounds 3
python -m heritage.report --run runs/full
```

Evaluate external mural restoration predictions using a CSV file with `id`, `seed`, and `image` columns:

```bash
python -m heritage.external_eval --config configs/full.json --predictions data/domain_predictions.csv --scales runs/full/validation_scales.json --out runs/domain_evaluation
```

Run pipeline tests:

```bash
python -m pytest -q
```

## 5. Experiments and evaluation

The internal comparison includes the proposed network, a U-Net-style diffusion network, DDPM-CNN, and a DiT-style network. Module ablations remove auxiliary modalities, cross-modal alignment, or attribute-specific supervision, and replace the Transformer backbone with a parameter-matched convolutional alternative. Branch ablations compare shared noise supervision with attribute-specific losses and remove individual attribute losses. Paired sampling compares fixed parameters and adaptive feedback under equal sampling budgets.

The evaluation module computes masked mean squared error (MSE) and peak signal-to-noise ratio (PSNR) from the same accumulated squared errors and valid RGB pixel counts:

$$
\mathrm{MSE}=\frac{\sum M\odot(I-\hat I)^2}{3\sum M},\qquad
\mathrm{PSNR}=10\log_{10}(1/\mathrm{MSE}).
$$

Structural similarity (SSIM) uses Gaussian windows with standard deviation 1.5. Learned perceptual image patch similarity (LPIPS) uses AlexNet features in the full configuration. Semantic consistency uses normalized CLIP image-text cosine similarity. Shape similarity uses Canny edge matching with thresholds 50 and 150 and a two-pixel matching neighborhood; texture and color similarity use masked attribute errors with validation-calibrated scales. Available attribute scores are averaged for overall feature similarity.

Resource evaluation records parameter counts, serialized inference weight size, step-level denoising latency, full restoration latency, and GPU peak memory when using CUDA.

## 6. File descriptions

### Configuration and entry points

| File | Description |
| --- | --- |
| `README.md` | Describes the model, configuration, execution commands, and project files. |
| `EXECUTION_NOTES.txt` | Records profile details, data handling, execution procedures, and output conventions. |
| `configs/full.json` | Defines the primary model architecture, training parameters, pretrained components, and evaluation settings. |
| `configs/local.json` | Defines the compact CPU interface-check configuration and example manifest. |
| `requirements.txt` | Lists dependencies for the full configuration. |
| `requirements-local.txt` | Lists dependencies for local CPU execution. |
| `scripts/run_full.sh` | Runs the full training, baseline evaluation, benchmark, and reporting workflow. |
| `scripts/run_local.sh` | Runs the local training, evaluation, benchmarking, tests, and reporting workflow. |
| `tests/test_pipeline.py` | Tests diffusion identities, mask validity, data isolation, feedback weights, metric consistency, and target exclusion. |

### Source modules

| File | Description |
| --- | --- |
| `heritage/__init__.py` | Initializes the heritage restoration Python package. |
| `heritage/model.py` | Defines condition encoders, latent codecs, Transformer and convolutional backbones, and noise prediction heads. |
| `heritage/diffusion.py` | Implements the Gaussian noise schedule, DDIM sampling, alignment loss, and attribute-specific training objective. |
| `heritage/agents.py` | Implements the four functional agents, candidate selection, and feedback parameter updates. |
| `heritage/data.py` | Loads manifests and images, validates partitions, applies augmentation, constructs masks, and selects reference conditions. |
| `heritage/metrics.py` | Implements image quality, attribute similarity, semantic consistency, and metric aggregation. |
| `heritage/runner.py` | Coordinates training, validation, checkpoint selection, ablations, and paired sampling evaluation. |
| `heritage/benchmark.py` | Measures inference latency, peak GPU memory, parameter counts, and inference weight size. |
| `heritage/pretrained_baselines.py` | Defines task adaptation and evaluation procedures for SD 2.1 and ControlNet. |
| `heritage/external_eval.py` | Evaluates externally generated mural predictions using the shared metric implementation. |
| `heritage/prepare_manifest.py` | Creates category-stratified, object-disjoint dataset partitions from an input catalog. |
| `heritage/infer.py` | Loads a saved model and generates a restored image for a selected sample identifier. |
| `heritage/report.py` | Collects experiment summaries and assembles comparison panels from saved predictions. |
| `heritage/common.py` | Provides configuration loading, random seed control, device selection, timing, and file-writing utilities. |

### Data files

| File | Description |
| --- | --- |
| `data/catalog_schema.csv` | Defines the metadata columns required by the manifest preparation command. |
| `data/examples.csv` | Stores the paths, metadata, and partitions of the bundled example images. |
| `data/external_prediction_schema.csv` | Defines the identifier, seed, and image path columns used for external prediction evaluation. |
| `data/files_sha256.csv` | Stores SHA-256 checksums for the bundled image files. |
| `data/images/*.jpg` | Contains the individual example images referenced by `data/examples.csv`. |

### Run-level files

The files below are organized under `runs/local/` for the bundled local run; the same naming convention is used for the corresponding full-run outputs.

| File | Description |
| --- | --- |
| `configuration.json` | Stores the resolved configuration used for the run. |
| `hardware.json` | Records the execution platform, Python and PyTorch versions, and device information. |
| `dependency_versions.json` | Records installed software dependency versions. |
| `dataset_counts.csv` | Records image counts by partition and category. |
| `validation_scales.json` | Stores the validation-derived normalization scales shared across comparison variants. |
| `all_results.csv` | Collects aggregated evaluation metrics across model variants and sampling strategies. |
| `resources.csv` | Stores model parameter counts, trainable parameter counts, and serialized weight sizes. |
| `run_status.json` | Records execution status for experiment components. |
| `runs/local_console.log` | Contains console output from local training and evaluation. |
| `runs/tests.log` | Contains the pipeline test output. |

### Model variant directories

Each variant directory contains the corresponding checkpoint, training log, evaluation outputs, and saved images.

| Directory | Description |
| --- | --- |
| `proposed/` | Contains the complete multimodal Transformer and adaptive feedback experiments. |
| `unet/` | Contains the independent U-Net-style diffusion baseline. |
| `ddpm_cnn/` | Contains the convolutional residual diffusion baseline. |
| `dit/` | Contains the single-head DiT-style baseline without long skip fusion. |
| `no_text/` | Contains the text-condition ablation. |
| `no_shape/` | Contains the structural-condition ablation. |
| `no_material/` | Contains the material-condition ablation. |
| `image_only/` | Contains the damaged-image-and-mask-only condition experiment. |
| `no_align/` | Contains the cross-modal alignment loss ablation. |
| `no_branch/` | Contains the attribute-specific supervision loss ablation. |
| `unet_backbone/` | Contains the U-Net backbone replacement experiment with the other model components retained. |
| `same_noise/` | Contains the three-branch common-noise supervision experiment. |
| `no_shape_loss/` | Contains the shape supervision ablation. |
| `no_texture_loss/` | Contains the texture supervision ablation. |
| `no_color_loss/` | Contains the color supervision ablation. |

### Files within model variant directories

| File | Description |
| --- | --- |
| `training_log.csv` | Records epoch-level training loss, validation loss, loss components, learning rate, and duration. |
| `best.pt` | Stores the checkpoint selected by validation loss. |
| `weights.pt` | Stores inference weights in the configured storage precision. |
| `per_image_metrics.csv` | Stores image-level evaluation metrics for each seed, strategy, and generation round. |
| `summary.csv` | Stores aggregated metrics and seed-level variability for each sampling configuration. |
| `category_metrics.csv` | Stores evaluation summaries grouped by cultural relic category. |
| `agent_trace.json` | Records candidate scores, guidance strengths, branch weights, and timing information. |
| `benchmark_<rounds>.json` | Stores repeated inference benchmark measurements for the specified generation budget. |
| `images/<id>/reference.png` | Stores the reference image used for paired evaluation. |
| `images/<id>/input.png` | Stores the damaged input presented to the restoration network. |
| `images/<id>/mask.png` | Stores the binary damage region mask. |
| `images/<id>/sketch.png` | Stores the structural condition extracted from the observed image. |
| `images/<id>/material.png` | Stores the material reference condition. |
| `images/<id>/fixed_<seed>_<round>.png` | Stores the selected output from fixed-parameter candidate sampling. |
| `images/<id>/feedback_<seed>_<round>.png` | Stores the selected output from adaptive feedback sampling. |

### Reports and example inference

| File | Description |
| --- | --- |
| `reports/general_comparison.csv` | Collects the internal baseline and proposed-model comparison metrics. |
| `reports/module_ablation.csv` | Collects multimodal condition, loss, and backbone ablation metrics. |
| `reports/branch_ablation.csv` | Collects common-noise and attribute-specific branch ablation metrics. |
| `reports/paired_sampling.csv` | Collects fixed-parameter and adaptive feedback sampling comparisons. |
| `reports/category_comparison.csv` | Collects category-level metrics for the three-round feedback configuration. |
| `reports/visual_comparison.png` | Presents a comparison panel assembled from the saved model predictions. |
| `runs/example/input.png` | Stores the input image for the individual inference example. |
| `runs/example/restored.png` | Stores the selected restored image for the individual inference example. |
| `runs/example/agent_trace.json` | Records the feedback trajectory for the individual inference example. |
