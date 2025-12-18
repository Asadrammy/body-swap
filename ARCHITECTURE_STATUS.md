# Face & Body Swap Workflow – Architecture & Status

## 1. Client Requirements (from conversation)

| Area | Requirements |
| --- | --- |
| Inputs | Up to 2 customer reference photos (male/female/child, any body size) + selected template (pose, outfit, background). |
| Output Quality | Must preserve template pose, clothing, fabric folds, lighting, background, and facial expression. No “plastic” faces. Works for open-chest outfits, obese clients, couples, families, kids. |
| Automation | Fully automated workflow inside Runninghub GPU environment. CLI + REST API for batch jobs. |
| Reliability | High-resolution, photorealistic, seamless clothing fit, identity fidelity. Manual touch-up support + refinement masks. |
| Deliverables | Source/notebook, requirements/Docker, API/CLI entry points, sample test set (avg + obese). |
| Stack Expectations | Can use SD/ControlNet/LoRA, face-landmark warping, segmentation, etc., but must install smoothly on GPU instance. |

## 2. Current System Snapshot (repo inspection)

| Component | Implementation status |
| --- | --- |
| Modular pipeline (`src/pipeline`) | Preprocessor, BodyAnalyzer, TemplateAnalyzer, FaceProcessor, BodyWarper, Composer, Refiner, QualityControl modules scaffolded with logging & placeholder logic. |
| Models (`src/models`) | Face detector, pose detector, segmenter, generator wrappers created; generator loads SD + ControlNet pipelines. |
| Orchestration (`src/pipeline/__main__.py`) | End-to-end runner connecting modules with progress updates. |
| API (`src/api`) | FastAPI app with `/api/v1/swap`, job status, result, refine endpoints; CLI interface. |
| Frontend | Tailwind-based single-page uploader (templates/index.html + static/script.js) with dark mode toggle. |
| Configs | `configs/default.yaml`, `production.yaml`, `.env` template, requirements, Dockerfile, docker-compose. |
| Documentation | README + DEPLOYMENT instructions kept; redundant docs removed. |

## 3. Gap Analysis

| Requirement | Implemented? | Notes |
| --- | --- | --- |
| Template catalog + selection logic | Partial | Frontend provides UI but backend lacks template store & selection mapping. |
| Body-shape conditioned clothing warping | Scaffolded only | BodyWarper, Refiner contain placeholder logic; real models & training absent. |
| Multi-subject (couples, families) handling | Not built | Current pipeline assumes single subject. |
| High-res SD/ControlNet refinement w/ LoRA tuning | Partially prepared | Generator loads models, but no LoRA training, inpainting masks, or dynamic prompts. |
| QC metrics + manual touch-up masks | Placeholder | QualityControl returns dummy data; need real metrics and mask export. |
| Test set w/ average + obese before/after | Missing | Need curated examples and automation script. |
| Runninghub automation + GPU resource scripts | Not validated | Dockerfile exists but no Runninghub-specific setup/testing logs. |
| Pricing/site integration | Frontend concept only | No template catalog management, order tracking, or payment integration. |

## 4. Remaining Technical Work

1. **Data & Template Management**
   - Define template metadata (pose, clothing type, mask hints).
   - Build storage + API endpoints to fetch template info.

2. **Body & Clothing Adaptation**
   - Implement full body-shape extraction (SMPL/SMPL-X or MediaPipe+depth) and fabric retargeting (Thin Plate Spline / cloth simulation).
   - Handle open-chest, sleeveless, action poses with skin synthesis.

3. **Face Identity + Expression Matching**
   - Integrate high-quality identity embeddings (ArcFace) with expression transfer (face-landmark warping + ControlNet facial guidance).

4. **Generator Refinement**
   - Configure ControlNet pose/depth, LoRA fine-tunes, and inpainting masks for template-specific clothes.
   - GPU memory optimization + deterministic seeds for batch processing.

5. **Multi-subject Support**
   - Extend pipeline + API to process couples/family templates (multiple body meshes, occlusions).

6. **Quality Control & Touch-up Tools**
   - Implement similarity metrics, landmark deviation checks, and auto-generated masks for selective reprocessing.

7. **Runninghub Deployment**
   - Validate Docker/requirements on Runninghub GPU; document CLI/API run commands; include monitoring hooks.

8. **Sample Deliverables**
   - Produce test suite (avg + obese) with before/after images, saved under `examples/`.

## 5. Next Steps

1. Define milestone plan with client (template management, core swap engine, QC/testing).
2. Start with body-shape + clothing retargeting prototype using synthetic data.
3. Integrate real template set provided by client; add manual override hooks.
4. Iterate with client on sample outputs (including open-chest/action cases).
5. Finalize documentation (architecture, troubleshooting) aligned with Runninghub deployment.

This document should guide further development and client communication by clarifying which requirements are covered and what remains to deliver the promised workflow.

