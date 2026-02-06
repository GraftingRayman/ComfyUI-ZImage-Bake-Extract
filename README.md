ZImage LoRA Manager Advanced
A ComfyUI custom node for baking LoRAs into base models and extracting LoRAs from merged/finetuned models. This node provides advanced functionality for managing LoRA weights with support for format conversion between sd-scripts and ComfyUI formats.

Features
Bake LoRA into Base Model: Merge LoRA weights into a base model to create a single, standalone model

Extract LoRA from Merged Model: Reverse the baking process to extract LoRA weights from a finetuned/merged model

Format Conversion: Automatic conversion between sd-scripts and ComfyUI LoRA formats

Alpha/Strength Control: Adjust the strength of LoRA application during baking

Rank Control: Specify rank during LoRA extraction

Duplicate Protection: Automatic filename management to prevent overwriting

Installation
Place the zimagebakeloraadvanced.py and __init__.py files in your ComfyUI custom nodes directory:

text
ComfyUI/custom_nodes/zimage_lora_manager/
├── __init__.py
└── zimagebakeloraadvanced.py
Restart ComfyUI or refresh your browser

Usage
Inputs
Parameter	Type	Description	Required For
mode	dropdown	Operation mode: "bake" or "extract"	Always
base_model	dropdown	Base model selection (from models/diffusion_models/)	Both modes
output_name	string	Name for the output file (without extension)	Always
rank	integer	Rank for LoRA extraction (1-32, default: 4)	Extract only
alpha	float	Strength multiplier for baking/extraction (0.0-2.0, default: 1.0)	Both modes
lora_model	dropdown	LoRA to bake (from models/loras/)	Bake only
merged_model	dropdown	Merged/finetuned model for extraction (from models/diffusion_models/)	Extract only
convert_to_comfy_format	boolean	Convert extracted LoRA to ComfyUI format (default: True)	Extract only
Outputs
Output	Type	Description
status	string	Detailed status message with operation results
Operation Modes
1. Bake Mode
Merges a LoRA into a base model:

Select "bake" from the mode dropdown

Choose a base model

Choose a LoRA model

Set output name and alpha strength

Output will be saved to ComfyUI/output/ as a .safetensors file

2. Extract Mode
Extracts LoRA weights from a merged/finetuned model:

Select "extract" from the mode dropdown

Choose the original base model

Choose the merged/finetuned model

Set output name, rank, and alpha

Optionally convert to ComfyUI format

Output will be saved to ComfyUI/output/ as a .safetensors file

File Structure
The node expects the following directory structure:

text
ComfyUI/
├── models/
│   ├── diffusion_models/     # Base and merged models
│   └── loras/                # LoRA models
└── output/                   # Output directory (created automatically)
Format Support
Supported Input Formats:
.safetensors files

.ckpt files

LoRA Format Conversion:
sd-scripts format: Separate Q, K, V attention layers

ComfyUI format: Combined QKV attention layers

Automatic detection and conversion during operations

Technical Details
Baking Process:
Load base model and LoRA weights

Detect LoRA format (convert sd-scripts format if needed)

Apply LoRA weights to base model with scaling: W' = W + BA * (alpha/rank)

Save merged model

Extraction Process:
Load base model and merged model

Compute weight differences

Perform SVD to extract LoRA components

Keep top rank components

Optionally convert to ComfyUI format

Save extracted LoRA

Error Handling
The node includes comprehensive error handling with detailed logging:

Missing file warnings

Format detection issues

Tensor operations errors

File I/O errors

All errors are logged to console and returned as status messages.

Examples
Example 1: Baking a LoRA
text
mode: bake
base_model: stable-diffusion-xl-base-1.0.safetensors
lora_model: my_style_lora.safetensors
output_name: sd-xl-with-my-style
alpha: 0.8
Result: Creates sd-xl-with-my-style.safetensors in output folder

Example 2: Extracting a LoRA
text
mode: extract
base_model: stable-diffusion-xl-base-1.0.safetensors
merged_model: finetuned-model.safetensors
output_name: extracted-changes
rank: 8
alpha: 1.0
convert_to_comfy_format: True
Result: Creates extracted-changes.safetensors in output folder

Troubleshooting
Common Issues:
"No models found" in dropdown:

Ensure models are placed in ComfyUI/models/diffusion_models/

Check file extensions (.safetensors or .ckpt)

"No LoRAs found" in dropdown:

Ensure LoRAs are placed in ComfyUI/models/loras/

Permission errors:

Check write permissions for output directory

Memory errors:

Ensure sufficient RAM/VRAM for model operations

Logs:
Check ComfyUI console for detailed operation logs

Look for [ZImage LoRA Manager] prefix in logs
