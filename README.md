# ZImage LoRA Manager Advanced
A ComfyUI custom node for baking LoRAs into base models and extracting LoRAs from merged/finetuned models. This node provides advanced functionality for managing LoRA weights with support for format conversion between sd-scripts and ComfyUI formats.

# Features
Bake LoRA into Base Model: Merge LoRA weights into a base model to create a single, standalone model

Extract LoRA from Merged Model: Reverse the baking process to extract LoRA weights from a finetuned/merged model

Format Conversion: Automatic conversion between sd-scripts and ComfyUI LoRA formats

Alpha/Strength Control: Adjust the strength of LoRA application during baking

Rank Control: Specify rank during LoRA extraction

Duplicate Protection: Automatic filename management to prevent overwriting

<img width="1024" height="816" alt="image" src="https://github.com/user-attachments/assets/4f0727b9-2932-4c33-9917-a3852ea6eed6" />


PROMPT: 

A close-up portrait of a young woman with a neutral expression, looking directly at the camera with a slight smile. she has long, wavy, dark brown hair and large, expressive brown eyes. her hair is styled in loose waves and falls down her back, framing her face. she is wearing a red sweater and gold earrings, which add a touch of elegance to her overall look. the background is blurred, focusing attention on her face and upper body, with a soft, warm color palette that complements her natural beauty. the lighting is soft and natural, highlighting her features and creating a serene atmosphere. the image has a high-quality, professional look with a focus on natural beauty and elegance.


<img width="2691" height="784" alt="image" src="https://github.com/user-attachments/assets/1637a050-9830-4d36-b644-ec580f92a872" />



# Installation
Place the zimagebakeloraadvanced.py and __init__.py files in your ComfyUI custom nodes directory:


ComfyUI/custom_nodes/zimage_lora_manager/

├── __init__.py

└── zimagebakeloraadvanced.py

Restart ComfyUI or refresh your browser


# Usage
## Inputs
Parameter	Type	Description	Required For
mode	dropdown	Operation mode: "bake" or "extract"	Always
base_model	dropdown	Base model selection (from models/diffusion_models/)	Both modes
output_name	string	Name for the output file (without extension)	Always
rank	integer	Rank for LoRA extraction (1-32, default: 4)	Extract only
alpha	float	Strength multiplier for baking/extraction (0.0-2.0, default: 1.0)	Both modes
lora_model	dropdown	LoRA to bake (from models/loras/)	Bake only
merged_model	dropdown	Merged/finetuned model for extraction (from models/diffusion_models/)	Extract only
convert_to_comfy_format	boolean	Convert extracted LoRA to ComfyUI format (default: True)	Extract only

## Outputs
Output	Type	Description
status	string	Detailed status message with operation results


#Operation Modes
#1. Bake Mode
Merges a LoRA into a base model:

Select "bake" from the mode dropdown

Choose a base model

Choose a LoRA model

Set output name and alpha strength

Output will be saved to ComfyUI/output/ as a .safetensors file

#2. Extract Mode
Extracts LoRA weights from a merged/finetuned model:

Select "extract" from the mode dropdown

Choose the original base model

Choose the merged/finetuned model

Set output name, rank, and alpha

Optionally convert to ComfyUI format

Output will be saved to ComfyUI/output/ as a .safetensors file

#File Structure
The node expects the following directory structure:

ComfyUI/
├── models/
│   ├── diffusion_models/     # Base and merged models
│   └── loras/                # LoRA models
└── output/                   # Output directory (created automatically)

# Format Support
##Supported Input Formats:
.safetensors files


