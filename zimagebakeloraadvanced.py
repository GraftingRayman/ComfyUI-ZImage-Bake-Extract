import torch
import re
import os
import time
import numpy as np
from safetensors.torch import load_file, save_file
import folder_paths
import traceback

class ZImageLoraManagerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        # Function to recursively find all .safetensors files
        def find_safetensors_files(root_dir):
            safetensors_files = []
            if not os.path.exists(root_dir):
                return safetensors_files
                
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('.safetensors', '.ckpt')):
                        # Get relative path from root_dir
                        rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                        safetensors_files.append(rel_path)
            return sorted(safetensors_files)
        
        # Get base ComfyUI directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define paths
        model_base_path = os.path.join(base_dir, "models", "diffusion_models")
        lora_base_path = os.path.join(base_dir, "models", "loras")
        
        # Create directories if they don't exist
        os.makedirs(model_base_path, exist_ok=True)
        os.makedirs(lora_base_path, exist_ok=True)
        
        # Find all model files
        model_files = find_safetensors_files(model_base_path)
        if not model_files:
            model_files = ["No models found"]
        
        # Find all LoRA files
        lora_files = find_safetensors_files(lora_base_path)
        if not lora_files:
            lora_files = ["No LoRAs found"]
        
        return {
            "required": {
                "mode": (["bake", "extract"], {"default": "bake"}),
                "base_model": (["None"] + model_files, {"default": "None"}),
                "output_name": ("STRING", {"default": "output"}),
                "rank": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "lora_model": (["None"] + lora_files, {"default": "None"}),
                "merged_model": (["None"] + model_files, {"default": "None"}),
                "convert_to_comfy_format": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process_lora"
    CATEGORY = "ZImage"
    
    def __init__(self):
        pass
    
    def log(self, message):
        """Log message to console"""
        print(f"[ZImage LoRA Manager] {message}")
    
    def load_weights(self, path):
        """Load weights from file"""
        self.log(f"Loading weights from: {os.path.basename(path)}")
        try:
            if path.endswith('.safetensors'):
                return load_file(path, device='cpu')
            else:  # .ckpt file
                weights = torch.load(path, map_location='cpu')
                if 'state_dict' in weights:
                    return weights['state_dict']
                return weights
        except Exception as e:
            raise Exception(f"Error loading {path}: {e}")
    
    def save_weights(self, weights, path):
        """Save weights to file with proper tensor contiguity"""
        self.log(f"Saving weights to: {os.path.basename(path)}")
        try:
            # Ensure all tensors are contiguous before saving
            contiguous_weights = {}
            for key, value in weights.items():
                if isinstance(value, torch.Tensor):
                    # Make tensor contiguous if it isn't already
                    contiguous_weights[key] = value.contiguous()
                else:
                    contiguous_weights[key] = value
            
            if path.endswith('.safetensors'):
                save_file(contiguous_weights, path)
            else:
                torch.save(contiguous_weights, path)
            return True
        except Exception as e:
            raise Exception(f"Error saving {path}: {e}")
    
    def get_full_path(self, filename, base_path):
        """Convert filename to full path"""
        if filename == "None":
            return None
        return os.path.join(base_path, filename)
    
    def is_lora_converted_to_comfy(self, state_dict):
        """Check if LoRA is already in ComfyUI format (converted)"""
        keys = list(state_dict.keys())
        
        # Check for ComfyUI specific patterns (converted format)
        comfyui_patterns = [
            "attention_q_norm",
            "attention_k_norm", 
            "attention_out",
            "attention_qkv"  # QKV combined layers are ComfyUI format
        ]
        
        # Check for sd-scripts patterns (original format)
        sd_patterns = [
            "attention_norm_q",
            "attention_norm_k",
            "attention_to_out_0",
        ]
        
        # Look for definitive patterns first
        has_qkv = any("attention_qkv" in key for key in keys)
        has_separate_q = any("to_q.lora_down" in key for key in keys)
        has_separate_k = any("to_k.lora_down" in key for key in keys)
        
        if has_qkv:
            return True  # Has combined QKV = ComfyUI format
        if has_separate_q and has_separate_k:
            return False  # Has separate Q/K = sd-scripts format
        
        # Fallback to checking other patterns
        has_comfyui_keys = any(any(pattern in key for pattern in comfyui_patterns) for key in keys)
        has_sd_keys = any(any(pattern in key for pattern in sd_patterns) for key in keys)
        
        if has_comfyui_keys and not has_sd_keys:
            return True  # Appears to be ComfyUI format
        elif has_sd_keys and not has_comfyui_keys:
            return False  # Appears to be sd-scripts format
        
        return None  # Mixed or unclear
    
    def convert_lora_format(self, state_dict, to_comfy=True):
        """Convert LoRA between sd-scripts and ComfyUI formats"""
        if to_comfy:
            self.log("Converting LoRA from sd-scripts to ComfyUI format...")
        else:
            self.log("Converting LoRA from ComfyUI to sd-scripts format...")
        
        new_state_dict = state_dict.copy()
        count = 0
        qkv_count = 0
        
        # Key mapping tables: (sd-scripts format, ComfyUI format)
        blocks_mappings = [
            ("attention_to_out_0", "attention_out"),
            ("attention_norm_k", "attention_k_norm"),
            ("attention_norm_q", "attention_q_norm"),
        ]
        
        keys = list(new_state_dict.keys())
        
        for key in keys:
            new_k = key
            
            # Apply key mappings based on conversion direction
            for src_key, dst_key in blocks_mappings:
                if to_comfy:
                    # sd-scripts to ComfyUI: replace src with dst
                    new_k = new_k.replace(src_key, dst_key)
                else:
                    # ComfyUI to sd-scripts: replace dst with src
                    new_k = new_k.replace(dst_key, src_key)
            
            if new_k != key:
                new_state_dict[new_k] = new_state_dict.pop(key)
                count += 1
        
        # Handle QKV layers - this is the main conversion logic
        if to_comfy:
            # Convert from separate Q/K/V to combined QKV (sd-scripts → ComfyUI)
            keys = list(new_state_dict.keys())
            processed_prefixes = set()
            
            for key in keys:
                if "attention" in key and "to_q" in key and "lora_up" in key:
                    # Extract the prefix before "to_q"
                    # Example: "lora_unet_down_blocks_0_attentions_0_to_q" -> "lora_unet_down_blocks_0_attentions_0_"
                    lora_name = key.split(".", 1)[0]
                    prefix = lora_name.replace("to_q", "")
                    
                    if prefix in processed_prefixes:
                        continue
                    
                    processed_prefixes.add(prefix)
                    
                    # Check if we have all three Q, K, V layers
                    has_q = f"{prefix}to_q.lora_down.weight" in new_state_dict
                    has_k = f"{prefix}to_k.lora_down.weight" in new_state_dict
                    has_v = f"{prefix}to_v.lora_down.weight" in new_state_dict
                    
                    if has_q and has_k and has_v:
                        # Get all weights
                        q_down = new_state_dict.pop(f"{prefix}to_q.lora_down.weight")
                        q_up = new_state_dict.pop(f"{prefix}to_q.lora_up.weight")
                        q_alpha = new_state_dict.pop(f"{prefix}to_q.alpha")
                        
                        k_down = new_state_dict.pop(f"{prefix}to_k.lora_down.weight")
                        k_up = new_state_dict.pop(f"{prefix}to_k.lora_up.weight")
                        k_alpha = new_state_dict.pop(f"{prefix}to_k.alpha")
                        
                        v_down = new_state_dict.pop(f"{prefix}to_v.lora_down.weight")
                        v_up = new_state_dict.pop(f"{prefix}to_v.lora_up.weight")
                        v_alpha = new_state_dict.pop(f"{prefix}to_v.alpha")
                        
                        # Concatenate down weights
                        down_weight = torch.cat([q_down, k_down, v_down], dim=0)
                        
                        # Create block diagonal up weight matrix
                        rank = q_up.shape[1]
                        out_dim_q = q_up.shape[0]
                        out_dim_k = k_up.shape[0]
                        out_dim_v = v_up.shape[0]
                        total_out_dim = out_dim_q + out_dim_k + out_dim_v
                        
                        up_weight = torch.zeros((total_out_dim, rank * 3), 
                                               device=down_weight.device, 
                                               dtype=down_weight.dtype)
                        
                        # Fill in the block diagonal
                        up_weight[:out_dim_q, :rank] = q_up
                        up_weight[out_dim_q:out_dim_q+out_dim_k, rank:rank*2] = k_up
                        up_weight[out_dim_q+out_dim_k:, rank*2:] = v_up
                        
                        # New alpha is 3x because rank is 3x larger
                        new_alpha = q_alpha * 3
                        
                        # Save as QKV combined
                        new_state_dict[f"{prefix}qkv.lora_down.weight"] = down_weight.contiguous()
                        new_state_dict[f"{prefix}qkv.lora_up.weight"] = up_weight.contiguous()
                        new_state_dict[f"{prefix}qkv.alpha"] = new_alpha
                        
                        qkv_count += 1
                        self.log(f"  ✓ Combined Q/K/V layers for {prefix}")
        
        else:
            # Convert from combined QKV to separate Q/K/V (ComfyUI → sd-scripts)
            keys = list(new_state_dict.keys())
            
            for key in keys:
                if "attention_qkv" in key and "lora_down" in key:
                    lora_name = key.split(".", 1)[0]
                    prefix = lora_name.replace("qkv", "")
                    
                    # Get combined weights
                    down_weight = new_state_dict.pop(f"{prefix}qkv.lora_down.weight")
                    up_weight = new_state_dict.pop(f"{prefix}qkv.lora_up.weight")
                    alpha = new_state_dict.pop(f"{prefix}qkv.alpha")
                    
                    # Split down weight into 3 equal parts
                    split_dims = [down_weight.size(0) // 3] * 3
                    down_weights = torch.chunk(down_weight, 3, dim=0)
                    
                    # Split up weight
                    out_dims = [up_weight.size(0) // 3] * 3
                    rank = up_weight.size(1) // 3
                    
                    q_up = up_weight[:out_dims[0], :rank]
                    k_up = up_weight[out_dims[0]:out_dims[0]+out_dims[1], rank:rank*2]
                    v_up = up_weight[out_dims[0]+out_dims[1]:, rank*2:]
                    
                    # New alpha is 1/3 because rank is 3x smaller
                    new_alpha = alpha / 3
                    
                    # Save as separate Q, K, V
                    new_state_dict[f"{prefix}to_q.lora_down.weight"] = down_weights[0].contiguous()
                    new_state_dict[f"{prefix}to_q.lora_up.weight"] = q_up.contiguous()
                    new_state_dict[f"{prefix}to_q.alpha"] = new_alpha
                    
                    new_state_dict[f"{prefix}to_k.lora_down.weight"] = down_weights[1].contiguous()
                    new_state_dict[f"{prefix}to_k.lora_up.weight"] = k_up.contiguous()
                    new_state_dict[f"{prefix}to_k.alpha"] = new_alpha
                    
                    new_state_dict[f"{prefix}to_v.lora_down.weight"] = down_weights[2].contiguous()
                    new_state_dict[f"{prefix}to_v.lora_up.weight"] = v_up.contiguous()
                    new_state_dict[f"{prefix}to_v.alpha"] = new_alpha
                    
                    qkv_count += 1
                    self.log(f"  ✓ Split QKV layer for {prefix}")
        
        self.log(f"  Key renames applied: {count}")
        self.log(f"  QKV layers processed: {qkv_count}")
        return new_state_dict
    
    def bake_lora(self, base_path, lora_path, output_path, alpha=1.0):
        """Bake LoRA into base model - ORIGINAL WORKING LOGIC - DO NOT CHANGE"""
        messages = []
        
        try:
            # Log start
            start_time = time.time()
            self.log("=" * 60)
            self.log("STARTING LoRA BAKING")
            self.log("=" * 60)
            self.log(f"Base model: {os.path.basename(base_path)}")
            self.log(f"LoRA: {os.path.basename(lora_path)}")
            self.log(f"Output: {os.path.basename(output_path)}")
            self.log(f"Alpha: {alpha}")
            
            # Load weights
            self.log("Loading base model weights...")
            base_weights = self.load_weights(base_path)
            self.log(f"✓ Loaded {len(base_weights)} weights from base model")
            
            self.log("Loading LoRA weights...")
            lora_weights = self.load_weights(lora_path)
            self.log(f"✓ Loaded {len(lora_weights)} LoRA weights")
            
            # Check if LoRA is in ComfyUI format and convert back if needed
            is_comfy_format = self.is_lora_converted_to_comfy(lora_weights)
            if is_comfy_format:
                self.log("⚠️ LoRA is in ComfyUI format - converting back to sd-scripts format for baking...")
                lora_weights = self.convert_lora_format(lora_weights, to_comfy=False)
                self.log("✓ Converted LoRA to sd-scripts format for baking")
            elif is_comfy_format is False:
                self.log("✓ LoRA is already in sd-scripts format (no conversion needed)")
            else:
                self.log("⚠️ Could not determine LoRA format - proceeding with baking")
            
            # Track merged weights
            merged_weights = {}
            
            # Copy all base weights first
            self.log("Copying base weights...")
            base_keys = list(base_weights.keys())
            for key in base_keys:
                merged_weights[key] = base_weights[key].clone()
            
            self.log(f"✓ Copied all {len(base_keys)} base weights")
            
            # Process LoRA weights by type
            processed = set()
            processed_count = 0
            
            # 1. First handle feed-forward layers
            self.log("Processing feed-forward layers...")
            ff_layers = 0
            for key in lora_weights.keys():
                if 'feed_forward_w' in key and 'lora_up' in key:
                    # Get the prefix (e.g., "lora_unet_layers_0_feed_forward_w1")
                    prefix = key.replace('.lora_up.weight', '')
                    
                    # Get layer number and w number
                    match = re.search(r'layers_(\d+)_feed_forward_w([123])', prefix)
                    if not match:
                        continue
                        
                    layer_num = match.group(1)
                    w_num = match.group(2)
                    
                    # Build base weight name
                    base_key = f'layers.{layer_num}.feed_forward.w{w_num}.weight'
                    
                    if base_key not in base_weights:
                        self.log(f"  Warning: Base key {base_key} not found. Skipping.")
                        continue
                    
                    # Get LoRA matrices
                    down_key = f'{prefix}.lora_down.weight'
                    up_key = f'{prefix}.lora_up.weight'
                    alpha_key = f'{prefix}.alpha'
                    
                    if down_key not in lora_weights or up_key not in lora_weights:
                        self.log(f"  Warning: Missing LoRA pair for {prefix}")
                        continue
                    
                    lora_down = lora_weights[down_key].float()
                    lora_up = lora_weights[up_key].float()
                    
                    # Get scaling
                    if alpha_key in lora_weights:
                        lora_alpha = lora_weights[alpha_key].item()
                    else:
                        lora_alpha = 1.0
                    
                    rank = lora_down.shape[0]
                    scaling = lora_alpha / rank
                    
                    # Merge: W' = W + BA * (alpha/rank)
                    lora_matrix = lora_up @ lora_down
                    merged_weights[base_key] = base_weights[base_key].float() + lora_matrix * scaling * alpha
                    
                    processed.add(down_key)
                    processed.add(up_key)
                    if alpha_key in lora_weights:
                        processed.add(alpha_key)
                    
                    processed_count += 1
                    ff_layers += 1
                    self.log(f"  ✓ Baked FF layer {layer_num}.w{w_num} (rank: {rank}, alpha: {lora_alpha})")
            
            self.log(f"✓ Processed {ff_layers} feed-forward layers")
            
            # 2. Handle attention layers
            self.log("Processing attention layers...")
            attn_layers = 0
            
            # Process attention QKV layers
            for key in lora_weights.keys():
                if 'attention_to_' in key and 'lora_up' in key and 'out' not in key:
                    prefix = key.replace('.lora_up.weight', '')
                    
                    # Parse: lora_unet_layers_0_attention_to_q
                    match = re.search(r'layers_(\d+)_attention_to_([qkv])', prefix)
                    if not match:
                        continue
                        
                    layer_num = match.group(1)
                    attn_type = match.group(2)
                    
                    # Skip if already processed
                    down_key = f'{prefix}.lora_down.weight'
                    up_key = f'{prefix}.lora_up.weight'
                    alpha_key = f'{prefix}.alpha'
                    
                    if down_key in processed:
                        continue
                    
                    # For Q, K, V - Base key is combined QKV
                    base_key = f'layers.{layer_num}.attention.qkv.weight'
                    
                    if base_key not in base_weights:
                        self.log(f"  Warning: Base key {base_key} not found. Skipping.")
                        continue
                    
                    # Initialize merged QKV if not done
                    if f'qkv_merged_{layer_num}' not in merged_weights:
                        merged_weights[f'qkv_merged_{layer_num}'] = base_weights[base_key].float().clone()
                    
                    # Get LoRA matrices
                    lora_down = lora_weights[down_key].float()
                    lora_up = lora_weights[up_key].float()
                    
                    # Get scaling
                    if alpha_key in lora_weights:
                        lora_alpha = lora_weights[alpha_key].item()
                    else:
                        lora_alpha = 1.0
                    
                    rank = lora_down.shape[0]
                    scaling = lora_alpha / rank
                    
                    # Create LoRA matrix
                    lora_matrix = lora_up @ lora_down
                    
                    # Apply to appropriate section of QKV
                    embed_dim = base_weights[base_key].shape[1]
                    section_size = embed_dim
                    
                    # Get current merged QKV
                    current_qkv = merged_weights[f'qkv_merged_{layer_num}']
                    
                    # Split into Q, K, V sections
                    q_section = current_qkv[:section_size, :]
                    k_section = current_qkv[section_size:section_size*2, :]
                    v_section = current_qkv[section_size*2:, :]
                    
                    # Apply to correct section
                    if attn_type == 'q':
                        q_section += lora_matrix * scaling * alpha
                        attn_type_str = "Q"
                    elif attn_type == 'k':
                        k_section += lora_matrix * scaling * alpha
                        attn_type_str = "K"
                    elif attn_type == 'v':
                        v_section += lora_matrix * scaling * alpha
                        attn_type_str = "V"
                    
                    # Recombine
                    merged_weights[f'qkv_merged_{layer_num}'] = torch.cat([q_section, k_section, v_section], dim=0)
                    
                    processed.add(down_key)
                    processed.add(up_key)
                    if alpha_key in lora_weights:
                        processed.add(alpha_key)
                    
                    processed_count += 1
                    attn_layers += 1
                    self.log(f"  ✓ Applied LoRA to layer {layer_num} {attn_type_str}")
            
            # Process attention output layers
            for key in lora_weights.keys():
                if 'attention_to_out' in key and 'lora_up' in key:
                    prefix = key.replace('.lora_up.weight', '')
                    
                    # Parse: lora_unet_layers_0_attention_to_out_0
                    match = re.search(r'layers_(\d+)_attention_to_out_0', prefix)
                    if not match:
                        continue
                        
                    layer_num = match.group(1)
                    
                    # Skip if already processed
                    down_key = f'{prefix}.lora_down.weight'
                    up_key = f'{prefix}.lora_up.weight'
                    alpha_key = f'{prefix}.alpha'
                    
                    if down_key in processed:
                        continue
                    
                    # Output is separate weight
                    out_base_key = f'layers.{layer_num}.attention.out.weight'
                    
                    if out_base_key not in base_weights:
                        self.log(f"  Warning: Base key {out_base_key} not found. Skipping.")
                        continue
                    
                    # Get LoRA matrices
                    lora_down = lora_weights[down_key].float()
                    lora_up = lora_weights[up_key].float()
                    
                    # Get scaling
                    if alpha_key in lora_weights:
                        lora_alpha = lora_weights[alpha_key].item()
                    else:
                        lora_alpha = 1.0
                    
                    rank = lora_down.shape[0]
                    scaling = lora_alpha / rank
                    
                    # Create LoRA matrix
                    lora_matrix = lora_up @ lora_down
                    merged_weights[out_base_key] = base_weights[out_base_key].float() + lora_matrix * scaling * alpha
                    
                    processed.add(down_key)
                    processed.add(up_key)
                    if alpha_key in lora_weights:
                        processed.add(alpha_key)
                    
                    processed_count += 1
                    attn_layers += 1
                    self.log(f"  ✓ Baked Attention Out layer {layer_num}")
            
            self.log(f"✓ Processed {attn_layers} attention layers")
            
            # Finalize: Replace temporary QKV keys with actual base keys
            self.log("Finalizing QKV merges...")
            final_weights = {}
            qkv_keys = [k for k in merged_weights.keys() if k.startswith('qkv_merged_')]
            for key in qkv_keys:
                layer_num = key.split('_')[-1]
                actual_key = f'layers.{layer_num}.attention.qkv.weight'
                final_weights[actual_key] = merged_weights[key]
                self.log(f"  ✓ Finalized QKV merge for layer {layer_num}")
            
            # Copy all non-QKV weights
            for key, value in merged_weights.items():
                if not key.startswith('qkv_merged_'):
                    final_weights[key] = value
            
            # Save merged model
            self.save_weights(final_weights, output_path)
            
            # Calculate statistics
            end_time = time.time()
            duration = end_time - start_time
            
            self.log("=" * 60)
            self.log("BAKING COMPLETE")
            self.log("=" * 60)
            self.log(f"✓ Time taken: {duration:.2f} seconds")
            self.log(f"✓ LoRA layers merged: {processed_count}")
            self.log(f"  - Feed-forward: {ff_layers}")
            self.log(f"  - Attention: {attn_layers}")
            self.log(f"✓ Saved to: {os.path.basename(output_path)}")
            
            messages.append(f"✅ Successfully baked LoRA to: {os.path.basename(output_path)}")
            messages.append(f"Time taken: {duration:.2f} seconds")
            messages.append(f"LoRA layers merged: {processed_count}")
            messages.append(f"  - Feed-forward: {ff_layers}")
            messages.append(f"  - Attention: {attn_layers}")
            
            return "\n".join(messages), True
            
        except Exception as e:
            error_msg = f"Error baking LoRA: {str(e)}"
            self.log("=" * 60)
            self.log("ERROR")
            self.log("=" * 60)
            self.log(error_msg)
            self.log(traceback.format_exc())
            messages.append(error_msg)
            return "\n".join(messages), False
    
    def extract_lora(self, base_path, merged_path, output_path, rank=4, alpha=1.0, convert_to_comfy=True):
        """Extract LoRA from merged model - EXACT REVERSE of bake_lora logic"""
        messages = []
        
        try:
            # Log start
            start_time = time.time()
            self.log("=" * 60)
            self.log("STARTING LoRA EXTRACTION")
            self.log("=" * 60)
            self.log(f"Base model: {os.path.basename(base_path)}")
            self.log(f"Merged model: {os.path.basename(merged_path)}")
            self.log(f"Output: {os.path.basename(output_path)}")
            self.log(f"Rank: {rank}")
            self.log(f"Alpha: {alpha}")
            self.log(f"Convert to ComfyUI format: {convert_to_comfy}")
            
            # Load weights
            self.log("Loading base model weights...")
            base_weights = self.load_weights(base_path)
            self.log(f"✓ Loaded {len(base_weights)} weights from base model")
            
            self.log("Loading merged model weights...")
            merged_weights = self.load_weights(merged_path)
            self.log(f"✓ Loaded {len(merged_weights)} weights from merged model")
            
            # Storage for extracted LoRA
            lora_weights = {}
            extracted_count = 0
            ff_extracted = 0
            qkv_extracted = 0
            out_extracted = 0
            
            # Extract feed-forward layers - EXACT REVERSE OF BAKE LOGIC
            self.log("Processing feed-forward layers...")
            
            # Find all feed-forward weights in base model
            for base_key in base_weights.keys():
                # Match: layers.{layer_num}.feed_forward.w{w_num}.weight
                match = re.search(r'layers\.(\d+)\.feed_forward\.w([123])\.weight', base_key)
                if not match:
                    continue
                    
                layer_num = match.group(1)
                w_num = match.group(2)
                
                if base_key not in merged_weights:
                    continue
                
                base_weight = base_weights[base_key].float()
                merged_weight = merged_weights[base_key].float()
                weight_diff = merged_weight - base_weight
                
                # If difference is significant, extract LoRA
                if torch.norm(weight_diff) > 1e-8:
                    # Perform SVD on the difference
                    U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
                    
                    # Keep only top 'rank' components
                    rank_to_use = min(rank, len(S))
                    U_k = U[:, :rank_to_use]
                    S_k = S[:rank_to_use]
                    Vh_k = Vh[:rank_to_use, :]
                    
                    # Create LoRA matrices
                    lora_down = Vh_k.contiguous()
                    lora_up = (U_k @ torch.diag(S_k)).contiguous()
                    
                    # Store weights with EXACT SAME NAMING as LoRA file uses
                    prefix = f"lora_unet_layers_{layer_num}_feed_forward_w{w_num}"
                    
                    lora_weights[f"{prefix}.lora_down.weight"] = lora_down
                    lora_weights[f"{prefix}.lora_up.weight"] = lora_up
                    lora_weights[f"{prefix}.alpha"] = torch.tensor(rank_to_use, dtype=torch.float32)
                    
                    extracted_count += 1
                    ff_extracted += 1
                    self.log(f"  ✓ Extracted FF layer {layer_num} w{w_num} (rank: {rank_to_use})")
            
            self.log(f"✓ Extracted {ff_extracted} feed-forward layers")
            
            # Extract attention QKV layers - FIXED VERSION
            self.log("Processing attention QKV layers...")
            
            qkv_keys_found = 0
            qkv_keys_processed = 0
            
            # Find all QKV weights in base model
            for base_key in base_weights.keys():
                # Match: layers.{layer_num}.attention.qkv.weight
                match = re.search(r'layers\.(\d+)\.attention\.qkv\.weight', base_key)
                if not match:
                    continue
                
                qkv_keys_found += 1
                layer_num = match.group(1)
                
                if base_key not in merged_weights:
                    self.log(f"  Warning: {base_key} not in merged weights")
                    continue
                
                base_weight = base_weights[base_key].float()
                merged_weight = merged_weights[base_key].float()
                
                # The QKV weight is stacked as [Q; K; V] along dimension 0
                # Each section should be embed_dim rows
                embed_dim = base_weight.shape[1]
                total_rows = base_weight.shape[0]
                
                self.log(f"  Processing layer {layer_num}: shape {base_weight.shape}, embed_dim={embed_dim}, total_rows={total_rows}")
                
                # Verify shape is correct (should be 3 * embed_dim)
                if total_rows == 3 * embed_dim:
                    qkv_keys_processed += 1
                    section_size = embed_dim
                    
                    # Compute difference for entire QKV
                    weight_diff = merged_weight - base_weight
                    total_diff_norm = torch.norm(weight_diff).item()
                    
                    self.log(f"  Layer {layer_num} total diff norm: {total_diff_norm:.6f}")
                    
                    # Split the difference into Q, K, V sections
                    delta_q = weight_diff[:section_size, :]
                    delta_k = weight_diff[section_size:section_size*2, :]
                    delta_v = weight_diff[section_size*2:, :]
                    
                    # Extract LoRA from each section separately
                    for attn_type, delta_section in [('q', delta_q), ('k', delta_k), ('v', delta_v)]:
                        section_norm = torch.norm(delta_section).item()
                        self.log(f"    {attn_type.upper()} section norm: {section_norm:.6f}")
                        
                        # Only extract if difference is significant
                        if section_norm > 1e-8:
                            # Perform SVD on this section
                            U, S, Vh = torch.linalg.svd(delta_section, full_matrices=False)
                            
                            # Keep only top 'rank' components
                            rank_to_use = min(rank, len(S))
                            U_k = U[:, :rank_to_use]
                            S_k = S[:rank_to_use]
                            Vh_k = Vh[:rank_to_use, :]
                            
                            # Create LoRA matrices and make them contiguous
                            lora_down = Vh_k.contiguous()  # rank × embed_dim
                            lora_up = (U_k @ torch.diag(S_k)).contiguous()  # section_size × rank
                            
                            # Store weights with EXACT SAME NAMING as LoRA file uses
                            prefix = f"lora_unet_layers_{layer_num}_attention_to_{attn_type}"
                            
                            lora_weights[f"{prefix}.lora_down.weight"] = lora_down
                            lora_weights[f"{prefix}.lora_up.weight"] = lora_up
                            lora_weights[f"{prefix}.alpha"] = torch.tensor(rank_to_use, dtype=torch.float32)
                            
                            extracted_count += 1
                            qkv_extracted += 1
                            self.log(f"  ✓ Extracted Attention {attn_type.upper()} layer {layer_num} (rank: {rank_to_use})")
                        else:
                            self.log(f"    Skipping {attn_type.upper()} - norm too small")
                else:
                    self.log(f"  Warning: QKV weight {base_key} has unexpected shape {base_weight.shape} (expected {3*embed_dim} rows)")
            
            self.log(f"  QKV keys found: {qkv_keys_found}, processed: {qkv_keys_processed}")
            self.log(f"✓ Extracted {qkv_extracted} Q/K/V attention layers")
            
            # Process attention output layers - EXACT REVERSE OF BAKE LOGIC
            self.log("Processing attention output layers...")
            
            # Find all attention output weights in base model
            for base_key in base_weights.keys():
                # Match: layers.{layer_num}.attention.out.weight
                match = re.search(r'layers\.(\d+)\.attention\.out\.weight', base_key)
                if not match:
                    continue
                    
                layer_num = match.group(1)
                
                if base_key not in merged_weights:
                    continue
                
                base_weight = base_weights[base_key].float()
                merged_weight = merged_weights[base_key].float()
                weight_diff = merged_weight - base_weight
                
                # If difference is significant, extract LoRA
                if torch.norm(weight_diff) > 1e-8:
                    # Perform SVD
                    U, S, Vh = torch.linalg.svd(weight_diff, full_matrices=False)
                    
                    # Keep only top 'rank' components
                    rank_to_use = min(rank, len(S))
                    U_k = U[:, :rank_to_use]
                    S_k = S[:rank_to_use]
                    Vh_k = Vh[:rank_to_use, :]
                    
                    # Create LoRA matrices and make them contiguous
                    lora_down = Vh_k.contiguous()
                    lora_up = (U_k @ torch.diag(S_k)).contiguous()
                    
                    # Store weights with EXACT SAME NAMING as LoRA file uses
                    prefix = f"lora_unet_layers_{layer_num}_attention_to_out_0"
                    
                    lora_weights[f"{prefix}.lora_down.weight"] = lora_down
                    lora_weights[f"{prefix}.lora_up.weight"] = lora_up
                    lora_weights[f"{prefix}.alpha"] = torch.tensor(rank_to_use, dtype=torch.float32)
                    
                    extracted_count += 1
                    out_extracted += 1
                    self.log(f"  ✓ Extracted Attention Out layer {layer_num} (rank: {rank_to_use})")
            
            self.log(f"✓ Extracted {out_extracted} attention output layers")
            
            # Convert to ComfyUI format if requested
            if convert_to_comfy and lora_weights:
                self.log("Converting extracted LoRA to ComfyUI format...")
                lora_weights = self.convert_lora_format(lora_weights, to_comfy=True)
                self.log("✓ Converted to ComfyUI format")
            
            # Save extracted LoRA
            if lora_weights:
                self.log("Saving extracted LoRA...")
                self.save_weights(lora_weights, output_path)
                
                # Calculate statistics
                end_time = time.time()
                duration = end_time - start_time
                self.log("=" * 60)
                self.log("EXTRACTION COMPLETE")
                self.log("=" * 60)
                self.log(f"✓ Time taken: {duration:.2f} seconds")
                self.log(f"✓ Total LoRA parameters: {len(lora_weights)}")
                self.log(f"✓ Layers extracted: {extracted_count}")
                self.log(f"  - Feed-forward: {ff_extracted}")
                self.log(f"  - Q/K/V attention: {qkv_extracted}")
                self.log(f"  - Attention output: {out_extracted}")
                self.log(f"✓ Format: {'ComfyUI' if convert_to_comfy else 'sd-scripts'}")
                self.log(f"✓ Saved to: {os.path.basename(output_path)}")
                
                messages.append(f"✅ Successfully extracted LoRA to: {os.path.basename(output_path)}")
                messages.append(f"Time taken: {duration:.2f} seconds")
                messages.append(f"Total LoRA parameters: {len(lora_weights)}")
                messages.append(f"Layers extracted: {extracted_count}")
                messages.append(f"  - Feed-forward: {ff_extracted}")
                messages.append(f"  - Q/K/V attention: {qkv_extracted}")
                messages.append(f"  - Attention output: {out_extracted}")
                messages.append(f"Format: {'ComfyUI' if convert_to_comfy else 'sd-scripts'}")
            else:
                self.log("=" * 60)
                self.log("WARNING")
                self.log("=" * 60)
                self.log("⚠️ No significant differences found. LoRA extraction may not be necessary.")
                messages.append("⚠️ No significant differences found. LoRA extraction may not be necessary.")
            
            return "\n".join(messages), True
            
        except Exception as e:
            error_msg = f"Error extracting LoRA: {str(e)}"
            self.log("=" * 60)
            self.log("ERROR")
            self.log("=" * 60)
            self.log(error_msg)
            self.log(traceback.format_exc())
            messages.append(error_msg)
            return "\n".join(messages), False
    
    def process_lora(self, mode, base_model, output_name, rank, alpha, lora_model="None", merged_model="None", convert_to_comfy_format=True):
        """Main processing function"""
        
        # Get base ComfyUI directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define paths
        model_base_path = os.path.join(base_dir, "models", "diffusion_models")
        lora_base_path = os.path.join(base_dir, "models", "loras")
        output_base_path = os.path.join(base_dir, "output")  # NEW: Output directory
        
        # Create output directory if it doesn't exist
        os.makedirs(output_base_path, exist_ok=True)  # NEW: Ensure output directory exists
        
        # Add .safetensors extension if not present
        if not output_name.endswith((".safetensors", ".ckpt")):
            output_name = output_name + ".safetensors"
        
        try:
            if mode == "bake":
                if base_model == "None":
                    return ("❌ Please select a base model.",)
                if lora_model == "None":
                    return ("❌ Please select a LoRA model for baking.",)
                
                # Get full paths
                base_path = self.get_full_path(base_model, model_base_path)
                lora_path = self.get_full_path(lora_model, lora_base_path)
                
                # Output goes to output directory (CHANGED)
                output_path = os.path.join(output_base_path, output_name)
                
                # Handle duplicate filenames
                if os.path.exists(output_path):
                    base_name, ext = os.path.splitext(output_name)
                    counter = 1
                    while os.path.exists(os.path.join(output_base_path, f"{base_name}_{counter}{ext}")):
                        counter += 1
                    output_path = os.path.join(output_base_path, f"{base_name}_{counter}{ext}")
                
                status, success = self.bake_lora(base_path, lora_path, output_path, alpha)
                return (status,)
                
            else:  # mode == "extract"
                if base_model == "None":
                    return ("❌ Please select the original base model.",)
                if merged_model == "None":
                    return ("❌ Please select the merged/finetuned model for extraction.",)
                
                # Get full paths
                base_path = self.get_full_path(base_model, model_base_path)
                merged_path = self.get_full_path(merged_model, model_base_path)
                
                # Output goes to output directory (CHANGED)
                output_path = os.path.join(output_base_path, output_name)
                
                # Handle duplicate filenames
                if os.path.exists(output_path):
                    base_name, ext = os.path.splitext(output_name)
                    counter = 1
                    while os.path.exists(os.path.join(output_base_path, f"{base_name}_{counter}{ext}")):
                        counter += 1
                    output_path = os.path.join(output_base_path, f"{base_name}_{counter}{ext}")
                
                status, success = self.extract_lora(base_path, merged_path, output_path, rank, alpha, convert_to_comfy_format)
                return (status,)
                
        except Exception as e:
            error_msg = f"Fatal error in process_lora: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            return (error_msg,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageLoraManagerAdvanced": ZImageLoraManagerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoRA Manager Advanced": "ZImage LoRA Manager Advanced (Bake/Extract)"
}
