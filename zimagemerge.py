import torch
import re
import os
import time
import numpy as np
from safetensors.torch import load_file, save_file
import folder_paths
import traceback

class ZImageLoraMergerAdvanced:
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
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define paths
        lora_base_path = os.path.join(base_dir, "models", "loras")
        
        # Create directory if it doesn't exist
        os.makedirs(lora_base_path, exist_ok=True)
        
        # Find all LoRA files
        lora_files = find_safetensors_files(lora_base_path)
        if not lora_files:
            lora_files = ["No LoRAs found"]
        
        return {
            "required": {
                "output_name": ("STRING", {"default": "merged_lora"}),
                "auto_normalize": ("BOOLEAN", {"default": True}),
                "output_rank": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "lora1": (["None"] + lora_files, {"default": "None"}),
                "weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "lora2": (["None"] + lora_files, {"default": "None"}),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "lora3": (["None"] + lora_files, {"default": "None"}),
                "weight3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "lora4": (["None"] + lora_files, {"default": "None"}),
                "weight4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "lora5": (["None"] + lora_files, {"default": "None"}),
                "weight5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "lora6": (["None"] + lora_files, {"default": "None"}),
                "weight6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "convert_to_comfy_format": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "merge_loras"
    CATEGORY = "ZImage"
    
    def __init__(self):
        pass
    
    def log(self, message):
        """Log message to console"""
        print(f"[ZImage LoRA Merger] {message}")
    
    def load_weights(self, path):
        """Load weights from file - copied from original script"""
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
        """Save weights to file with proper tensor contiguity - copied from original script"""
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
        """Convert filename to full path - copied from original script"""
        if filename == "None":
            return None
        return os.path.join(base_path, filename)
    
    def is_lora_converted_to_comfy(self, state_dict):
        """Check if LoRA is already in ComfyUI format (converted) - copied from original script"""
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
        """Convert LoRA between sd-scripts and ComfyUI formats - copied from original script"""
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
    
    def merge_different_ranks(self, tensors, weights, output_rank):
        """Merge tensors with different ranks using SVD reduction"""
        # Simple weighted sum for same shape tensors
        if all(t.shape == tensors[0].shape for t in tensors):
            merged = sum(w * t for w, t in zip(weights, tensors))
            
            # If output rank is different, apply SVD truncation/expansion
            if output_rank != merged.shape[1]:
                U, S, Vh = torch.linalg.svd(merged.float(), full_matrices=False)
                
                if output_rank < len(S):
                    # Truncate to output_rank
                    U_k = U[:, :output_rank]
                    S_k = S[:output_rank]
                    Vh_k = Vh[:output_rank, :]
                    merged = (U_k @ torch.diag(S_k)) @ Vh_k
                else:
                    # Pad with zeros to output_rank
                    pad_size = output_rank - merged.shape[1]
                    merged = torch.cat([merged, torch.zeros(merged.shape[0], pad_size, 
                                                          dtype=merged.dtype, 
                                                          device=merged.device)], dim=1)
            
            return merged.contiguous()
        
        # For different shapes, use the largest shape and pad smaller ones
        max_rows = max(t.shape[0] for t in tensors)
        max_cols = max(t.shape[1] for t in tensors)
        
        padded_tensors = []
        for t in tensors:
            pad_rows = max_rows - t.shape[0]
            pad_cols = max_cols - t.shape[1]
            
            if pad_rows > 0 or pad_cols > 0:
                padded = torch.zeros((max_rows, max_cols), dtype=t.dtype, device=t.device)
                padded[:t.shape[0], :t.shape[1]] = t
                padded_tensors.append(padded)
            else:
                padded_tensors.append(t)
        
        # Weighted sum of padded tensors
        merged = sum(w * t for w, t in zip(weights, padded_tensors))
        
        # Apply SVD to achieve desired output rank
        if output_rank < max_cols:
            U, S, Vh = torch.linalg.svd(merged.float(), full_matrices=False)
            
            # Truncate to output_rank
            U_k = U[:, :output_rank]
            S_k = S[:output_rank]
            Vh_k = Vh[:output_rank, :]
            merged = (U_k @ torch.diag(S_k)) @ Vh_k
        
        return merged.contiguous()
    
    def merge_loras(self, output_name, auto_normalize, output_rank,
                   lora1="None", weight1=1.0,
                   lora2="None", weight2=1.0,
                   lora3="None", weight3=1.0,
                   lora4="None", weight4=1.0,
                   lora5="None", weight5=1.0,
                   lora6="None", weight6=1.0,
                   convert_to_comfy_format=True):
        
        messages = []
        
        try:
            # Log start
            start_time = time.time()
            self.log("=" * 60)
            self.log("STARTING LoRA MERGING")
            self.log("=" * 60)
            
            # Get base ComfyUI directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lora_base_path = os.path.join(base_dir, "models", "loras")
            output_base_path = os.path.join(base_dir, "output")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_base_path, exist_ok=True)
            
            # Collect LoRAs and weights
            lora_configs = [
                (lora1, weight1),
                (lora2, weight2),
                (lora3, weight3),
                (lora4, weight4),
                (lora5, weight5),
                (lora6, weight6),
            ]
            
            # Filter out "None" LoRAs
            active_loras = []
            active_weights = []
            
            for lora_name, weight in lora_configs:
                if lora_name != "None":
                    active_loras.append(lora_name)
                    active_weights.append(weight)
            
            if len(active_loras) == 0:
                return ("❌ Please select at least one LoRA to merge.",)
            
            if len(active_loras) == 1:
                return ("ℹ️ Only one LoRA selected. Nothing to merge.",)
            
            self.log(f"Merging {len(active_loras)} LoRAs")
            self.log(f"Weights: {active_weights}")
            self.log(f"Output rank: {output_rank}")
            self.log(f"Auto normalize: {auto_normalize}")
            
            # Normalize weights if auto_normalize is True
            if auto_normalize:
                weight_sum = sum(active_weights)
                if weight_sum > 0:
                    active_weights = [w / weight_sum for w in active_weights]
                    self.log(f"Normalized weights: {[f'{w:.3f}' for w in active_weights]}")
                else:
                    self.log("⚠️ All weights are zero, using equal weights")
                    active_weights = [1.0 / len(active_loras)] * len(active_loras)
            
            # Load all LoRAs
            loaded_loras = []
            lora_paths = []
            
            for lora_name in active_loras:
                lora_path = self.get_full_path(lora_name, lora_base_path)
                lora_weights = self.load_weights(lora_path)
                loaded_loras.append(lora_weights)
                lora_paths.append(lora_path)
                self.log(f"  ✓ Loaded: {os.path.basename(lora_path)}")
            
            # Check formats and convert to sd-scripts format for consistent merging
            converted_loras = []
            for i, lora_weights in enumerate(loaded_loras):
                is_comfy = self.is_lora_converted_to_comfy(lora_weights)
                if is_comfy:
                    self.log(f"  Converting LoRA {i+1} to sd-scripts format for merging...")
                    converted = self.convert_lora_format(lora_weights, to_comfy=False)
                    converted_loras.append(converted)
                else:
                    converted_loras.append(lora_weights)
            
            # Get all unique keys from all LoRAs
            all_keys = set()
            for lora_weights in converted_loras:
                all_keys.update(lora_weights.keys())
            
            self.log(f"Total unique parameters: {len(all_keys)}")
            
            # Initialize merged LoRA
            merged_lora = {}
            
            # Process each parameter type
            alpha_keys = [k for k in all_keys if k.endswith('.alpha')]
            down_keys = [k for k in all_keys if 'lora_down' in k]
            up_keys = [k for k in all_keys if 'lora_up' in k]
            
            # Process alpha values (weighted average)
            self.log("Processing alpha values...")
            alpha_count = 0
            for key in alpha_keys:
                # Collect alphas from all LoRAs that have this key
                alphas = []
                for lora_weights in converted_loras:
                    if key in lora_weights:
                        alphas.append(lora_weights[key].float())
                    else:
                        alphas.append(torch.tensor(0.0))
                
                # Weighted average of alphas
                weighted_sum = sum(w * a for w, a in zip(active_weights, alphas))
                merged_lora[key] = weighted_sum.contiguous()
                alpha_count += 1
            
            self.log(f"✓ Processed {alpha_count} alpha values")
            
            # Process lora_down weights
            self.log("Processing lora_down weights...")
            down_count = 0
            for key in down_keys:
                # Collect tensors from all LoRAs that have this key
                tensors = []
                tensor_weights = []
                
                for i, lora_weights in enumerate(converted_loras):
                    if key in lora_weights:
                        tensors.append(lora_weights[key].float())
                        tensor_weights.append(active_weights[i])
                
                if len(tensors) > 0:
                    # Merge tensors with different ranks
                    merged_tensor = self.merge_different_ranks(tensors, tensor_weights, output_rank)
                    merged_lora[key] = merged_tensor
                    down_count += 1
            
            self.log(f"✓ Processed {down_count} lora_down weights")
            
            # Process lora_up weights
            self.log("Processing lora_up weights...")
            up_count = 0
            for key in up_keys:
                # Collect tensors from all LoRAs that have this key
                tensors = []
                tensor_weights = []
                
                for i, lora_weights in enumerate(converted_loras):
                    if key in lora_weights:
                        tensors.append(lora_weights[key].float())
                        tensor_weights.append(active_weights[i])
                
                if len(tensors) > 0:
                    # Merge tensors with different ranks
                    merged_tensor = self.merge_different_ranks(tensors, tensor_weights, output_rank)
                    merged_lora[key] = merged_tensor
                    up_count += 1
            
            self.log(f"✓ Processed {up_count} lora_up weights")
            
            # Convert to ComfyUI format if requested
            if convert_to_comfy_format:
                self.log("Converting merged LoRA to ComfyUI format...")
                merged_lora = self.convert_lora_format(merged_lora, to_comfy=True)
            
            # Save output
            if not output_name.endswith((".safetensors", ".ckpt")):
                output_name = output_name + ".safetensors"
            
            output_path = os.path.join(output_base_path, output_name)
            
            # Handle duplicate filenames
            if os.path.exists(output_path):
                base_name, ext = os.path.splitext(output_name)
                counter = 1
                while os.path.exists(os.path.join(output_base_path, f"{base_name}_{counter}{ext}")):
                    counter += 1
                output_path = os.path.join(output_base_path, f"{base_name}_{counter}{ext}")
            
            self.save_weights(merged_lora, output_path)
            
            # Calculate statistics
            end_time = time.time()
            duration = end_time - start_time
            
            self.log("=" * 60)
            self.log("MERGE COMPLETE")
            self.log("=" * 60)
            self.log(f"✓ Time taken: {duration:.2f} seconds")
            self.log(f"✓ LoRAs merged: {len(active_loras)}")
            self.log(f"✓ Output rank: {output_rank}")
            self.log(f"✓ Parameters: {len(merged_lora)}")
            self.log(f"✓ Saved to: {os.path.basename(output_path)}")
            
            messages.append(f"✅ Successfully merged {len(active_loras)} LoRAs")
            messages.append(f"Time taken: {duration:.2f} seconds")
            messages.append(f"Output rank: {output_rank}")
            messages.append(f"Total parameters: {len(merged_lora)}")
            messages.append(f"Saved to: {os.path.basename(output_path)}")
            
            return ("\n".join(messages),)
            
        except Exception as e:
            error_msg = f"Error merging LoRAs: {str(e)}"
            self.log("=" * 60)
            self.log("ERROR")
            self.log("=" * 60)
            self.log(error_msg)
            self.log(traceback.format_exc())
            return (error_msg,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageLoraMergerAdvanced": ZImageLoraMergerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoraMergerAdvanced": "ZImage LoRA Merger Advanced (Multi-LoRA)"
}