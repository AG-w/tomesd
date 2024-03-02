import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
from .utils import isinstance_str, init_generator

import math
import torch.nn.functional as F

def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest"):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1).permute(0, 3, 1, 2)
    item = F.interpolate(item, size=(new_h, new_w), mode=method).permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item

def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    downsample_factor_1 = args['downsample_factor']
    downsample_factor_2 = args['downsample_factor_level_2']	
    downsample_factor_3 = args['downsample_factor_level_3']
	
    cur_h = original_h // downsample
    cur_w = original_w // downsample
    print(downsample, " xxx ", downsample_factor_1)
    m = lambda v: v
    u = lambda v: v
    if downsample == 1 and downsample_factor_1 > 1:
        new_h = int(cur_h / downsample_factor_1)
        new_w = int(cur_w / downsample_factor_1)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])
        u = lambda v: up_or_downsample(v, new_w, new_h, cur_w, cur_h, args["downsample_method"])		
    elif downsample == 2 and downsample_factor_2 > 1:
        new_h = int(cur_h / downsample_factor_2)
        new_w = int(cur_w / downsample_factor_2)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])
        u = lambda v: up_or_downsample(v, new_w, new_h, cur_w, cur_h, args["downsample_method"])
    elif downsample == 4 and downsample_factor_3 > 1:
        new_h = int(cur_h / downsample_factor_3)
        new_w = int(cur_w / downsample_factor_3)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])
        u = lambda v: up_or_downsample(v, new_w, new_h, cur_w, cur_h, args["downsample_method"])
		
    return m, u


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m, u = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = self.attn1(self.norm1(x), context = context if self.disable_self_attn else None) + x
            x = u(self.attn2(m(self.norm2(x)), context = context)) + x
            x = self.ff(self.norm3(x)) + x

            return x
    
    return ToMeBlock



def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock






def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))








def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        merge_tokens: str = "keys/values",
        merge_method: str = "downsample",
        downsample_method: str = "nearest",
        downsample_factor: float = 2,
        #timestep_threshold_switch: float = 0.2,
        #timestep_threshold_stop: float = 0.0,
        #secondary_merge_method: str = "similarity",
        downsample_factor_level_2: float = 1,
	downsample_factor_level_3: float = 1,
        ratio_level_2: float = 0.5
        ):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._tome_info = {
        "size": None,
        #"timestep": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
			
            "merge_tokens": merge_tokens,  # ["all", "keys/values"]
            "merge_method": merge_method,  # ["none","similarity", "downsample"]
            "downsample_method": downsample_method, # native torch interpolation methods ["nearest", "linear", "bilinear", "bicubic", "nearest-exact"]
            "downsample_factor": downsample_factor,  # amount to downsample by
            #"timestep_threshold_switch": timestep_threshold_switch, # timestep to switch to secondary method, 0.2 means 20% steps remaining
            #"timestep_threshold_stop": timestep_threshold_stop, # timestep to stop merging, 0.0 means stop at 0 steps remaining
            #"secondary_merge_method": secondary_merge_method, # ["none", "similarity", "downsample"]

            "downsample_factor_level_2": downsample_factor_level_2, # amount to downsample by at the 2nd down block of unet
            "ratio_level_2": ratio_level_2, # ratio of tokens to merge at the 2nd down block of unet
	    "downsample_factor_level_3": downsample_factor_level_3,
        }
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
