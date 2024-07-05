from .rein_mask2former import ReinMask2FormerHead
from .hrda import HRDAHead
from .daformer_head import DAFormerHead
from .linear_head import LinearHead
<<<<<<< HEAD
from .attention_head import AttentionHead
=======
from .VFMHead import VFMHead
from .Transformer import TransformerDecoder
>>>>>>> f827a44b2846c4f885b84f99de1e8a993005b81b

__all__ = ["ReinMask2FormerHead",
           "HRDAHead",
           "DAFormerHead",
<<<<<<< HEAD
           "DINOhead",
           "LinearHead",
           "AttentionHead"]
=======
           "LinearHead",
           "VFMHead",
           "TransformerDecoder"]
>>>>>>> f827a44b2846c4f885b84f99de1e8a993005b81b
