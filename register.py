import copy

from BCB_model import Detecter
from detecter import module_tools
from detecter.tree_transformer import TreeTransformer

# tt = TreeTransformer(128, 128, num_layers=2, short_heads=2, long_heads=4, global_heads=2, dropout=0.1)
# tt_nm = TreeTransformer(128, 128, num_layers=2, short_heads=0, long_heads=0, global_heads=8, dropout=0.1, use_mask=False)
# tt_tpe = TreeTransformer(128, 128, num_layers=2, short_heads=2, long_heads=4, global_heads=2, dropout=0.1, use_pe=False)

args = [128, 128, 2, 2, 4, 2, 0.1]

tt_no_mask_pe = TreeTransformer(*args, use_mask=False, use_pe=True)
tt_no_mask_tpe = TreeTransformer(*args, use_mask=False, use_pe=False)
tt_mask_pe = TreeTransformer(*args, use_mask=True, use_pe=True)
tt_mask_tpe = TreeTransformer(*args, use_mask=True, use_pe=False)

module_tools.register_module("BCBdetecter_basic", Detecter(tt_no_mask_pe))
module_tools.register_module("BCBdetecter_mask", Detecter(tt_mask_pe))
module_tools.register_module("BCBdetecter_tpe", Detecter(tt_no_mask_tpe))
module_tools.register_module("BCBdetecter_complete", Detecter(copy.deepcopy(tt_mask_tpe)))
module_tools.register_module("BCBdetecter", Detecter(copy.deepcopy(tt_mask_tpe)))
module_tools.register_module("BCBdetecter_finetune", Detecter(copy.deepcopy(tt_mask_tpe)))
