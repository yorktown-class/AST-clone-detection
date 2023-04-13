from detecter import module_tools

from BCB_model import Detecter
from detecter.tree_transformer import TreeTransformer

tt = TreeTransformer(128, 128, num_layers=2, short_heads=2, long_heads=4, global_heads=2, dropout=0.1)
tt_nm = TreeTransformer(128, 128, num_layers=2, short_heads=0, long_heads=0, global_heads=8, dropout=0.1, use_mask=False)
tt_tpe = TreeTransformer(128, 128, num_layers=2, short_heads=2, long_heads=4, global_heads=2, dropout=0.1, use_pe=False)

# module_tools.register_module("tt", tt)
# module_tools.register_module("tt_nm", tt_nm)
# module_tools.register_module("tt_tpe", tt_tpe)

module_tools.register_module("BCBdetecter", Detecter(tt))
module_tools.register_module("BCBdetecter_no_mask", Detecter(tt_nm))
module_tools.register_module("BCBdetecter_tpe", Detecter(tt_tpe))

module_tools.register_module("BCBdetecter_final", Detecter(tt_tpe))
