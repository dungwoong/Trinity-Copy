from dataclasses import dataclass
from typing import List, Any, Optional, Tuple
from .NodeType import NodeType

@dataclass
class ASTNode:
    node_type: NodeType
    children: List['ASTNode']
    value: Any = None
    tensor_shape: Optional[Tuple[Any, ...]] = None
    block_shape: Optional[Tuple[Any, ...]] = None
    
    def __repr__(self):
        if self.value is not None:
            return f"{self.node_type.value}({self.value})"
        return f"{self.node_type.value}({self.children})"
