"""
Loop and sequence emission helpers.

Extracted from control_flow.py to keep responsibilities smaller and isolated.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...AstNode import ASTNode
from ...NodeType import NodeType

if TYPE_CHECKING:
    from ...state import CodeGenState
    from ....TritonGen import TritonCodeGen


class LoopEmitter:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_ploop(self, node: ASTNode) -> str:
        """Generate parallel loop code - executed across GPU grid dimensions"""
        # (ploop start end tile_size loop_var body)
        start = self.gen.dispatch.generate_node(node.children[0])
        end = self.gen.dispatch.generate_node(node.children[1])
        tile_size = self.gen.dispatch.generate_node(node.children[2])
        loop_var = node.children[3].value
        body = node.children[4]

        # Store loop info for tile generation
        self.state.loop_vars[loop_var] = (start, end, tile_size, True)  # True = parallel

        # For parallel loops, map to grid dimensions
        # Find the grid dimension for this loop variable
        grid_dim = -1
        for idx, (lv, _, _, _) in enumerate(self.state.parallel_dims):
            if lv == loop_var:
                grid_dim = idx
                break

        # If not found (shouldn't happen), use the current count
        if grid_dim == -1:
            grid_dim = len(self.state.parallel_dims)
            self.state.parallel_dims.append((loop_var, start, end, tile_size))

        indent = '    ' * self.state.indent_level
        # Use BLOCK_SIZE parameter instead of hardcoded tile_size
        block_param = f"BLOCK_{loop_var.upper()}"

        # Determine the end value - use tensor dimension parameter if available
        end_param = self._get_end_param(loop_var, end)

        code = f"{indent}# Parallel loop {loop_var} from {start} to {end_param} with tile size {block_param}\n"
        code += f"{indent}# Executed across grid dimension {grid_dim}\n"
        code += f"{indent}{loop_var} = {start} + tl.program_id({grid_dim}) * {block_param}\n"
        code += f"{indent}\n"
        # Generate body without boundary check
        body_code = self.gen.dispatch.generate_node(body)
        # Remove any leading/trailing newlines from body to avoid double newlines
        body_code = body_code.strip('\n')
        if body_code:
            code += body_code + '\n'
        else:
            # Empty body
            code += f"{indent}pass\n"

        return code

    def generate_sloop(self, node: ASTNode) -> str:
        """Generate sequential loop code - executed as traditional for loop"""
        # (sloop start end tile_size loop_var body)
        start = self.gen.shape_utils.resolve_value(self.gen.dispatch.generate_node(node.children[0]))
        end = self.gen.shape_utils.resolve_value(self.gen.dispatch.generate_node(node.children[1]))
        tile_size = self.gen.dispatch.generate_node(node.children[2])  # No need to resolve - used as BLOCK parameter
        loop_var = node.children[3].value
        body = node.children[4]

        # Check if body is only dummy - skip loop generation
        if body.node_type == NodeType.DUMMY:
            indent = '    ' * self.state.indent_level
            return f"{indent}# Skipped empty sloop with dummy body\n"

        # Store loop info for tile generation
        self.state.loop_vars[loop_var] = (start, end, tile_size, False)  # False = sequential

        indent = '    ' * self.state.indent_level
        # Use BLOCK_SIZE parameter instead of hardcoded tile_size
        block_param = f"BLOCK_{loop_var.upper()}"

        # For sloop, use the end value directly (whether it's a number or variable)
        end_param = end

        code = f"{indent}# Sequential loop {loop_var} from {start} to {end_param} with tile size {block_param}\n"
        code += f"{indent}for {loop_var} in range({start}, {end_param}, {block_param}):\n"

        # Increment loop instance counter for this new loop
        self.state.loop_instance_counter += 1
        saved_loop_instance = self.state.current_loop_instance
        self.state.current_loop_instance = self.state.loop_instance_counter

        # Clear load cache for loop body since loop variable changes
        # We need to reload tensors that depend on the loop variable
        saved_load_cache = self.state.load_cache.copy()
        self.state.load_cache = {}

        # Set current sloop context
        saved_sloop_info = self.state.current_sloop_info
        self.state.current_sloop_info = (loop_var, node)

        self.state.indent_level += 1
        self.state.sloop_depth += 1  # Increment sloop depth

        # Check if this sloop contains nested accumulators that need initialization
        # Pass the loop variable to only get accumulators that use this specific loop var
        nested_accumulators = self.gen.analyzer.find_nested_sloop_accumulators(body, loop_var)

        # Temporarily remove nested accumulators from cross_sloop_memory_tensors
        # so they accumulate in registers inside this loop
        saved_cross_sloop = self.state.cross_sloop_memory_tensors.copy()
        self.state.cross_sloop_memory_tensors = self.state.cross_sloop_memory_tensors - nested_accumulators

        if nested_accumulators:
            # Add nested accumulators to kernel_accumulators temporarily
            # so they use register accumulation inside this loop
            for acc in nested_accumulators:
                self.state.kernel_accumulators.add(acc)

            # Generate initialization code for nested accumulators
            init_code = self.gen.analyzer.generate_nested_accumulator_init(nested_accumulators)
            if init_code:
                code += init_code

        body_code = self.gen.dispatch.generate_node(body)

        # Remove any leading/trailing newlines from body to avoid double newlines
        body_code = body_code.strip('\n')
        if body_code:
            code += body_code + '\n'
        else:
            # Empty body
            code += f"{indent}    pass\n"
        self.state.indent_level -= 1
        self.state.sloop_depth -= 1  # Decrement sloop depth

        # Remove nested accumulators from kernel_accumulators AFTER processing the body
        # and generate stores for them (like FF2 pattern)
        if nested_accumulators:
            # Use indent_level + 1 because we're still inside the loop
            indent_str = '    ' * (self.state.indent_level + 1)
            for acc in sorted(nested_accumulators):
                self.state.kernel_accumulators.discard(acc)

                # Generate store for this accumulator if it has index info
                if acc in self.state.intermediate_tensor_indices:
                    index_node = self.state.intermediate_tensor_indices[acc]

                    # Check if this index uses the current loop variable
                    uses_current_loop = False
                    if self.state.current_sloop_info:
                        current_loop_var = self.state.current_sloop_info[0]
                        for child in index_node.children:
                            if child.node_type == NodeType.TILE:
                                if child.children and child.children[0].node_type == NodeType.VAR:
                                    if child.children[0].value == current_loop_var:
                                        uses_current_loop = True
                                        break

                    if uses_current_loop:
                        # Generate store like FF2 pattern
                        index_code = self.gen.indexer.generate_index(index_node, acc)
                        code += f"{indent_str}offset_{self.state.offset_counter} = {index_code}\n"

                        # Generate mask with proper indentation
                        saved_indent = self.state.indent_level
                        self.state.indent_level += 1  # Temporarily increase for mask generation
                        mask_code, mask_var = self.gen.masking.generate_mask_for_index(index_node, acc)
                        self.state.indent_level = saved_indent  # Restore

                        if mask_code:
                            # Add proper indentation to mask code lines
                            mask_lines = mask_code.strip().split('\n')
                            for line in mask_lines:
                                if line.strip():
                                    code += f"{indent_str}{line.strip()}\n"

                        if mask_var:
                            code += f"{indent_str}tl.store({acc}_ptr + offset_{self.state.offset_counter}, {acc}, mask={mask_var})\n"
                        else:
                            code += f"{indent_str}tl.store({acc}_ptr + offset_{self.state.offset_counter}, {acc})\n"

                        self.state.offset_counter += 1

        # Restore load cache after loop
        self.state.load_cache = saved_load_cache

        # Restore previous sloop context
        self.state.current_sloop_info = saved_sloop_info

        # Restore previous loop instance
        self.state.current_loop_instance = saved_loop_instance

        # Restore cross_sloop_memory_tensors
        self.state.cross_sloop_memory_tensors = saved_cross_sloop

        # Find tensors stored in this sloop that are used in OTHER sloops.
        # This triggers Triton's OptimizeThreadLocality "loopResult.hasOneUse()" assertion.
        # We add materialization (+ 0.0) only for such tensors to break the multi-loop use pattern.
        stored_in_this_sloop = set()
        if len(node.children) > 4:
            body = node.children[4]
            def find_stored_tensors(n: ASTNode):
                if n.node_type == NodeType.STORE:
                    tensor_node = n.children[0]
                    if tensor_node.node_type == NodeType.TENSOR:
                        for child in tensor_node.children:
                            if child.node_type == NodeType.VAR:
                                if child.value in self.state.intermediate_tensors:
                                    stored_in_this_sloop.add(child.value)
                for child in n.children:
                    if isinstance(child, ASTNode) and child.node_type != NodeType.SLOOP:
                        find_stored_tensors(child)
            find_stored_tensors(body)

        # Count how many distinct sloops use each tensor
        def count_sloops_using_tensor(tensor_name: str) -> int:
            """Count how many distinct sloops load this tensor"""
            sloop_count = [0]
            def check_sloop(n: ASTNode, current_sloop_id: int = 0):
                if n.node_type == NodeType.SLOOP:
                    # New sloop - increment ID and check inside
                    new_id = current_sloop_id + 1
                    uses_tensor = [False]
                    def check_loads(inner: ASTNode):
                        if inner.node_type == NodeType.LOAD:
                            if len(inner.children) > 0:
                                t_node = inner.children[0]
                                if t_node.node_type == NodeType.TENSOR:
                                    for child in t_node.children:
                                        if child.node_type == NodeType.VAR and child.value == tensor_name:
                                            uses_tensor[0] = True
                                            return
                        for child in inner.children:
                            if isinstance(child, ASTNode) and child.node_type != NodeType.SLOOP:
                                check_loads(child)
                    # Check if this sloop uses the tensor (excluding nested sloops)
                    if len(n.children) > 4:
                        check_loads(n.children[4])
                    if uses_tensor[0]:
                        sloop_count[0] += 1
                    # Continue checking nested/sibling sloops
                    for child in n.children:
                        if isinstance(child, ASTNode):
                            check_sloop(child, new_id)
                else:
                    for child in n.children:
                        if isinstance(child, ASTNode):
                            check_sloop(child, current_sloop_id)
            if self.state.current_ast:
                check_sloop(self.state.current_ast)
            return sloop_count[0]

        # Add materialization only for tensors used in multiple sloops
        local_intermediate_tensors = set(self.state.intermediate_tensors)
        if hasattr(self.state, 'cross_sloop_memory_tensors'):
            local_intermediate_tensors -= self.state.cross_sloop_memory_tensors
        local_intermediate_tensors -= self.state.cross_kernel_tensors
        if os.environ.get("TRITON_GEN_DEBUG") == "1":
            print(
                "[TRITON_GEN_DEBUG] materialize "
                f"stored_in_sloop={sorted(stored_in_this_sloop)} "
                f"local_intermediates={sorted(local_intermediate_tensors)} "
                f"cross_sloop={sorted(getattr(self.state, 'cross_sloop_memory_tensors', set()))} "
                f"cross_kernel={sorted(self.state.cross_kernel_tensors)}"
            )
        for tensor in sorted(stored_in_this_sloop):
            if tensor not in local_intermediate_tensors:
                continue
            # If used in 2+ distinct sloops, needs materialization
            if count_sloops_using_tensor(tensor) >= 2:
                code += f"{indent}{tensor} = {tensor} + 0.0\n"

        return code

    def generate_seq(self, node: ASTNode) -> str:
        """Generate sequence of operations (for nested seq nodes only)"""
        code = ""
        for child in node.children:
            # Skip dummy nodes in sequences
            if child.node_type == NodeType.DUMMY:
                continue
            child_code = self.gen.dispatch.generate_node(child)
            if child_code:  # Only add if there's actual code
                code += child_code
                # Only add newline if the code doesn't already end with one
                if not child_code.endswith('\n'):
                    code += '\n'
        return code

    def _get_end_param(self, loop_var: str, end_value: str) -> str:
        """Get the appropriate parameter name for the end value"""
        # Check if we have a mapping for this loop variable
        if loop_var in self.state.loop_var_to_tensor_dim:
            tensor_name, dim = self.state.loop_var_to_tensor_dim[loop_var]
            return f"{tensor_name}_dim{dim}"

        # Default to the literal value if no mapping found
        return end_value

    def collect_all_loops(self, node: ASTNode):
        """Collect all loop information for BLOCK parameters"""
        if node.node_type == NodeType.PLOOP:
            # Extract loop info: (ploop start end tile_size loop_var body)
            # Resolve start value (might be a constant variable)
            start_val = node.children[0].value if hasattr(node.children[0], 'value') else "0"
            start = self.gen.shape_utils.resolve_value(start_val)

            # Resolve end value (might be a constant variable)
            end_val = node.children[1].value if hasattr(node.children[1], 'value') else "N"
            end = self.gen.shape_utils.resolve_value(end_val)
            # Get tile size - preserve variable names like tile_p, tile_n
            if node.children[2].node_type == NodeType.NUM:
                tile_size = node.children[2].value
            else:
                # It's a variable name like tile_p, tile_n
                tile_size = node.children[2].value if hasattr(node.children[2], 'value') else str(node.children[2])
            loop_var = node.children[3].value

            # Check if this loop variable already exists in parallel_dims
            loop_var_exists = any(lv == loop_var for lv, _, _, _ in self.state.parallel_dims)

            # Only add if not already present
            if not loop_var_exists:
                self.state.parallel_dims.append((loop_var, str(start), str(end), str(tile_size)))

            # Also add to all_loops if not already present
            all_loop_exists = any(lv == loop_var for lv, _, _, _ in self.state.all_loops)
            if not all_loop_exists:
                self.state.all_loops.append((loop_var, str(start), str(end), str(tile_size)))

            # Continue collecting in body
            if len(node.children) > 4:
                self.collect_all_loops(node.children[4])
        elif node.node_type == NodeType.SLOOP:
            # Check if body is dummy - skip if so
            body = node.children[4] if len(node.children) > 4 else None
            if body and body.node_type == NodeType.DUMMY:
                # Skip loops with dummy body
                return

            # Sequential loops are NOT parallel - don't add them to parallel_dims
            # But add to all_loops for BLOCK parameters
            # Resolve start value (might be a constant variable)
            start_val = node.children[0].value if hasattr(node.children[0], 'value') else "0"
            start = self.gen.shape_utils.resolve_value(start_val)

            # Resolve end value (might be a constant variable)
            end_val = node.children[1].value if hasattr(node.children[1], 'value') else "N"
            end = self.gen.shape_utils.resolve_value(end_val)
            # Get tile size - preserve variable names like tile_p, tile_n
            if node.children[2].node_type == NodeType.NUM:
                tile_size = node.children[2].value
            else:
                # It's a variable name like tile_p, tile_n
                tile_size = node.children[2].value if hasattr(node.children[2], 'value') else str(node.children[2])
            loop_var = node.children[3].value

            # Add to all_loops if not already present
            all_loop_exists = any(lv == loop_var for lv, _, _, _ in self.state.all_loops)
            if not all_loop_exists:
                self.state.all_loops.append((loop_var, str(start), str(end), str(tile_size)))

            # Continue collecting in body
            if len(node.children) > 4:
                self.collect_all_loops(node.children[4])
        else:
            # Recursively collect from children
            for child in node.children:
                if isinstance(child, ASTNode):
                    self.collect_all_loops(child)
