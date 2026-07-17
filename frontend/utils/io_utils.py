import json
import dataclasses
import os
from dataclasses import is_dataclass
from typing import List, Optional
import ir.AST as T
from core.to_ir import ast_to_lisp, calls_to_ir, calls_to_ir_with_groups

def format_main_func(main_func: T.MainFunc, role_map: dict[str, str] | None = None) -> str:
    if role_map is None:
        role_map = {t.name: "input" for t in main_func.input_tensors}
        for tensor in main_func.output_tensors:
            role_map[tensor.name] = "output"
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("MainFunc")
    lines.append("- Inputs")
    if main_func.input_tensors:
        for tensor in main_func.input_tensors:
            lines.append(f"  {_format_tensor_info(tensor)}")
    else:
        lines.append("  (none)")
    lines.append("- Outputs")
    if main_func.output_tensors:
        for tensor in main_func.output_tensors:
            lines.append(f"  {_format_tensor_info(tensor)}")
    else:
        lines.append("  (none)")
    lines.append("- Intermediate Tensors")
    if main_func.intermediate_tensors:
        for tensor in main_func.intermediate_tensors:
            lines.append(f"  {_format_tensor_info(tensor)}")
    else:
        lines.append("  (none)")
    lines.append("- Calls")
    if main_func.calls:
        for call in main_func.calls:
            lines.append(_format_primfunc_call(call, role_map=role_map))
    else:
        lines.append("  (none)")
    return "\n".join(lines)

def format_primfunc_nodes(primfunc_nodes: List[T.PrimFunc]) -> str:
    return "\n".join([_format_primfunc_node(pf) for pf in primfunc_nodes])

def _format_primfunc_node(
    primfunc: T.PrimFunc,
    use_dividen: bool = True,
    role_map: dict[str, str] | None = None,
) -> str:
    lines: List[str] = []
    if use_dividen:
        lines.append("=" * 60)
    lines.append(f"PrimFunc: {primfunc.name}")
    lines.append("\n- Input Tensors")
    if primfunc.input_tensors:
        for tensor in primfunc.input_tensors:
            lines.append(f"  {_format_tensor_info(tensor)}")
    else:
        lines.append("  (none)")
    lines.append("\n- Output Tensor")
    lines.append(f"  {_format_tensor_info(primfunc.output_tensor)}")
    # lines.append("\n- Spatial Axes")
    # if getattr(primfunc, "spatial_axes", None):
    #     lines.append(f"  {', '.join(primfunc.spatial_axes)}")
    # else:
    #     lines.append("  (none)")
    # lines.append("\n- Reduce Axes")
    # if getattr(primfunc, "reduce_axes", None):
    #     lines.append(f"  {', '.join(primfunc.reduce_axes)}")
    # else:
    #     lines.append("  (none)")
    
    if primfunc.allocated_tensors:
        lines.append("\n- Allocated Tensors")
        for tensor in primfunc.allocated_tensors:
            lines.append(f"  {_format_tensor_info(tensor)}")
    lines.append("\n- Root Node (Lisp)")
    lines.append(ast_to_lisp(primfunc.root_node, role_map=role_map))
    return "\n".join(lines)

def _format_primfunc_call(call: T.PrimFuncCall, role_map: dict[str, str] | None = None) -> str:
    lines: List[str] = []
    lines.append("  " + "=" * 60)
    inputs_str = ", ".join([_format_tensor_info(val) for val in call.input_tensors])
    out_str = _format_tensor_info(call.out_var_tensor)
    lines.append(f"  - {call.primfunc.name} (call {call.call_index}): out={out_str}, inputs=[{inputs_str}]")
    lines.append("  " + "=" * 60)
    primfunc_text = _format_primfunc_node(call.primfunc, use_dividen=False, role_map=role_map)
    lines.extend([f"    {line}" for line in primfunc_text.splitlines()])
    lines.append("\n\n\n\n")
    return "\n".join(lines)

def _format_tensor_info(tensor: T.TensorInfo) -> str:
    shape_str = ", ".join([str(dim) for dim in tensor.shape])
    return f"{tensor.name}: shape=({shape_str}), dtype={tensor.dtype}"


def save_to_json(func_nodes, output_path):
    """
    FuncNode 리스트를 JSON 파일로 저장합니다.
    Dataclass를 딕셔너리로 자동 변환합니다.
    """
    class DataclassJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(func_nodes, f, cls=DataclassJSONEncoder, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

def save_to_text(func_nodes, output_path):
    """
    이쁘게 포맷팅된 텍스트 형태로 파일에 저장합니다.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for func in func_nodes:
                f.write(f"Function: {func.name}\n")
                f.write(f"  Params: {func.params}\n")
                f.write(f"  Blocks:\n")
                
                for block in func.blocks:
                    if block.name == 'root' and not block.value and not block.init:
                        continue

                    f.write(f"    [Block: {block.name}]\n")
                    
                    if block.iter_vars:
                        f.write(f"      Iterators:\n")
                        for name, var in block.iter_vars.items():
                            f.write(f"        - {var.name}: {var.range} ({var.type}) <- {var.iter_value}\n")

                    if block.init:
                        f.write(f"      Init Ops:\n")
                        for idx, op in enumerate(block.init):
                            f.write(f"        {idx:2d}: {op}\n")

                    if block.value:
                        f.write(f"      Body Ops:\n")
                        for idx, op in enumerate(block.value):
                            f.write(f"        {idx:2d}: {op}\n")
                    
                    f.write("\n")
                f.write("-" * 60 + "\n")
        print(f"✅ Text report saved successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving text: {e}")



def export_main_func(
    main_func: T.MainFunc,
    output_dir: str,
    basename: str,
    fusion_groups=None,
    flat_output: bool = False,
) -> None:
    if flat_output:
        inner_output_dir = output_dir
    else:
        inner_output_dir = f"{output_dir}/trinity/{basename}"
    os.makedirs(inner_output_dir, exist_ok=True)

    main_path = f"{inner_output_dir}/main.txt"
    seq_path = f"{inner_output_dir}/ir.txt"
    shapes_path = f"{inner_output_dir}/shapes.json"

    with open(main_path, "w") as f:
        f.write(format_main_func(main_func))
    with open(seq_path, "w") as f:
        if fusion_groups:
            f.write(calls_to_ir_with_groups(main_func, fusion_groups))
        else:
            f.write(calls_to_ir(main_func))
    with open(shapes_path, "w") as f:
        json.dump(_collect_tensor_shapes(main_func), f, indent=2)

    print(f"Main IR saved to: {main_path}")
    print(f"Seq IR saved to: {seq_path}")
    print(f"Shapes saved to: {shapes_path}")


def _collect_tensor_shapes(main_func: T.MainFunc) -> dict:
    shape_map: dict[str, dict] = {}
    for tensor in main_func.input_tensors:
        dims = _shape_to_list(tensor.shape)
        if dims is None:
            continue
        shape_map[tensor.name] = {"shape": dims, "type": "input"}
    for tensor in main_func.intermediate_tensors:
        dims = _shape_to_list(tensor.shape)
        if dims is None:
            continue
        shape_map[tensor.name] = {"shape": dims, "type": "intermediate"}
    for tensor in main_func.output_tensors:
        dims = _shape_to_list(tensor.shape)
        if dims is None:
            continue
        shape_map[tensor.name] = {"shape": dims, "type": "output"}
    return shape_map


def _shape_to_list(shape) -> Optional[List[int]]:
    dims: List[int] = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(dim)
        else:
            return None
    return dims

def print_tir_ir(func_nodes):
    """파싱된 TIR 정보를 깔끔한 텍스트 형태로 출력합니다."""
    for func in func_nodes:
        print(f"Function: {func.name}")
        print(f"  Params: {func.params}")
        
        print(f"  Blocks:")
        for block in func.blocks:
            # Root 블록이 내용이 없으면 생략
            if block.name == 'root' and not block.value and not block.init:
                continue

            print(f"    [Block: {block.name}]")
            
            # Iter Vars 출력
            if block.iter_vars:
                print(f"      Iterators:")
                for name, var in block.iter_vars.items():
                    print(f"        - {var.name}: {var.range} ({var.type}) <- {var.iter_value}")

            # Init 구문 출력
            if block.init:
                print(f"      Init Ops:")
                for idx, op in enumerate(block.init):
                    print(f"        {idx:2d}: {op}")

            # Body 구문 출력
            if block.value:
                print(f"      Body Ops:")
                for idx, op in enumerate(block.value):
                    print(f"        {idx:2d}: {op}")
            
            print("") # 블록 간 공백
        print("-" * 60) # 구분선
        
def save_pretty_tir(func_nodes, output_path):
    """
    파싱된 TIR 정보를 깔끔한 텍스트 포맷으로 파일에 저장합니다.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for func in func_nodes:
                # 1. 함수 헤더
                f.write(f"Function: {func.name}\n")
                f.write(f"  Params: {func.params}\n")
                f.write(f"  Blocks:\n")
                
                for block in func.blocks:
                    # 빈 Root 블록 건너뛰기
                    if block.name == 'root' and not block.value and not block.init:
                        continue

                    # 2. 블록 헤더
                    f.write(f"    [Block: {block.name}]\n")
                    
                    # 3. Iterators 출력
                    if block.iter_vars:
                        f.write(f"      Iterators:\n")
                        for name, var in block.iter_vars.items():
                            f.write(f"        - {var.name}: {var.range} ({var.type}) <- {var.iter_value}\n")

                    # 4. Init 구문 출력
                    if block.init:
                        f.write(f"      Init Ops:\n")
                        for idx, op in enumerate(block.init):
                            f.write(f"        {idx:2d}: {op}\n")

                    # 5. Body 구문 출력
                    if block.value:
                        f.write(f"      Body Ops:\n")
                        for idx, op in enumerate(block.value):
                            f.write(f"        {idx:2d}: {op}\n")
                    
                    f.write("\n") # 블록 간 공백 추가
                
                f.write("-" * 60 + "\n") # 함수 구분선
        
        print(f"✅ Saved pretty text to: {output_path}")

    except Exception as e:
        print(f"❌ Error saving file: {e}")
