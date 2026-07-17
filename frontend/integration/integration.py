import torch
from tvm import relax
from tvm.relax.frontend.torch import from_fx
# from .frontend import TIRToTrinityVisitor  
# from .backend import TrinityCompiler       

def _extract_input_info(example_inputs):
    """
    List[Tensor] -> [(shape, dtype)]
    """
    input_info = []
    for i, inp in enumerate(example_inputs):
        name = f"input_{i}"
        shape = tuple(inp.shape)
        dtype = str(inp.dtype).split(".")[-1]
        input_info.append((shape, dtype))
    return input_info

def trinity_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """
    Custom backend for PyTorch torch.compile
    """
    print(f"\n>>> [Trinity Backend] Received Graph from Dynamo")
    
    # 1. PyTorch FX Graph -> TVM Relax 
    input_info = _extract_input_info(example_inputs)
    mod = from_fx(gm, input_info)
    
    # 2. Relax -> TIR 
    mod = relax.transform.LegalizeOps()(mod)
    
    print(mod.script())
    
    kernel_binaries = None
    #TODO 3. TIR -> Trinity IR (Translation)
    # visitor = TIRToTrinityVisitor()
    # trinity_ir_map = {}
    
    # for gv, func in mod.functions.items():ㅆ
    #     if isinstance(func, tvm.tir.PrimFunc):
    #         func_name = gv.name_hint
    #         # Visitor를 통해 Trinity S-Expr 문자열 생성
    #         ir_code = visitor.generate(func)
    #         trinity_ir_map[func_name] = ir_code
            
    #         print(f" -> Generated Trinity IR for kernel: {func_name}")

    # # -----------------------------------------------------------
    # # 4. Trinity Compiler 호출 (Compilation)
    # # -----------------------------------------------------------
    # compiler = TrinityCompiler()
    # # 컴파일된 커널 객체들 (함수 포인터 등)을 받음
    # kernel_binaries = compiler.compile(trinity_ir_map)
    
    # # -----------------------------------------------------------
    # # 5. 실행 래퍼(Runner) 생성 및 반환
    # # -----------------------------------------------------------
    # # PyTorch는 이 반환된 함수를 실행합니다.
    # # 주의: 여기서는 단순화를 위해 메인 커널 하나만 있다고 가정합니다.
    # # 실제로는 Relax 그래프의 순서대로 커널을 호출하는 런타임(VM)이 필요할 수 있습니다.
    
    def runner(*args):
        # 컴파일된 커널 중 메인 함수 실행
        # (이름은 상황에 따라 'main' 혹은 'matmul' 등으로 달라질 수 있음)
        target_kernel = list(kernel_binaries.values())[0]
        
        # 연구자님의 커널 실행 (Input Tensors -> Output Tensor)
        return target_kernel(*args)

    # print(">>> [Trinity Backend] Compilation Done. Returning Runner.")
    # return runner
    
    return gm.forward