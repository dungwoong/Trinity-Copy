import argparse
import importlib
from typing import Any

from utils.pipeline import export_model_ir


def _load_factory(module_name: str, factory_name: str) -> Any:
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise AttributeError(f"Factory '{factory_name}' not found in module '{module_name}'")
    return factory()


def _parse_factory_result(
    result: Any, module_name: str, basename: str | None
) -> tuple[Any, Any, str, str | None, bool, bool, int | None, float | None]:
    context = None
    inline_shape_op = True
    inline_elementwise_op = True
    if isinstance(result, dict):
        model = result["model"]
        example_inputs = result["example_inputs"]
        basename = result.get("basename", basename or module_name.split(".")[-1])
        context = result.get("context")
        inline_shape_op = result.get("inline_shape_op", True)
        inline_elementwise_op = result.get("inline_elementwise_op", True)
        remove_short_loop_threshold = result.get("remove_short_loop_threshold")
        decompose_nested_op_ratio = result.get("decompose_nested_op_ratio")
        return (
            model,
            example_inputs,
            basename,
            context,
            inline_shape_op,
            inline_elementwise_op,
            remove_short_loop_threshold,
            decompose_nested_op_ratio,
        )
    if isinstance(result, tuple):
        if len(result) == 2:
            model, example_inputs = result
            basename = basename or module_name.split(".")[-1]
            return (
                model,
                example_inputs,
                basename,
                context,
                inline_shape_op,
                inline_elementwise_op,
                None,
                None,
            )
        if len(result) == 3:
            model, example_inputs, basename_from_factory = result
            return (
                model,
                example_inputs,
                basename or basename_from_factory,
                context,
                inline_shape_op,
                inline_elementwise_op,
                None,
                None,
            )
    raise ValueError("Factory must return (model, example_inputs), (model, example_inputs, basename), or dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to Trinity IR.")
    parser.add_argument("--module", required=True, help="Python module path, e.g. model.DecAttn")
    parser.add_argument("--factory", default="build_model_and_inputs", help="Factory function in the module")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory root")
    parser.add_argument("--basename", default=None, help="Basename for output files")
    parser.add_argument("--context", default=None, help="Validation context name override")
    parser.add_argument("--no-inline-shape", action="store_true", help="Disable shape-op inlining")
    parser.add_argument("--no-inline-elementwise", action="store_true", help="Disable elementwise inlining")
    parser.add_argument(
        "--remove-short-loop-threshold",
        type=int,
        default=None,
        help="Max loop extent to replace with fulltile in remove_short_loop_nodes",
    )
    parser.add_argument(
        "--decompose-nested-op-ratio",
        type=float,
        default=None,
        help="Temp tensor ratio for decompose operations",
    )
    args = parser.parse_args()

    result = _load_factory(args.module, args.factory)
    (
        model,
        example_inputs,
        basename,
        context,
        inline_shape_op,
        inline_elementwise_op,
        remove_short_loop_threshold,
        decompose_nested_op_ratio,
    ) = _parse_factory_result(
        result, args.module, args.basename
    )
    if args.context is not None:
        context = args.context
    if args.no_inline_shape:
        inline_shape_op = False
    if args.no_inline_elementwise:
        inline_elementwise_op = False
    if args.remove_short_loop_threshold is not None:
        remove_short_loop_threshold = args.remove_short_loop_threshold
    if args.decompose_nested_op_ratio is not None:
        decompose_nested_op_ratio = args.decompose_nested_op_ratio
    if remove_short_loop_threshold is None:
        remove_short_loop_threshold = 64
    if decompose_nested_op_ratio is None:
        decompose_nested_op_ratio = 0.3

    export_model_ir(
        model,
        example_inputs,
        basename=basename,
        output_dir=args.output_dir,
        context=context,
        inline_shape_op=inline_shape_op,
        inline_elementwise_op=inline_elementwise_op,
        remove_short_loop_threshold=remove_short_loop_threshold,
        decompose_nested_op_ratio=decompose_nested_op_ratio,
    )


if __name__ == "__main__":
    main()
