use rayon::prelude::*;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use trinity::language::SHAPE_TRACKER;
use trinity::shape::ShapeTracker;
use trinity::*;

struct Args {
    ir_path: PathBuf,
    shapes_path: PathBuf,
    output_path: PathBuf,
    semi_output_path: PathBuf,
    max_cost: usize,
    max_num_kernel: usize,
    max_iter: usize,
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} --ir <path> --shapes <path> --output <path> --semi-output <path> [--cost <n>] [--kern <n>] [--iter <n>]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);

    let mut ir_path: Option<PathBuf> = None;
    let mut shapes_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut semi_output_path: Option<PathBuf> = None;
    let mut max_cost = 6usize;
    let mut max_num_kernel = 2usize;
    let mut max_iter = 8usize;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ir" => {
                ir_path = args.next().map(PathBuf::from);
            }
            "--shapes" => {
                shapes_path = args.next().map(PathBuf::from);
            }
            "--output" => {
                output_path = args.next().map(PathBuf::from);
            }
            "--semi-output" => {
                semi_output_path = args.next().map(PathBuf::from);
            }
            "--cost" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "Missing value for --cost".to_string())?;
                max_cost = raw
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --cost value: {raw}"))?;
            }
            "--kern" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "Missing value for --kern".to_string())?;
                max_num_kernel = raw
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --kern value: {raw}"))?;
            }
            "--iter" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "Missing value for --iter".to_string())?;
                max_iter = raw
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --iter value: {raw}"))?;
            }
            "--help" | "-h" => {
                let program = env::args()
                    .next()
                    .unwrap_or_else(|| "trinity_opt".to_string());
                print_usage(&program);
                std::process::exit(0);
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
    }

    Ok(Args {
        ir_path: ir_path.ok_or_else(|| "Missing required --ir".to_string())?,
        shapes_path: shapes_path.ok_or_else(|| "Missing required --shapes".to_string())?,
        output_path: output_path.ok_or_else(|| "Missing required --output".to_string())?,
        semi_output_path: semi_output_path
            .ok_or_else(|| "Missing required --semi-output".to_string())?,
        max_cost,
        max_num_kernel,
        max_iter,
    })
}

fn setup_shape_tracker(shapes: Vec<(String, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(&name, dims);
        }
    });
}

fn load_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read shapes file {}: {err}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|err| format!("Invalid shapes JSON: {err}"))?;
    let obj = json
        .as_object()
        .ok_or_else(|| "Shapes JSON must be an object".to_string())?;

    let mut out = Vec::new();
    for (name, dims_val) in obj {
        let arr = if let Some(arr) = dims_val.as_array() {
            arr
        } else if let Some(obj) = dims_val.as_object() {
            match obj.get("shape").and_then(|shape| shape.as_array()) {
                Some(arr) => arr,
                None => continue,
            }
        } else {
            continue;
        };

        let mut dims = Vec::new();
        let mut ok = true;
        for dim in arr {
            if let Some(n) = dim.as_u64() {
                dims.push(n as usize);
            } else {
                ok = false;
                break;
            }
        }

        if ok {
            out.push((name.clone(), dims));
        }
    }

    if out.is_empty() {
        return Err(format!("No valid tensor shapes found in {}", path.display()));
    }

    Ok(out)
}

fn run(args: Args) -> Result<(), String> {
    if !args.ir_path.exists() {
        return Err(format!("IR file not found: {}", args.ir_path.display()));
    }
    if !args.shapes_path.exists() {
        return Err(format!("Shapes file not found: {}", args.shapes_path.display()));
    }

    let shapes = load_shapes(&args.shapes_path)?;
    setup_shape_tracker(shapes);

    let expr_str = fs::read_to_string(&args.ir_path)
        .map_err(|err| format!("Failed to read IR file {}: {err}", args.ir_path.display()))?;

    println!("Running equality saturation on {}", args.ir_path.display());
    let runner = run_until_saturated(expr_str.as_str(), rules(), args.max_iter);

    if let Some(parent) = args.semi_output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create {}: {err}", parent.display()))?;
    }
    if let Some(parent) = args.output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create {}: {err}", parent.display()))?;
    }

    let semi_output = args
        .semi_output_path
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 semi output path: {}", args.semi_output_path.display()))?;
    let saved = list_expressions_with_target_cost_v3_part1(
        &runner,
        semi_output,
        args.max_cost,
        args.max_num_kernel,
    )
    .map_err(|err| format!("Failed to save semi-expressions: {err}"))?;
    println!("Saved {saved} semi-expressions");

    let (expressions, tile_sets) =
        list_expressions_from_semi_with_cost(&runner, semi_output, usize::MAX)
            .map_err(|err| format!("Failed to load expressions from semi file: {err}"))?;
    println!("Loaded {} final expressions", expressions.len());

    let file = File::create(&args.output_path)
        .map_err(|err| format!("Failed to create {}: {err}", args.output_path.display()))?;
    let mut writer = BufWriter::new(file);

    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{i}: {new_expr}")
        })
        .filter(|line| !line.contains("dummydata"))
        .collect::<Vec<String>>()
        .iter()
        .try_for_each(|line| writeln!(writer, "{line}"))
        .map_err(|err| format!("Failed to write {}: {err}", args.output_path.display()))?;

    writer
        .flush()
        .map_err(|err| format!("Failed to flush {}: {err}", args.output_path.display()))?;

    println!("Wrote expressions to {}", args.output_path.display());
    Ok(())
}

fn main() {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "trinity_opt".to_string());
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}");
            print_usage(&program);
            std::process::exit(2);
        }
    };

    if let Err(err) = run(args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
