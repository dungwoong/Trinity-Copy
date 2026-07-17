use crate::language::{LoopAnalysis, TileLang};
use egg::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Save the raw egraph structure to a file with proper ID references
pub fn save_raw_egraph(
    runner: &Runner<TileLang, LoopAnalysis>,
    filepath: &str,
) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(File::create(filepath)?);

    // Save metadata
    writeln!(file, "# Raw EGraph Save File V3")?;
    writeln!(file, "# Iterations: {}", runner.iterations.len())?;
    writeln!(
        file,
        "# Total nodes: {}",
        runner.egraph.total_number_of_nodes()
    )?;
    writeln!(
        file,
        "# Total classes: {}",
        runner.egraph.number_of_classes()
    )?;
    writeln!(file, "# ===== BEGIN EGRAPH =====")?;

    // Create a deterministic ordering of classes by ID
    let mut class_ids: Vec<Id> = runner.egraph.classes().map(|c| c.id).collect();
    class_ids.sort_by_key(|id| Into::<usize>::into(*id));

    // First, we need to save all eclasses with their nodes in a way that preserves ID references
    for &class_id in &class_ids {
        if let Some(class) = runner.egraph.classes().find(|c| c.id == class_id) {
            let canonical_id = runner.egraph.find(class.id);
            writeln!(file, "CLASS {} {}", class.id, canonical_id)?;

            // Save nodes count for this class
            let nodes: Vec<_> = class.iter().collect();
            writeln!(file, "NODES_COUNT {}", nodes.len())?;

            // For each node in the class, we need to save its structure with ID references
            for node in nodes {
                // Convert the node to a string that preserves ID references
                let node_str = format_node_with_ids(&runner.egraph, node);
                writeln!(file, "NODE {}", node_str)?;
            }

            writeln!(file)?;
        }
    }

    writeln!(file, "# ===== END EGRAPH =====")?;

    // Save roots
    writeln!(file, "# ===== ROOTS =====")?;
    for (i, &root_id) in runner.roots.iter().enumerate() {
        writeln!(file, "ROOT {} {}", i, root_id)?;
    }

    println!(
        "Saved raw egraph with {} classes to {}",
        runner.egraph.number_of_classes(),
        filepath
    );

    Ok(())
}

// Helper function to format a node with ID references instead of expanded expressions
fn format_node_with_ids(_egraph: &EGraph<TileLang, LoopAnalysis>, node: &TileLang) -> String {
    use std::fmt::Write;
    use TileLang::*;
    let mut result = String::new();

    // Check if it's a leaf node (no children)
    let children: Vec<Id> = node.children().iter().cloned().collect();

    if children.is_empty() {
        // Leaf node - just write the node itself
        write!(&mut result, "{}", node).unwrap();
    } else {
        // Non-leaf node - write with ID references
        // Get the operator name based on the node variant
        let op_name = match node {
            Loop(_) => "loop",
            DLoop(_) => "dloop",
            TLoop(_) => "tmp_loop",
            Input(_) => "input",
            Output(_) => "output",
            Tensor(_) => "tensor",
            Tile(_) => "tile",
            ConstTile(_) => "const_tile",
            FullTile => "fulltile",
            Elem(_) => "elem",
            Index(_) => "index",
            Load(_) => "load",
            Store(_) => "store",
            Seq(_) => "seq",
            Const(_) => "const",
            Add(_) => "+",
            Sub(_) => "-",
            Mul(_) => "*",
            Div(_) => "/",
            Le(_) => "<=",
            Max(_) => "max",
            Min(_) => "min",
            Exp(_) => "exp",
            Matmul(_) => "@",
            ReduceSum(_) => "rsum",
            ReduceMin(_) => "rmin",
            ReduceMax(_) => "rmax",
            Sqr(_) => "sqr",
            Sqrt(_) => "sqrt",
            Sigmoid(_) => "sigmoid",
            Erf(_) => "erf",
            Abs(_) => "abs",
            Cast(_) => "cast",
            Concat(_) => "concat",
            Broadcast(_) => "bcast",
            Transpose(_) => "transpose",
            Permute3(_) => "permute3",
            Permute4(_) => "permute4",
            Squeeze(_) => "squeeze",
            Unsqueeze(_) => "unsqueeze",
            Dummy => "dummy",
            SLoop(_) => "sloop",
            PLoop(_) => "ploop",
            Num(_) => return format!("{}", node),
            Var(_) => return format!("{}", node),
        };

        write!(&mut result, "({}", op_name).unwrap();

        for child in children {
            write!(&mut result, " {}", child).unwrap();
        }

        write!(&mut result, ")").unwrap();
    }

    result
}

/// Load raw egraph structure from a file
///
/// Note: The loaded egraph will be semantically equivalent to the original
/// (same equivalences and number of classes), but will contain additional placeholder
/// nodes to handle circular dependencies. The internal IDs will also differ.
/// This is a limitation of the egg library which doesn't allow direct control
/// over node IDs or removal of nodes after insertion.
pub fn load_raw_egraph(filepath: &str) -> Result<EGraph<TileLang, LoopAnalysis>, Box<dyn Error>> {
    use crate::language::TileLang;
    use std::io::{BufRead, BufReader};

    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let mut egraph = EGraph::<TileLang, LoopAnalysis>::default();

    // First pass: collect all classes and their nodes
    #[derive(Debug)]
    struct ClassData {
        id: Id,
        _canonical_id: Id,
        nodes: Vec<String>,
    }

    let mut classes: Vec<ClassData> = Vec::new();
    let mut current_class: Option<ClassData> = None;
    let mut roots: Vec<Id> = Vec::new();

    // Parse the file
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse CLASS line
        if line.starts_with("CLASS ") {
            // Save previous class
            if let Some(class) = current_class.take() {
                classes.push(class);
            }

            // Parse new class
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(id), Ok(canonical_id)) =
                    (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                {
                    current_class = Some(ClassData {
                        id: Id::from(id),
                        _canonical_id: Id::from(canonical_id),
                        nodes: Vec::new(),
                    });
                }
            }
        }
        // Parse NODE line
        else if line.starts_with("NODE ") {
            if let Some(ref mut class) = current_class {
                class.nodes.push(line[5..].to_string());
            }
        }
        // Parse ROOT line
        else if line.starts_with("ROOT ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(old_id) = parts[2].parse::<usize>() {
                    roots.push(Id::from(old_id));
                }
            }
        }
    }

    // Save last class
    if let Some(class) = current_class {
        classes.push(class);
    }

    // Step 1: Create placeholder nodes for each class to handle circular dependencies
    let mut old_to_new: HashMap<Id, Id> = HashMap::new();

    // Create a unique placeholder for each class
    for class in &classes {
        // Use a special symbol as placeholder
        let placeholder = format!("__placeholder_{}", class.id);
        let expr: RecExpr<TileLang> = placeholder.parse().unwrap_or_else(|_| {
            // If parsing fails, create a simple variable node
            format!("placeholder_{}", class.id).parse().unwrap()
        });
        let placeholder_id = egraph.add_expr(&expr);
        old_to_new.insert(class.id, placeholder_id);
    }

    // Step 2: Now parse all nodes with placeholders available for all references
    let mut class_nodes: HashMap<Id, Vec<Id>> = HashMap::new();

    for class in &classes {
        let mut nodes_for_class = Vec::new();

        for node_str in &class.nodes {
            if let Some(node_id) = parse_node(&mut egraph, node_str, &old_to_new) {
                nodes_for_class.push(node_id);
            }
        }

        if !nodes_for_class.is_empty() {
            class_nodes.insert(class.id, nodes_for_class);
        }
    }

    // Step 3: Union all nodes that belong to the same class, including the placeholder
    for class in &classes {
        if let Some(nodes) = class_nodes.get(&class.id) {
            let placeholder_id = old_to_new[&class.id];

            // Union all nodes with the placeholder
            for &node_id in nodes {
                egraph.union(placeholder_id, node_id);
            }

            // Update the mapping to the canonical ID after unions
            let canonical = egraph.find(placeholder_id);
            old_to_new.insert(class.id, canonical);
        }
    }

    // Final rebuild to ensure consistency
    egraph.rebuild();

    println!(
        "Loaded egraph with {} classes from {}",
        egraph.number_of_classes(),
        filepath
    );

    Ok(egraph)
}

// Helper function to parse a node string with ID references
fn parse_node(
    egraph: &mut EGraph<TileLang, LoopAnalysis>,
    node_str: &str,
    old_to_new: &HashMap<Id, Id>,
) -> Option<Id> {
    use crate::language::TileLang::*;

    // If it's a simple atom (no parentheses), parse it directly
    if !node_str.starts_with('(') {
        match node_str.parse::<RecExpr<TileLang>>() {
            Ok(expr) => Some(egraph.add_expr(&expr)),
            Err(_) => {
                // Don't print error for atoms, they might be parsed already
                None
            }
        }
    } else {
        // Parse complex expression with ID references
        // Format: (op id1 id2 ...)
        let node_str = node_str.trim_start_matches('(').trim_end_matches(')');
        let parts: Vec<&str> = node_str.split_whitespace().collect();

        if parts.is_empty() {
            return None;
        }

        let op = parts[0];

        // Convert ID references to actual IDs
        let mut child_ids: Vec<Id> = Vec::new();
        for part in &parts[1..] {
            if let Ok(id_num) = part.parse::<usize>() {
                let old_id = Id::from(id_num);
                if let Some(&new_id) = old_to_new.get(&old_id) {
                    child_ids.push(new_id);
                } else {
                    // ID not found yet - defer this node
                    return None;
                }
            }
        }

        // Create the appropriate TileLang node based on the operator
        let node = match op {
            "loop" if child_ids.len() >= 5 => Loop([
                child_ids[0],
                child_ids[1],
                child_ids[2],
                child_ids[3],
                child_ids[4],
            ]),
            "seq" if child_ids.len() >= 2 => Seq([child_ids[0], child_ids[1]]),
            "store" if child_ids.len() >= 3 => Store([child_ids[0], child_ids[1], child_ids[2]]),
            "load" if child_ids.len() >= 2 => Load([child_ids[0], child_ids[1]]),
            "+" if child_ids.len() >= 2 => Add([child_ids[0], child_ids[1]]),
            "*" if child_ids.len() >= 2 => Mul([child_ids[0], child_ids[1]]),
            "@" if child_ids.len() >= 2 => Matmul([child_ids[0], child_ids[1]]),
            "index" => {
                // index takes a Box<[Id]>
                Index(child_ids.clone().into_boxed_slice())
            }
            "tile" if child_ids.len() >= 1 => Tile(child_ids[0]),
            "input" if child_ids.len() >= 1 => Input(child_ids[0]),
            "output" if child_ids.len() >= 1 => Output(child_ids[0]),
            "tensor" if child_ids.len() >= 1 => Tensor(child_ids[0]),
            "const_tile" if child_ids.len() >= 2 => ConstTile([child_ids[0], child_ids[1]]),
            "fulltile" => FullTile,
            "elem" if child_ids.len() >= 1 => Elem(child_ids[0]),
            "const" if child_ids.len() >= 1 => Const(child_ids[0]),
            "-" if child_ids.len() >= 2 => Sub([child_ids[0], child_ids[1]]),
            "/" if child_ids.len() >= 2 => Div([child_ids[0], child_ids[1]]),
            "<=" if child_ids.len() >= 2 => Le([child_ids[0], child_ids[1]]),
            "max" if child_ids.len() >= 2 => Max([child_ids[0], child_ids[1]]),
            "min" if child_ids.len() >= 2 => Min([child_ids[0], child_ids[1]]),
            "exp" if child_ids.len() >= 1 => Exp(child_ids[0]),
            "rsum" if child_ids.len() >= 2 => ReduceSum([child_ids[0], child_ids[1]]),
            "rmin" if child_ids.len() >= 2 => ReduceMin([child_ids[0], child_ids[1]]),
            "rmax" if child_ids.len() >= 2 => ReduceMax([child_ids[0], child_ids[1]]),
            "sqr" if child_ids.len() >= 1 => Sqr(child_ids[0]),
            "sqrt" if child_ids.len() >= 1 => Sqrt(child_ids[0]),
            "sigmoid" if child_ids.len() >= 1 => Sigmoid(child_ids[0]),
            "erf" if child_ids.len() >= 1 => Erf(child_ids[0]),
            "abs" if child_ids.len() >= 1 => Abs(child_ids[0]),
            "cast" if child_ids.len() >= 2 => Cast([child_ids[0], child_ids[1]]),
            "concat" if child_ids.len() >= 3 => Concat([child_ids[0], child_ids[1], child_ids[2]]),
            "bcast" if child_ids.len() >= 2 => Broadcast([child_ids[0], child_ids[1]]),
            "transpose" if child_ids.len() >= 1 => {
                Transpose(child_ids[0])
            }
            "permute3" if child_ids.len() >= 4 => {
                Permute3([child_ids[0], child_ids[1], child_ids[2], child_ids[3]])
            }
            "permute4" if child_ids.len() >= 5 => {
                Permute4([
                    child_ids[0],
                    child_ids[1],
                    child_ids[2],
                    child_ids[3],
                    child_ids[4],
                ])
            }
            "squeeze" if child_ids.len() >= 2 => Squeeze([child_ids[0], child_ids[1]]),
            "unsqueeze" if child_ids.len() >= 2 => Unsqueeze([child_ids[0], child_ids[1]]),
            "dummy" => Dummy,
            "dloop" if child_ids.len() >= 5 => DLoop([
                child_ids[0],
                child_ids[1],
                child_ids[2],
                child_ids[3],
                child_ids[4],
            ]),
            "tmp_loop" if child_ids.len() >= 6 => TLoop([
                child_ids[0],
                child_ids[1],
                child_ids[2],
                child_ids[3],
                child_ids[4],
                child_ids[5],
            ]),
            "sloop" if child_ids.len() >= 5 => SLoop([
                child_ids[0],
                child_ids[1],
                child_ids[2],
                child_ids[3],
                child_ids[4],
            ]),
            "ploop" if child_ids.len() >= 5 => PLoop([
                child_ids[0],
                child_ids[1],
                child_ids[2],
                child_ids[3],
                child_ids[4],
            ]),
            _ => {
                eprintln!(
                    "Unknown operator or wrong arity: {} with {} children",
                    op,
                    child_ids.len()
                );
                return None;
            }
        };

        Some(egraph.add(node))
    }
}

/// Save egraph in binary format using bincode (requires serde feature)
#[cfg(feature = "serde-1")]
pub fn save_egraph_binary(
    runner: &Runner<TileLang, LoopAnalysis>,
    filepath: &str,
) -> Result<(), Box<dyn Error>> {
    use bincode;
    use std::io::BufWriter;

    let file = File::create(filepath)?;
    let writer = BufWriter::new(file);

    // Note: This requires TileLang and LoopAnalysis to implement Serialize
    // You may need to implement custom serialization
    bincode::serialize_into(writer, &runner.egraph)?;

    println!("Saved egraph in binary format to {}", filepath);
    Ok(())
}

/// Load egraph from binary format
#[cfg(feature = "serde-1")]
pub fn load_egraph_binary(
    filepath: &str,
) -> Result<EGraph<TileLang, LoopAnalysis>, Box<dyn Error>> {
    use bincode;
    use std::io::BufReader;

    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let egraph = bincode::deserialize_from(reader)?;

    println!("Loaded egraph from binary format: {}", filepath);
    Ok(egraph)
}
