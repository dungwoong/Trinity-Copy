//! E-graph visualization utilities

use crate::language::LoopAnalysis;
use crate::language::TileLang;
use egg::Runner;
use std::fs::File;
use std::io::Write;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

pub fn save_egraph(runner: &Runner<TileLang, LoopAnalysis>, filename: &str) {
    let dot_string = runner.egraph.dot().to_string();
    let mut file = File::create(filename).expect("Failed to create dot file");
    file.write_all(dot_string.as_bytes())
        .expect("Failed to write dot file");
}

pub fn save_egraph_from_egraph(egraph: &EGraph, filename: &str) {
    let dot_string = egraph.dot().to_string();
    let mut file = File::create(filename).expect("Failed to create dot file");
    file.write_all(dot_string.as_bytes())
        .expect("Failed to write dot file");
}
