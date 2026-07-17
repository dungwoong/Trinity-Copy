// Track tensor/tile shape
use std::collections::HashMap;

// Represents a dimension that can be either concrete or wildcard
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Concrete(usize),
    Wildcard, // To be inferred from context (e.g., broadcast dimension)
}

// Represents the shape of a tensor or tile
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    pub dims: Vec<Dimension>,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(Dimension::Concrete).collect(),
        }
    }

    pub fn new_with_dims(dims: Vec<Dimension>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn total_elements(&self) -> Option<usize> {
        let mut total = 1;
        for dim in &self.dims {
            match dim {
                Dimension::Concrete(size) => total *= size,
                Dimension::Wildcard => return None, // Cannot compute if wildcard present
            }
        }
        Some(total)
    }
}

// Track tensor shape for given tensor base
pub struct ShapeTracker {
    pub tensor_shapes: HashMap<String, TensorShape>,
    pub tile_size: usize,
}

impl ShapeTracker {
    pub fn new() -> Self {
        Self {
            tensor_shapes: HashMap::new(),
            tile_size: 64,
        }
    }
    pub fn add_tensor(&mut self, name: &str, dims: Vec<usize>) {
        self.tensor_shapes
            .insert(name.to_string(), TensorShape::new(dims));
    }
    pub fn get_tensor_shape(&self, name: &str) -> Option<&TensorShape> {
        self.tensor_shapes.get(name)
    }

    pub fn get_tile_size(&self, full_size: usize) -> usize {
        full_size.min(self.tile_size)
    }

    pub fn compute_tile_shape(
        &self,
        tensor_shape: &TensorShape,
        tile_indices: &[TileIndex],
    ) -> TensorShape {
        let mut tile_dims = Vec::new();

        for (i, tile_idx) in tile_indices.iter().enumerate() {
            if i < tensor_shape.dims.len() {
                match &tensor_shape.dims[i] {
                    Dimension::Concrete(full_dim) => {
                        let tile_size = match tile_idx {
                            TileIndex::FullTile => Dimension::Concrete(*full_dim),
                            TileIndex::Tile => Dimension::Concrete(self.get_tile_size(*full_dim)),
                            TileIndex::ConstTile(_start, interval) => {
                                Dimension::Concrete(*interval)
                            }
                            TileIndex::Elem => Dimension::Concrete(1),
                        };
                        tile_dims.push(tile_size);
                    }
                    Dimension::Wildcard => {
                        // Wildcard dimensions remain wildcard in tiles
                        tile_dims.push(Dimension::Wildcard);
                    }
                }
            }
        }

        TensorShape::new_with_dims(tile_dims)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileIndex {
    FullTile,
    Tile,
    ConstTile(usize, usize),
    Elem,
}
