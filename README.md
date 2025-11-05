This notebook implements a hierarchical, multi-level directed graph modeling class called `HierarchicalGraph`. It supports rich metadata on both nodes and edges, multi-edge support, clustered + colored visualization, graph merging, selective subgraph viewing, and structured attribute extraction.

## Core Concepts

|           Level | Meaning                                                                                                                                              | Implemented as                                                    |
| --------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Inner Graph** | Detailed node-to-node directed relations (supports parallel edges)                                                                                   | `networkx.MultiDiGraph()`                                         |
| **Outer Graph** | **Roots-only** abstraction (top-level groups). Each inner edge `u→v` is **lifted** to `root(u)→root(v)`; edges where `root(u)==root(v)` are omitted. | `graphviz` rendering from the inner graph (no extra parent nodes) |

### Hierarchy

* Hierarchical containment is defined by the **`parent`** attribute on nodes.
* A node with `parent=None` (or absent) is a **root** (top-level group/container).
* Inner visualization renders clusters (subgraphs) for parents **without drawing extra parent nodes**, and edges to/from parents are attached to their clusters (so you don’t get stray grey duplicates).

### Colors

* You can set `color` per node. If omitted, visualization falls back to white.
* You may still pass a palette on construction; apply it as you prefer in normalization (optional).

## Example Node Format

```python
nodes = [
    {'label': 'Group A', 'type': 'system', 'color': 'yellow', 'description': 'Top A'},
    {'label': 'A', 'parent': 'Group A', 'type': 'system', 'color': 'yellow'},
    {'label': 'B', 'parent': 'Group A', 'type': 'user',   'color': 'lightblue'},
]
```

> Required: `label`
> Optional: `parent`, `type`, `color`, `description`, and any other attributes (preserved).

## Example Edge Format

```python
edges = [
    {'start': 'A', 'end': 'B', 'type': 'control', 'weight': 0.4, 'color': 'blue'}
]
```

> Required: `start`, `end`
> Optional: `type`, `weight`, `color`, `description`, plus arbitrary extra attributes.
> Parallel edges (same `start`/`end` but different `type` or attrs) are supported.

## Usage Example

```python
hg = HierarchicalGraph(nodes, edges)

# Inner view: clustered, hierarchical, multi-edge
hg.visualize_inner_graph_with_clusters()

# Outer view: roots only; edges are lifted to root(u) -> root(v)
hg.visualize_outer_graph()
```

## Outer View — What You’ll See

* **Nodes:** only **root** nodes (those without a `parent`).
* **Edges:** every inner edge `u→v` is mapped to `root(u)→root(v)`.

  * If `root(u) == root(v)`, the edge is **suppressed** in the outer view (it’s an internal detail).
  * Multiple lifted edges between the same roots are allowed (shows parallel cross-group relations).
* **Colors:** taken from the root nodes’ `color` attribute (fallback: white).

## Merge Multiple Graphs

```python
hg1 = HierarchicalGraph(nodes1, edges1)
hg2 = HierarchicalGraph(nodes2, edges2)

hg1.merge(hg2)
```

* Node merging is by `label`.
* Edge merging distinguishes edges by the **triple key** `(start, end, type)`. Matching edges are updated; new ones are appended.

## Subgraph Visualization

```python
hg.visualize_subgraph(['A', 'C', 'F'])
```

Renders the induced subgraph for the provided labels (silently ignores unknown nodes), with colors inherited when available.

## Attribute Fetching

```python
hg.get_node_attributes()
hg.get_edge_attributes()
```

Returns consolidated attributes (implementation-specific helpers).

---

## 📤 CSV Export & Import

This class supports exporting and importing node and edge data using CSV files. This is useful for persistence, versioning, sharing, or editing data externally (e.g., in Excel).

### 🔄 Export Methods

```python
hg.export_nodes_to_csv("nodes.csv")
hg.export_edges_to_csv("edges.csv")
```

* **`export_nodes_to_csv(filename)`** — Saves all current node attributes.
* **`export_edges_to_csv(filename)`** — Saves all edge relationships and attributes.

### 🔄 Import Methods

```python
hg.import_nodes_from_csv("nodes.csv")
hg.import_edges_from_csv("edges.csv")
```

* **`import_nodes_from_csv(filename)`** — Loads node data, validates/normalizes, rebuilds graphs.
* **`import_edges_from_csv(filename)`** — Loads edge data, validates, rebuilds graphs.

> After importing, the internal graphs are **automatically rebuilt**.

### 📝 CSV Structure

#### Nodes CSV (`nodes.csv`)

Must include at minimum:

* `label` (required) — Unique node identifier
* `parent` (optional) — Parent node label (for hierarchy)
* Any other attributes are allowed (e.g., `color`, `type`, `description`, etc.)

#### Edges CSV (`edges.csv`)

Must include at minimum:

* `start` (required) — Source node label
* `end` (required) — Target node label
* Any other attributes are allowed (e.g., `type`, `weight`, `color`, `description`, etc.)
