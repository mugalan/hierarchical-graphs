# HierarchicalGraph Python Class

This project implements a hierarchical, multi-level directed graph modeling class called `HierarchicalGraph`. It supports rich metadata on both nodes and edges, multi-edge support, clustered + colored visualization, graph merging, selective subgraph viewing and structured attribute extraction.

## Core Concepts

| Level | Meaning | Implemented as |
|-------|----------|----------------|
| Inner Graph | Detailed node-to-node directed relations (multi-edge) | `networkx.MultiDiGraph()` |
| Outer Graph | Group-to-group directed abstraction | `networkx.MultiDiGraph()` |

Nodes are grouped by attribute `group`. If missing, it defaults to `"Default"`.

Group colors are consistent and auto-managed.

## Example Node Format

```python
nodes = [
    {'label': 'A', 'group': 'Group A', 'type': 'system', 'color':'yellow'},
    {'label': 'B', 'group': 'Group A', 'type':'user','color':'lightblue'},
]
```

## Example Edge Format

```python
edges = [
    {'start': 'A', 'end': 'B', 'type': 'control', 'weight': 0.4, 'color':'blue'}
]
```

## Usage Example

```python
hg = HierarchicalGraph(nodes, edges)
hg.visualize_inner_graph_with_clusters()
hg.visualize_outer_graph()
```

## Merge Multiple Graphs

```python
hg1 = HierarchicalGraph(nodes1, edges1)
hg2 = HierarchicalGraph(nodes2, edges2)

hg1.merge(hg2)
```

## Subgraph Visualization

```python
hg.visualize_subgraph(['A', 'C', 'F'])
```

## Attribute Fetching

```python
hg.get_node_attributes()
hg.get_edge_attributes()
```

Here’s a ready-to-paste `README.md` section you can add to document the **CSV export/import features** in your `HierarchicalGraph` class:

---

## 📤 CSV Export & Import

This class supports exporting and importing node and edge data using CSV files. This is useful for persistence, versioning, sharing, or editing data externally (e.g., in Excel).

### 🔄 Export Methods

You can save the current state of the graph's nodes and edges as CSV files:

```python
hg.export_nodes_to_csv("nodes.csv")
hg.export_edges_to_csv("edges.csv")
```

* **`export_nodes_to_csv(filename)`**
  Saves all current node attributes to a CSV file.

* **`export_edges_to_csv(filename)`**
  Saves all edge relationships and their attributes to a CSV file.

### 🔄 Import Methods

You can reload and replace the graph’s structure from CSV files:

```python
hg.import_nodes_from_csv("nodes.csv")
hg.import_edges_from_csv("edges.csv")
```

* **`import_nodes_from_csv(filename)`**
  Loads node data from a CSV file, normalizes it, and regenerates the graph.

* **`import_edges_from_csv(filename)`**
  Loads edge data from a CSV file and regenerates the graph.

> ⚠️ After importing, the internal graphs are **automatically rebuilt** to reflect the new data.

### 📝 CSV Structure

#### Nodes CSV (`nodes.csv`)

Must include at minimum:

* `label` (required) – Unique identifier for the node
* `group` (optional) – Group name for clustering
* Any other attributes are allowed (e.g., `color`, `type`, `description`, etc.)

#### Edges CSV (`edges.csv`)

Must include at minimum:

* `start` (required) – Source node label
* `end` (required) – Target node label
* Any other attributes are allowed (e.g., `type`, `weight`, `color`, etc.)

---

Let me know if you'd like a downloadable CSV template or validation steps added too.

