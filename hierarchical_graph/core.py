import json, csv
import pandas as pd
import networkx as nx
import graphviz
from IPython.display import display, Image

from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional


class NodeModel(BaseModel):
    label: str
    parent: Optional[str] = None  
    type: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = 'allow'

class EdgeModel(BaseModel):
    start: str
    end: str
    type: str
    weight: Optional[float] = None
    color: Optional[str] = None
    description: Optional[str] = None

    # this validator executes BEFORE type coercion
    @field_validator('*', mode="before")
    def fix_nan(cls, v):
        # convert nan → None
        # pandas blank cell = nan => string validators freak out
        if v != v:   # nan check canonical
            return None
        return v

class HierarchicalGraph:
    def __init__(
        self,
        nodes_data: list[dict],
        edges_data: list[dict],
        default_colors: list[str] | None = None
    ):
        """
        Parameters
        ----------
        nodes_data : list[dict]
            list of node objects. each dict must minimally satisfy pydantic NodeModel
            required fields:
                - label : str
            optional:
                - group : str (default assigned "Default" if missing)
                any additional node attributes are allowed and preserved.

        edges_data : list[dict]
            list of edge objects. each dict must minimally satisfy pydantic EdgeModel
            required fields:
                - start : str
                - end   : str
                - type  : str
            optional:
                weight, color, description and arbitrary extra attributes allowed.

        default_colors : list[str] | None
            optional override palette for groups. If None, internal default palette used.
        """
        self.nodes_data = self._validate_nodes(nodes_data)
        self.edges_data = self._validate_edges(edges_data)
        self._normalize_nodes()

        if not default_colors:
            self.default_colors = [
                'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray',
                'lightsalmon', 'lightcyan', 'wheat', 'plum', 'lightgoldenrod'
            ]
        else:
            self.default_colors = default_colors

        self.inner_graph = nx.MultiDiGraph()
        self.outer_graph = nx.MultiDiGraph()

        self.create_hierarchical_graphs_iterative()

    def _validate_nodes(self, node_list):
        validated = []
        for item in node_list:
            try:
                validated.append(NodeModel(**item).model_dump())
            except ValidationError as e:
                raise ValueError(f"Invalid node data:\n{e}")
        return validated

    def _validate_edges(self, edge_list):
        validated = []
        for item in edge_list:
            try:
                model = EdgeModel.model_validate(item)
                cleaned = model.model_dump(exclude_none=True)  # 💡 omit None values
                validated.append(cleaned)
            except ValidationError as e:
                raise ValueError(f"Invalid edge data:\n{e}")
        return validated

    def _normalize_nodes(self):
        # enforce group presence always
        for node in self.nodes_data:
            if 'group' not in node:
                node['group'] = 'Default'

    def _build_clusters_recursive(self, dot_lines, parent_label, node_map, children_map):
        """
        Emit nested cluster subgraphs. Crucially, if a parent has children,
        we also draw the parent node *inside* its own cluster so edges to that
        node don't create an orphan node outside the cluster.
        """
        if parent_label is not None:
            cluster_id = parent_label.replace(" ", "_")
            dot_lines.append(f'    subgraph cluster_{cluster_id} {{')
            dot_lines.append(f'        label="{parent_label}";')
            bg_color = node_map.get(parent_label, {}).get('color', 'white')
            dot_lines.append(f'        bgcolor="{bg_color}";')

            # draw the parent node itself inside the cluster (prevents grey orphan)
            if parent_label in node_map:
                node_color = node_map[parent_label].get('color', 'white')
                dot_lines.append(f'        "{parent_label}" [fillcolor="{node_color}"];')

        # children: recurse for groups, draw leaves as nodes
        for child in children_map.get(parent_label, []):
            if child in children_map:  # child has its own children => subcluster
                self._build_clusters_recursive(dot_lines, child, node_map, children_map)
            else:
                node_color = node_map[child].get('color', 'white')
                dot_lines.append(f'        "{child}" [fillcolor="{node_color}"];')

        if parent_label is not None:
            dot_lines.append('    }')  # close subgraph



    def create_hierarchical_graphs_iterative(self):
        inner_graph = nx.MultiDiGraph()

        # --- Add nodes to inner graph ---
        for node_data in self.nodes_data:
            label = node_data['label']
            attributes = {k: v for k, v in node_data.items() if k != 'label'}

            # Ensure 'parent' is always present
            if 'parent' not in attributes:
                attributes['parent'] = None

            inner_graph.add_node(label, **attributes)

        # --- Add edges to inner graph ---
        for edge_data in self.edges_data:
            start = edge_data['start']
            end = edge_data['end']
            attributes = {k: v for k, v in edge_data.items() if k not in ['start', 'end']}
            inner_graph.add_edge(start, end, **attributes)

        # --- Create outer MultiDiGraph reflecting recursive parent-child hierarchy ---
        outer_graph = nx.MultiDiGraph()

        # Add all nodes from inner graph
        outer_graph.add_nodes_from(inner_graph.nodes(data=True))

        # Add parent-child edges from 'parent' attribute (hierarchical structure)
        for node, attrs in inner_graph.nodes(data=True):
            parent = attrs.get('parent')
            if parent:
                outer_graph.add_edge(parent, node, relation='hierarchy')

        # Optionally also reflect cross-node functional edges (non-structural)
        for u, v, data in inner_graph.edges(data=True):
            if not outer_graph.has_edge(u, v):
                outer_graph.add_edge(u, v, **data)

        # Assign back
        self.inner_graph = inner_graph
        self.outer_graph = outer_graph


    # --------------------------
    # Merge Graphs
    # --------------------------

    def merge(self, other):
        """
        Merge another HierarchicalGraph into this one.
        - Updates existing nodes and edges with attributes from the other.
        - Adds new nodes and edges if they don't already exist.
        - Rebuilds hierarchical structure.
        """

        # --- Merge Nodes ---
        existing_labels = {node['label'] for node in self.nodes_data}

        for other_node in other.nodes_data:
            label = other_node['label']
            if label in existing_labels:
                # update attributes of existing node
                for node in self.nodes_data:
                    if node['label'] == label:
                        node.update(other_node)
                        break
            else:
                # add new node
                self.nodes_data.append(other_node)

        # --- Merge Edges (distinguish by start, end, type) ---
        existing_edges = {(e['start'], e['end'], e.get('type')) for e in self.edges_data}

        for other_edge in other.edges_data:
            key = (other_edge['start'], other_edge['end'], other_edge.get('type'))
            if key in existing_edges:
                # update attributes of the matching edge
                for edge in self.edges_data:
                    if (edge['start'], edge['end'], edge.get('type')) == key:
                        edge.update(other_edge)
                        break
            else:
                # add new edge
                self.edges_data.append(other_edge)

        # --- Normalize and rebuild ---
        self._normalize_nodes()
        self.create_hierarchical_graphs_iterative()


    # --------------------------
    # Visualization Methods
    # --------------------------
    def visualize_outer_graph(self, filename='outer_level_graph'):
        dot = graphviz.Digraph(comment='Outer Level Graph', engine='dot')
        dot.attr(rankdir='LR')

        # Add nodes with fill colors
        for node, attrs in self.outer_graph.nodes(data=True):
            fillcolor = attrs.get('color', 'white')
            dot.node(str(node), style='filled', fillcolor=fillcolor)

        # Add edges with distinction between hierarchy and functional links
        for u, v, key, data in self.outer_graph.edges(keys=True, data=True):
            relation = data.get('relation', None)

            if relation == 'hierarchy':
                # Parent-child edge (hierarchy)
                dot.edge(str(u), str(v),
                        label='parent',
                        style='dashed',
                        color='gray',
                        penwidth='1.5')
            else:
                # Functional/cross edge
                label = data.get('type', '')
                penwidth = str(data.get('weight', 1.0) * 2)
                color = data.get('color', 'black')
                dot.edge(str(u), str(v),
                        label=label,
                        color=color,
                        penwidth=penwidth)

        filepath = dot.render(filename, format='png', cleanup=True)
        print(f"Outer-level graph saved as {filepath}")
        try:
            display(Image(filename=filepath))
        except:
            print("Open the saved image to view the graph.")

    def visualize_inner_graph_with_clusters(self, filename="inner_level_graph"):
        dot_lines = []
        dot_lines.append('digraph G {')
        dot_lines.append('    rankdir="LR";')
        dot_lines.append('    compound=true;')  # nicer routing across clusters
        dot_lines.append('    node [shape=box, style="filled"];')

        # --- Build node map and children map ---
        node_map = {}       # {label: attributes}
        children_map = {}   # {parent_label: [child_labels]}
        for node, attrs in self.inner_graph.nodes(data=True):
            node_map[node] = attrs
            parent = attrs.get('parent', None)
            children_map.setdefault(parent, []).append(node)

        # --- Build clusters from root (None) using the class helper ---
        self._build_clusters_recursive(dot_lines, None, node_map, children_map)

        # --- Add edges ---
        for u, v, key, data in self.inner_graph.edges(keys=True, data=True):
            label = data.get('type', '')
            penwidth = str(data.get('weight', 1.0) * 2)
            color = data.get('color', 'black')
            dot_lines.append(
                f'    "{u}" -> "{v}" [label="{label}", id="{key}", penwidth={penwidth}, color="{color}"];'
            )

        dot_lines.append('}')
        dot_source = '\n'.join(dot_lines)
        dot = graphviz.Source(dot_source, format='png')
        filepath = dot.render(filename, cleanup=True)
        print(f"Inner-level graph saved as {filepath}")
        try:
            display(Image(filename=filepath))
        except:
            print("Open the saved image to view the graph.")


    def visualize_subgraph(self, node_labels, filename="subgraph"):
        """
        Visualize a subgraph induced by a list of node labels.
        Nodes that don't exist are silently ignored.
        Parameters:
            node_labels (list of str): Nodes to include in the subgraph
            filename (str): Filename for output image
        """
        if not node_labels:
            print("No nodes provided for subgraph.")
            return

        # Filter only nodes that exist in the graph
        valid_nodes = [n for n in node_labels if n in self.inner_graph.nodes]
        if not valid_nodes:
            print("None of the provided nodes exist in the inner graph.")
            return

        # Create subgraph
        subgraph = self.inner_graph.subgraph(valid_nodes).copy()

        dot_lines = []
        dot_lines.append('digraph G {')
        dot_lines.append('    rankdir="LR";')
        dot_lines.append('    node [shape=box, style="filled"];')

        # Add nodes with fallback coloring (own color → parent's color → default)
        for node in subgraph.nodes:
            attrs = subgraph.nodes[node]
            color = attrs.get('color')

            # Try to inherit parent color if own color not defined
            if not color:
                parent = attrs.get('parent')
                color = self.inner_graph.nodes[parent].get('color') if parent in self.inner_graph else 'white'

            dot_lines.append(f'    "{node}" [fillcolor="{color}"];')

        # Add edges with styling
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            label = data.get('type', '')
            penwidth = str(data.get('weight', 1.0) * 2)
            color = data.get('color', 'black')
            dot_lines.append(f'    "{u}" -> "{v}" [label="{label}", id="{key}", penwidth={penwidth}, color="{color}"];')

        dot_lines.append('}')

        dot_source = '\n'.join(dot_lines)
        dot = graphviz.Source(dot_source, format='png')
        filepath = dot.render(filename, cleanup=True)
        print(f"Subgraph saved as {filepath}")
        try:
            display(Image(filename=filepath))
        except:
            print("Open the saved image to view the subgraph.")

    # --------------------------
    # Node Operations
    # --------------------------
    def add_nodes(self, node_data):
        """
        Add one or more nodes.
        Automatically updates group color mapping if new groups are introduced.
        """
        if isinstance(node_data, dict):
            node_data = [node_data]

        self.nodes_data.extend(node_data)
        self.create_hierarchical_graphs_iterative()

    def edit_nodes(self, label_or_labels, new_data):
        """
        Edit one or more nodes by label.
        Parameters:
            label_or_labels: str or list of str
            new_data: dict of attributes to update
        """
        if isinstance(label_or_labels, str):
            label_or_labels = [label_or_labels]

        for node in self.nodes_data:
            if node['label'] in label_or_labels:
                node.update(new_data)

        self.create_hierarchical_graphs_iterative()

    def delete_nodes(self, label_or_labels):
        """
        Delete one or more nodes and remove any edges involving them.
        Parameters:
            label_or_labels: str or list of str
        """
        if isinstance(label_or_labels, str):
            label_or_labels = [label_or_labels]

        self.nodes_data = [node for node in self.nodes_data if node['label'] not in label_or_labels]
        self.edges_data = [edge for edge in self.edges_data if edge['start'] not in label_or_labels and edge['end'] not in label_or_labels]
        self.create_hierarchical_graphs_iterative()

    # --------------------------
    # Edge Operations
    # --------------------------
    def add_edges(self, edge_data):
        """Add one or more edges."""
        if isinstance(edge_data, list):
            self.edges_data.extend(edge_data)
        else:
            self.edges_data.append(edge_data)
        self.create_hierarchical_graphs_iterative()

    def edit_edges(self, start_end_pairs, new_data):
        """
        Edit one or more edges.
        `start_end_pairs`: tuple or list of tuples like [(start, end), ...]
        `new_data`: dict with new edge attributes
        """
        if isinstance(start_end_pairs, tuple):
            start_end_pairs = [start_end_pairs]

        for start, end in start_end_pairs:
            for edge in self.edges_data:
                if edge['start'] == start and edge['end'] == end:
                    edge.update(new_data)
                    break
        self.create_hierarchical_graphs_iterative()

    def delete_edges(self, start_end_pairs):
        """
        Delete one or more edges.
        `start_end_pairs`: tuple or list of tuples like [(start, end), ...]
        """
        if isinstance(start_end_pairs, tuple):
            start_end_pairs = [start_end_pairs]

        self.edges_data = [
            edge for edge in self.edges_data
            if (edge['start'], edge['end']) not in start_end_pairs
        ]
        self.create_hierarchical_graphs_iterative()


 

    #----Get Node, edge attributes

    def get_node_attributes(self, labels=None):
        """
        Get attributes of one or more nodes.
        If labels is None or [] → return ALL node attributes.
        Parameters:
            labels (str or list of str or None)
        Returns:
            dict: {label: attributes}
        """

        # return ALL
        if labels is None or labels == []:
            return {label: dict(attrs) for label, attrs in self.inner_graph.nodes(data=True)}

        # normalize input
        if isinstance(labels, str):
            labels = [labels]

        results = {}
        for label in labels:
            if label in self.inner_graph.nodes:
                results[label] = dict(self.inner_graph.nodes[label])
            else:
                results[label] = None
        return results

    def get_edge_attributes(self, edge_tuples=None):
        """
        Get attributes of one or more edges.
        If edge_tuples is None or [] → return ALL edges in the graph
        Parameters:
            edge_tuples: tuple or list of tuples [(start,end),...]
        Returns:
            dict: {(start, end, key): attributes}
        """

        results = {}

        # return ALL edges
        if edge_tuples is None or edge_tuples == []:
            for u, v, key, attrs in self.inner_graph.edges(keys=True, data=True):
                results[(u, v, key)] = dict(attrs)
            return results

        # normalize input
        if isinstance(edge_tuples, tuple):
            edge_tuples = [edge_tuples]

        # specific subset edges
        for start, end in edge_tuples:
            if self.inner_graph.has_edge(start, end):
                for key, attrs in self.inner_graph[start][end].items():
                    results[(start, end, key)] = dict(attrs)
            else:
                results[(start, end, None)] = None
        return results

    #-----export/import---------

    # Method to export graph data to JSON
    def export_json(self, path):
        """
        Export the graph data (nodes and edges) to a JSON file.
        """
        data = {
            'nodes': self.nodes_data,
            'edges': self.edges_data
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Graph exported to {path}")

    # Method to load a graph from JSON
    def load_from_json(path):
        """
        Load a HierarchicalGraph from a JSON file.
        Returns:
            HierarchicalGraph instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return HierarchicalGraph(data['nodes'], data['edges'])

    def export_nodes_to_csv(self, filename='nodes.csv'):
        df = pd.DataFrame(self.nodes_data)
        df.to_csv(filename, index=False)
        print(f"Nodes exported to {filename}")

    def export_edges_to_csv(self, filename='edges.csv'):
        df = pd.DataFrame(self.edges_data)
        df.to_csv(filename, index=False)
        print(f"Edges exported to {filename}")

    def import_nodes_from_csv(self, filename):
        df = pd.read_csv(filename)
        raw_nodes = df.to_dict(orient='records')
        self.nodes_data = self._validate_nodes(raw_nodes)
        self._normalize_nodes()
        self.create_hierarchical_graphs_iterative()
        print(f"Nodes loaded from {filename}")


    def import_edges_from_csv(self, filename):
        df = pd.read_csv(filename)
        raw_edges = df.to_dict(orient='records')
        self.edges_data = self._validate_edges(raw_edges)
        self.create_hierarchical_graphs_iterative()
        print(f"Edges loaded from {filename}")

