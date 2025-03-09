from itertools import accumulate

import networkx as nx
import matplotlib.pyplot as plt
import ngsolve as ngs


class GeometryDescription:
    """TODO
    """
    def __init__(self, mesh: ngs.Mesh) -> None:
        """Create a new geometry description.
        Args:
            mesh: The mesh to create the geometry description from.
        """
        self.mesh = mesh

        # Add all other compartments as nodes (with special node for 'outside of
        # computational domain')
        regions = {
            region: name
            for name in set(mesh.GetMaterials())
            for region in mesh.Materials(name).Split()
        }
        self.regions: tuple[str, ...] = tuple(set(regions.values())) + ('no_domain',)

        # Cluster regions into compartments
        compartments = set(_extract_compartment(region) for region in self.regions)
        self.compartments: tuple[str, ...] = tuple(compartments)

        # Store all edges as (left, right, name)
        boundaries = {
            bnd: name
            for name in set(mesh.GetBoundaries())
            for bnd in mesh.Boundaries(name).Split()
        }
        self._membrane_connectivity = set()
        for bnd, name in boundaries.items():
            neighbors = bnd.Neighbours(ngs.VOL).Split()

            n1 = regions[neighbors[0]]
            n2 = regions[neighbors[1]] if len(neighbors) > 1 else 'no_domain'

            self._membrane_connectivity.add((n1, n2, name))

        # Store membrane names
        membranes = set(name for _, _, name in self._membrane_connectivity)
        self.membranes: tuple[str, ...] = tuple(membranes)


    def visualize(self, resolve_regions: bool = True) -> None:
        """Visualize the geometry description. The compartments are
        represented as nodes and the membranes as edges.
        Args:
            resolve_regions: Whether to resolve regions or not. If True, the
                regions are used as nodes. If False, the compartments are used
                as nodes.
        """
        # Create a graph with compartments as nodes and membranes as edges
        graph = nx.MultiGraph()
        if resolve_regions:
            graph.add_nodes_from(self.regions)
            for left, right, name in self._membrane_connectivity:
                graph.add_edge(left, right, name=name)
        else:
            graph.add_nodes_from(self.compartments)
            for left, right, name in self._membrane_connectivity:
                graph.add_edge(
                    _extract_compartment(left),
                    _extract_compartment(right),
                    name=name,
                )

        # Draw the graph
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos=pos, node_color='white')
        nx.draw_networkx_labels(graph, pos=pos)

        # Draw lables for (potentially multiple!) edges
        multiplicity = max(count for _, _, count in graph.edges(keys=True)) + 1
        connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.15] * multiplicity)]
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            connectionstyle=connectionstyle,
            edge_color="lightgrey"
        )
        labels = {
            tuple(edge): attrs['name']
            for *edge, attrs in graph.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            pos=pos,
            connectionstyle=connectionstyle,
            edge_labels=labels,
            bbox={"alpha": 0},
        )
        plt.show()


def _extract_compartment(region: str) -> str:
    """Extract the compartment name from a region name.
    Args:
        region: The region name.
    Returns:
        The compartment name.
    """
    return region.split(':')[0] if ':' in region else region
