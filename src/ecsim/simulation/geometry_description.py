from itertools import accumulate
import logging

import networkx as nx
import matplotlib.pyplot as plt
import ngsolve as ngs


NO_DOMAIN = 'exterior'
logger = logging.getLogger(__name__)


class GeometryDescription:
    """TODO
    """
    def __init__(self, mesh: ngs.Mesh) -> None:
        """Create a new geometry description. Regions and interfaces are
        inferred from the mesh. Regions can either be compartments or part of a
        multi-region compartment, which is indicated by naming it
        'compartment:region'. Membranes are mesh boundaries that separate two
        compartments.

        :param mesh: The mesh to create the geometry description from.
        """
        logger.info('Creating geometry description from mesh.')
        self.mesh = mesh

        # Add all other compartments as nodes (with special node for 'outside of
        # computational domain')
        regions = {
            region: name
            for name in set(mesh.GetMaterials())
            for region in mesh.Materials(name).Split()
        }
        self.regions: list[str] = list(set(regions.values()))
        logger.info('Found %d regions: %s', len(self.regions), self.regions)

        # Cluster regions into compartments
        self._compartments: dict[str, str] = {}
        for region in self.regions:
            compartment_name = _extract_compartment(region)
            compartment_regions = self._compartments.get(compartment_name, [])
            compartment_regions.append(region)
            self._compartments[compartment_name] = compartment_regions
            logger.debug('Inferred compartment %s from region %s', compartment_name, region)
        logger.info('Found %d compartments: %s',
                    len(self._compartments), list(self._compartments.keys()))

        # Store all boundaries as (left, right, name)
        boundaries = {
            bnd: name
            for name in set(mesh.GetBoundaries())
            for bnd in mesh.Boundaries(name).Split()
        }
        self._full_membrane_connectivity = set()
        for bnd, name in boundaries.items():
            neighbors = bnd.Neighbours(ngs.VOL).Split()

            n1 = regions[neighbors[0]]
            n2 = regions[neighbors[1]] if len(neighbors) > 1 else NO_DOMAIN

            self._full_membrane_connectivity.add((n1, n2, name))
            logger.debug('Connected %s to %s via membrane %s', n1, n2, name)

        # Store membrane names
        self._membranes = {}
        for left, right, name in self._full_membrane_connectivity:
            left_compartment = _extract_compartment(left)
            right_compartment = _extract_compartment(right)

            if left_compartment == right_compartment:
                # A membrane connects different compartments
                continue

            membrane_neighbors = self._membranes.get(name, (set(), set()))
            membrane_neighbors[0].add(left_compartment)
            membrane_neighbors[1].add(right_compartment)
            self._membranes[name] = membrane_neighbors
        logger.info('Found %d membranes: %s', len(self._membranes), list(self._membranes.keys()))


    @property
    def compartments(self) -> list[str]:
        """The list of compartments.
        """
        return list(self._compartments.keys())


    @property
    def membranes(self) -> list[str]:
        """The list of membranes.
        """
        return list(self._membranes.keys())


    def get_regions(self, compartment: str, *, full_names=False) -> list[str]:
        """Get all regions of a compartment.

        :param compartment: The compartment to get the regions of.
        :param full_names: Whether to return the full names of the regions or
            the names without the compartments.
        :returns regions: The regions of the given compartment.
        """
        return self._compartments[compartment] if full_names \
            else [_strip_compartment(region)
                  for region in self._compartments[compartment]]


    def get_membrane_neighbors_left(self, membrane: str) -> set[str]:
        """Get all compartments on the left side of a membrane.
        
        :param membrane: The membrane to get the neighbors of.
        :returns neighbors: The compartments on the left side of the membrane.
        """
        return self._membranes[membrane][0]


    def get_membrane_neighbors_right(self, membrane: str) -> set[str]:
        """Get all compartments on the right side of a membrane.
        
        :param membrane: The membrane to get the neighbors of.
        :returns neighbors: The compartments on the right side of the membrane.
        """
        return self._membranes[membrane][1]


    def get_membrane_neighbors(self, membrane: str) -> set[str]:
        """Get all compartments on both sides of a membrane.
        
        :param membrane: The membrane to get the neighbors of.
        :returns neighbors: The compartments on both sides of the membrane.
        """
        return self.get_membrane_neighbors_left(membrane) \
            | self.get_membrane_neighbors_right(membrane)


    def visualize(self, resolve_regions: bool = True) -> None:
        """Visualize the geometry description. The compartments are
        represented as nodes and the membranes as edges.
        
        :param resolve_regions: Whether to resolve regions or not. If True, the
            regions are used as nodes. If False, the compartments are used as
            nodes.
        """
        # Create a graph with compartments as nodes and membranes as edges
        graph = nx.MultiGraph()
        graph.add_node(NO_DOMAIN)
        if resolve_regions:
            graph.add_nodes_from(self.regions)
            for left, right, name in self._full_membrane_connectivity:
                graph.add_edge(left, right, name=name)
        else:
            graph.add_nodes_from(self.compartments)
            for left, right, name in self._full_membrane_connectivity:
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
    
    :param region: The region name.
    :returns: The compartment name.
    """
    return region.split(':')[0] if ':' in region else region


def _strip_compartment(region: str) -> str:
    """Strip the compartment name from a region name. If the region is the sole
    region within a compartment, return the full name.
    
    :param region: The region name.
    :returns: The region name without the compartment.
    """
    return region.split(':')[1] if ':' in region else region


def full_name(compartment: str, region: str) -> str:
    """Get the full name of a region within a compartment.
    
    :param compartment: The compartment name.
    :param region: The region name.
    :returns: The full name of the region.
    """
    return f'{compartment}:{region}'
