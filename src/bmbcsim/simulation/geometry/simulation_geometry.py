from itertools import accumulate

import networkx as nx
import matplotlib.pyplot as plt
import ngsolve as ngs

from bmbcsim.logging import logger
from bmbcsim.simulation.geometry.compartment import Compartment, Region
from bmbcsim.simulation.geometry.membrane import Membrane

class SimulationGeometry:
    """A high-level description of the simulation geometry. The description is
    based on the following concepts:
    - :class:`Region`: A part of the simulation geometry that is resolved and
        labeled in the mesh.
    - :class:`Compartment`: A collection of regions that share the same
        bio-chemical agents. A compartment can be made up of one or more
        regions.
    - :class:`Membrane`: A mesh boundary that separates two compartments.
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

        # Names can refer to multiple mesh_regions, so check if they're unique
        mesh_region_to_name = {}
        for name in set(mesh.GetMaterials()):
            mesh_region = mesh.Materials(name)
            if len(mesh_region.Split()) != 1:
                raise ValueError(f'Region name "{name}" is not unique.')
            mesh_region_to_name[mesh_region] = name

        # Sort regions into compartments
        compartment_to_regions: dict[str, list[Region]] = {}
        for mesh_region, name in mesh_region_to_name.items():
            compartment_name, region_name = _split_compartment_and_region(name)
            logger.debug('Inferred compartment %s from region %s', compartment_name, name)

            regions = compartment_to_regions.get(compartment_name, [])
            regions.append(Region(name=region_name, volume=0.0))
            compartment_to_regions[compartment_name] = regions

        self.compartments = {name: Compartment(name, mesh, regions)
                             for name, regions in compartment_to_regions.items()}
        logger.info('Found %d compartments:', len(self.compartments))
        for compartment in self.compartments.values():
            logger.info(" - '%s' with regions %s", compartment.name, compartment.regions)

        # Process interfaces
        boundaries = {
            bnd: name
            for name in set(mesh.GetBoundaries())
            for bnd in mesh.Boundaries(name).Split()
        }
        self._full_interface_info = []
        for bnd, name in boundaries.items():
            # Record detailed information about the interfaces
            neighbors = bnd.Neighbours(ngs.VOL).Split()

            left = mesh_region_to_name[neighbors[0]]
            right = mesh_region_to_name[neighbors[1]] if len(neighbors) > 1 else None

            self._full_interface_info.append((left, right, name))
            logger.debug('Connected regions %s and %s via membrane %s', left, right, name)

        membranes_to_neighbors = {}
        for left, right, name in self._full_interface_info:
            # Find the interfaces that connect compartments (= membranes)
            left, _ = _split_compartment_and_region(left)
            right, _ = _split_compartment_and_region(right)

            if left == right:
                # A membrane connects different compartments
                continue

            neighbors = membranes_to_neighbors.get(name, set())
            neighbors.add((self.compartments[left], self.compartments[right] if right else None))
            membranes_to_neighbors[name] = neighbors

        self.membranes = {name: Membrane(name, mesh, neighbors, 0.0)
                          for name, neighbors in membranes_to_neighbors.items()}

        logger.info('Found %d membranes: %s', len(self.membranes), self.membranes)
        self.update_measures()

    def update_measures(self) -> None:
        """Update compartment volumes and membrane areas on the current mesh."""
        # Update region volumes
        for compartment in self.compartments.values():
            region_names = compartment.get_region_names(full_names=True)
            for region, name in zip(compartment.regions, region_names):
                region._volume_parameter.Set(
                    ngs.Integrate(1, self.mesh.Materials(name), ngs.VOL)
                )

        # Update membrane areas
        for membrane in self.membranes.values():
            membrane._area_parameter.Set(
                ngs.Integrate(1, self.mesh.Boundaries(membrane.name), ngs.BND)
            )


    @property
    def compartment_names(self) -> list[str]:
        """The names of all compartments in the geometry.
        """
        return list(self.compartments.keys())


    @property
    def membrane_names(self) -> list[str]:
        """The names of all membranes in the geometry.
        """
        return list(self.membranes.keys())

    @property
    def region_names(self) -> list[str]:
        """The full names of all regions in the geometry. This includes the
        compartment name as a prefix, e.g. 'compartment:region'.
        """
        return [name for compartment in self.compartments.values()
                for name in compartment.get_region_names(full_names=True)]


    def visualize(self, resolve_regions: bool = True) -> None:
        """Visualize the geometry description. The compartments are
        represented as nodes and the membranes as edges.
        
        :param resolve_regions: Whether to resolve regions or not. If True, the
            regions are used as nodes. If False, the compartments are used as
            nodes.
        """
        # Create a graph with compartments as nodes and membranes as edges
        graph = nx.MultiGraph()
        graph.add_node('<outside>')
        if resolve_regions:
            graph.add_nodes_from(self.region_names)
            for left, right, name, _ in self._full_interface_info:
                # Add edges between regions for each membrane
                graph.add_edge(left, right if right else '<outside>', name=name)
        else:
            graph.add_nodes_from(self.compartment_names)
            for membrane in self.membranes.values():
                for left, right in membrane.connects:
                    name = membrane.name
                    right_name = right.name if right else '<outside>'
                    # Add edges between compartments for each membrane
                    # Note: this may create multiple edges if a membrane connects
                    # the same compartments in both directions
                    graph.add_edge(left.name, right_name, name=name)

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


def _split_compartment_and_region(domain: str) -> tuple[str, str]:
    """Split a compartment:region string into its components. If the region is
    the sole region within a compartment, return the full name for both.
    
    :param domain: The domain name.
    :returns: A tuple with the compartment and region names.
    """
    if domain and ':' in domain:
        return domain.split(':')
    else:
        return domain, domain
