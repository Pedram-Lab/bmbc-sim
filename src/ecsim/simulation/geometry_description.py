import abc
import ngsolve as ngs


class Region(abc.ABC):
    """A region within a geometry representing a biological structure. Chemical
    solutions are continuous across region boundaries which are not
    :class:`Membrane`s.
    """
    def __init__(self, name: str) -> None:
        """Create a new region.
        Args:
            name: The name of the region.
        """
        self.name = name

    @abc.abstractmethod
    def diffusion_coefficient(self) -> float:
        """Get the diffusion coefficient of the region in simulation units.
        Returns:
            The diffusion coefficient of the region.
        """


class Compartment(abc.ABC):
    """A compartment within a geometry representing a biological structure.
    Chemical solutions are discontinuous across compartment boundaries. Those
    boundaries have to be modeled explicitly as :class:`Membrane`s).
    """
    def __init__(self, name: str) -> None:
        """Create a new compartment.
        Args:
            name: The name of the compartment.
        """
        self.name = name

    @abc.abstractmethod
    def get_regions(self) -> list[Region]:
        """Get the regions within the compartment.
        Returns:
            The regions within the compartment.
        """

    @abc.abstractmethod
    def add_diffusion_to_form(
        self,
        form: ngs.BilinearForm,
        test_fun: ngs.comp.ProxyFunction,
        trial_fun: ngs.comp.ProxyFunction
    ) -> None:
        """Add the ion kinetics within the compartment to the given bilinear
        form.
        Args:
            form: The form to describe diffusion within the compartment (changed
                in place).
            test_fun: A test function for the compartment.
            trial_fun: A trial function for the compartment
        """


class Membrane(abc.ABC):
    """A membrane within a geometry representing a biological structure. A
    membrane separates two :class:`Compartment`s. Transport across the membrane
    is modeled explicitly.
    """
    def __init__(self, name: str) -> None:
        """Create a new membrane.
        Args:
            name: The name of the membrane.
        """
        self.name = name

    @abc.abstractmethod
    def add_transport_to_form(
        self,
        form: ngs.BilinearForm,
        test_fun_left: ngs.comp.ProxyFunction,
        trial_fun_left: ngs.comp.ProxyFunction,
        test_fun_right: ngs.comp.ProxyFunction,
        trial_fun_right: ngs.comp.ProxyFunction
    ) -> None:
        """Add the ion kinetics within the membrane to the given bilinear form.
        Args:
            form: The form to describe diffusion within the membrane.
            test_fun_left: A test function for the compartment left of the membrane.
            trial_fun_left: A trial function for the compartment left of the membrane.
            test_fun_right: A test function for the compartment right of the membrane.
            trial_fun_right: A trial function for the compartment right of the membrane.
        """

class GeometryDescription:
    """A bio-chemical description of the geometry of a simulation.
    """
    def __init__(
        self,
        compartments: list[Compartment],
        membranes: list[Membrane],
    ) -> None:
        """Create a new geometry description.
        Args:
            compartments: The compartments within the geometry.
            membranes: The membranes within the geometry.
        """
        self.compartments = compartments
        self.membranes = membranes


    def validate(self, mesh: ngs.Mesh) -> None:
        """Validate the geometry description.
        Args:
            mesh: The mesh to validate the geometry description against.
        Raises:
            ValueError: If the geometry description is invalid.
        """
        expected = GeometryDescription.describe_mesh(mesh)
        actual_regions = set(r.name for c in self.compartments for r in c.get_regions())
        actual_membranes = set(m.name for m in self.membranes)

        _check_if_double_or_missing(
            actual_regions, expected["domains"], "compartments"
        )
        _check_if_double_or_missing(
            actual_membranes, expected["interfaces"], "membranes"
        )


    @staticmethod
    def describe_mesh(mesh: ngs.Mesh) -> dict[str, list[str]]:
        """Return a list of all domains and interfaces in the mesh.
        Args:
            mesh: The mesh to describe.
        Returns:
            description: A dictionary with two keys: "domains" and "interfaces".
        """
        return {
            "domains": list(mesh.GetMaterials()),
            "interfaces": list(mesh.GetBoundaries())
        }


def _check_if_double_or_missing(
        entities_to_check: list[Compartment | Membrane],
        expected_names: list[str],
        entity_name: str
    ) -> None:
    double_names = list(name for name in entities_to_check if entities_to_check.count(name) > 1)
    if double_names:
        raise ValueError(f"Duplicate {entity_name}: {double_names}")
    missing_names = set(e.name for e in entities_to_check) - set(expected_names)
    if missing_names:
        raise ValueError(f"Missing {entity_name}: {missing_names}")
