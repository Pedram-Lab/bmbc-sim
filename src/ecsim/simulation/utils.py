from math import ceil
from typing import Dict

try:
    # Check if we are running in a Jupyter notebook
    get_ipython
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


class SimulationClock:
    """Simple simulation clock that keeps track of the current time and advances
    it by a fixed time step size. The clock also keeps track of events that
    occur in fixed time intervals. Either a fixed number of time steps or a
    fixed end time can be provided to determine when the simulation should stop.
    """
    def __init__(
            self,
            *,
            end_time: float,
            time_step: float = None,
            n_steps: int = None,
            events: Dict[str, int] = None,
            verbose: bool = False
    ) -> None:
        self.end_time = end_time
        self.current_time = 0.0
        self.events = events or {}
        self.counter = 0

        if time_step is None and n_steps is None or time_step is not None and n_steps is not None:
            raise ValueError("Exactly one of {time_step, n_steps} must be provided.")

        if time_step is not None:
            self.time_step = time_step
            self.n_steps = ceil(end_time / time_step)
        else:
            self.n_steps = n_steps
            self.time_step = end_time / n_steps

        self._progress = tqdm(total=self.n_steps, desc="Simulation time") if verbose else None

    def advance(self) -> None:
        """Advance the simulation clock by one time step."""
        self.current_time += self.time_step
        self.counter += 1
        if self._progress:
            self._progress.update(1)

    def is_running(self) -> bool:
        """Check if the simulation is still running."""
        return self.current_time < self.end_time

    def event_status(self, event_name: str) -> bool:
        """Return the current status of a given event.
        :param event_name: The name of the event to check.
        :return: True if the event is active, False otherwise.
        :raises KeyError: If the event does not exist.
        """
        return self.counter % self.events[event_name] == 0

if __name__ == "__main__":
    clock = SimulationClock(n_steps=10, end_time=10.0, events={"foo": 5}, verbose=True)
    while clock.is_running():
        clock.advance()
        print(f"{clock.counter}: {clock.current_time} foo={clock.event_status('foo')}")
