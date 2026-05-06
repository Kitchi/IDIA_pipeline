"""Abstract base class for facility (cluster) configuration."""

from abc import ABC, abstractmethod


class FacilityConfig(ABC):
    """Encapsulates everything that differs between compute facilities.

    Subclass this and implement ``validate_account`` and ``validate_reservation``
    to add a new facility. Set the class-level attributes to match your cluster.
    """

    # ------------------------------------------------------------------ #
    # Resource limits — subclasses must set these                         #
    # ------------------------------------------------------------------ #
    total_nodes_limit: int
    cpus_per_node_limit: int
    mem_per_node_gb_limit: int
    mem_per_node_gb_limit_highmem: int

    # ------------------------------------------------------------------ #
    # Defaults — subclasses should set sensible values                    #
    # ------------------------------------------------------------------ #
    default_container: str
    default_mpi_wrapper: str
    default_account: str
    default_partition: str
    default_modules: list

    # ------------------------------------------------------------------ #
    # Validation hooks                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def validate_account(self, account, config, parser=None):
        """Validate that *account* is usable on this facility.

        Parameters
        ----------
        account : str or None
            The accounting group / project to use.
        config : str
            Path to the pipeline config file (for error messages).
        parser : argparse.ArgumentParser, optional
            If provided, errors are raised as parser errors.

        Returns
        -------
        str
            The resolved (possibly auto-detected) account string.
        """

    @abstractmethod
    def validate_reservation(self, reservation, args, config, parser=None):
        """Validate that *reservation* exists on this facility.

        Parameters
        ----------
        reservation : str
            The reservation name (empty string means no reservation).
        args : dict
            Full argument dictionary (for context).
        config : str
            Path to the pipeline config file.
        parser : argparse.ArgumentParser, optional
            If provided, errors are raised as parser errors.
        """

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        return f'<{self.__class__.__name__}>'
