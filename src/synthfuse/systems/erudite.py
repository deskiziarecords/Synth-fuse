# src/synthfuse/systems/erudite.py
import jax.numpy as jnp
from synthfuse.tools.foundation.math_utils import zeta_transform

class SystemErudite:
    """
    The Librarian of the Synth-Fuse Manifold.
    Manages Native Data Ingestion and Spectral Memory.
    """
    def ingest_directory(self, path="./ingest/raw/"):
        # 1. Byte-stream inhalation
        # 2. Apply Zeta Transform to project into Frequency Space
        # 3. Store as a 'Lazy Tensor' (Generation function, not raw data)
        for data_patch in self._scan(path):
            spectral_map = zeta_transform(data_patch)
            self._register_in_vault(spectral_map)

    def retrieve_context(self, query_vector):
        """
        Uses Manifold Pruning to find the data most relevant to 
        the current 'Spell' without scanning the whole database.
        """
        return self.vault.query(query_vector, method="manifold_projection")
