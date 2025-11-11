import numpy as np


def _estimate_chunk_size(shape, dtype, target_mb=10.0):
    """Estimate balanced chunk sizes aiming ~target_mb per chunk."""
    itemsize = np.dtype(dtype).itemsize or 1
    if not shape:
        return tuple()
    target_items = max(1, int((target_mb * 1024 * 1024) / itemsize))
    total_items = int(np.prod(shape))

    # Small arrays → use full shape
    if total_items <= target_items:
        return tuple(int(s) for s in shape)

    # 1D arrays
    if len(shape) == 1:
        return (min(int(shape[0]), target_items),)

    # Balanced per-dimension chunk target with a small floor
    per_dim = max(32, int(round(target_items ** (1.0 / len(shape)))))
    return tuple(int(min(int(dim), per_dim)) for dim in shape)


def _sanitize_chunks_from_dask(dask_chunks, shape):
    """
    Convert dask's chunks (tuple of tuples) → per-dim tuple.
    Takes the first chunk per dim; falls back to full dim if None/invalid.
    Ensures ≥1 and ≤ dim length.
    """
    try:
        out = []
        for chks, dim in zip(dask_chunks, shape):
            if (chks is None) or (len(chks) == 0) or (chks[0] is None):
                c0 = int(dim)
            else:
                c0 = int(chks[0])
            c0 = max(1, min(int(dim), c0))
            out.append(c0)
        return tuple(out)
    except Exception:
        return None


def _make_var_chunks(var, target_chunk_mb):
    """Return a valid chunks tuple for a single xarray Variable/DataArray."""
    # Scalars: return None to indicate "do not set chunks"
    if getattr(var, "ndim", 0) == 0:
        return None

    # Try Dask
    data = var.data
    chunks = None
    if hasattr(data, "chunks") and data.chunks is not None:
        chunks = _sanitize_chunks_from_dask(data.chunks, var.shape)

    # Otherwise estimate
    if chunks is None:
        chunks = _estimate_chunk_size(var.shape, var.dtype, target_mb=target_chunk_mb)

    # Final sanity
    try:
        chunks = tuple(int(max(1, c)) for c in chunks)
    except Exception:
        return None
    return chunks


def generate_chunk_encoding_for_datatree(datatree, target_chunk_mb=10.0, include_coords=True):
    """
    Build encoding dict for DataTree.to_zarr() that ONLY sets 'chunks' and
    ensures no None is passed for non-scalar arrays. Works for both data_vars and coords.
    """
    encoding = {}

    # Iterate nodes
    try:
        nodes = datatree.subtree.items()
    except AttributeError:
        nodes = [(node.path, node) for node in datatree.subtree]

    for node_path, node in nodes:
        ds = getattr(node, "ds", None)
        if ds is None:
            continue

        group_encoding = {}

        # Data variables
        for name, var in ds.data_vars.items():
            chunks = _make_var_chunks(var, target_chunk_mb)
            if chunks is not None:
                group_encoding[name] = {"chunks": chunks}
            else:
                group_encoding[name] = {}  # scalar → leave unset

        # Coordinates (important: avoid chunks=None here)
        if include_coords:
            for name, var in ds.coords.items():
                chunks = _make_var_chunks(var, target_chunk_mb)
                if chunks is not None:
                    group_encoding[name] = {"chunks": chunks}
                else:
                    group_encoding[name] = {}  # scalar coord

        if group_encoding:
            encoding[node_path] = group_encoding

    return encoding
