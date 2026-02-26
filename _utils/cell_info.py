from gdsfactory import Component


def copy_info(dst: Component, src: Component, prefix: str | None = None) -> None:
    """Copy `src.info` entries into `dst.info` with a prefix.

    By default, the prefix is derived from `src.function_name` when available.
    """
    if not hasattr(dst, "info") or not hasattr(src, "info"):
        raise ValueError("Both dst and src must expose an 'info' attribute.")

    src_prefix = prefix or getattr(src, "function_name", None) or "cell"
    src_data = (
        src.info.model_dump() if hasattr(src.info, "model_dump") else dict(src.info)
    )

    dst.info.update({f"{src_prefix}_{key}": value for key, value in src_data.items()})
