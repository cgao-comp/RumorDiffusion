from typing import Any, Dict


def read_yaml(path: str) -> Dict[str, Any]:
    """Reads a YAML file into a dictionary.

    Args:
        path: Path to YAML file.

    Returns:
        Loaded dictionary.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


