__all__ = ["PATH"]

import pathlib

cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
module = pathlib.Path(__file__).parent.absolute()
repo = module.parent


class Path:
    module = module
    repo = repo
    gds = module / "gds"
    klayout = module / "klayout"

    lyp = klayout / "tech" / "lnoi400.lyp"
    lyt = klayout / "tech" / "tech.lyt"
    lyp_yaml = module / "layers.yaml"
    tech = module / "klayout" / "tech"


PATH = Path()
