import simulacra as si  # somewhere there's ordering in these imports
import ionization  # will get nasty import error if not this first

from pathlib import Path
import random

import gzip
import pickle


import htmap


import click
from tqdm import tqdm
from halo import Halo
from spinners import Spinners


SPINNERS = list(name for name in Spinners.__members__ if "dots" in name)


def make_spinner(*args, **kwargs):
    return Halo(*args, spinner=random.choice(SPINNERS), stream=sys.stderr, **kwargs)


CLI_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CLI_CONTEXT_SETTINGS)
@click.argument("tag")
@click.option("--outdir", default=None)
def main(tag, outdir):
    with make_spinner(f"loading map {tag}...") as spinner:
        map = htmap.load(tag)
        spinner.succeed(f"loaded map {tag}")

    if outdir is None:
        outdir = Path.cwd()
    outdir = Path(outdir)
    outpath = outdir / f"{tag}.sims"

    try:
        with si.utils.BlockTimer() as timer:
            with gzip.open(outpath, mode="wb") as f:
                pickle.dump(len(map), f)
                for sim in tqdm(map, desc="pickling sims...", total=len(map)):
                    pickle.dump(sim, f)
        print(f"pickled sims from {tag} (took {timer.wall_time_elapsed})")
    except:
        if outpath.exists():
            outpath.unlink()


if __name__ == "__main__":
    main()
