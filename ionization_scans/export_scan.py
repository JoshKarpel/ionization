import simulacra as si  # somewhere there's ordering in these imports
import ionization  # will get nasty import error if not this first

from pathlib import Path

import gzip
import pickle

from tqdm import tqdm
import click

import htmap

from . import mesh_scan_utils


@click.command(context_settings=mesh_scan_utils.CLI_CONTEXT_SETTINGS)
@click.argument("tag")
@click.option("--outdir", default=None)
def main(tag, outdir):
    with mesh_scan_utils.make_spinner(f"loading map {tag}...") as spinner:
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
