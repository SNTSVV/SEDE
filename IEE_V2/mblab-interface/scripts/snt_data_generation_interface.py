import os
import subprocess as sp
import sys
import argparse
import pathlib as pl


def setup_cli(args):
    parser = argparse.ArgumentParser()

    default_out = os.path.join(pl.Path(__file__).resolve().parent.parent.parent,'snt_simulator','data','mblab')
    parser.add_argument('-n', '--number', type=int, default=1, help='The number of characters to generate')
    parser.add_argument('-s', '--start-index', type=int, default=1, help='The starting index for the labelling')
    parser.add_argument('-o', '--output', type=str, default=default_out,
                        help="The output directory containg the results for the different runs")
    parser.add_argument('-g', '--gpu', type=str, default="0", help='Which gpus to use')
    parser.add_argument('--random-engine', type=str, default='Noticeable', choices=['Light', 'Realistic', 'Noticeable', 'Caricature', 'Extreme'],
                        help='Decide how extreme the variations in the randomly generated characters can be.')

    return parser.parse_args(args)


if __name__ == "__main__":

    args = setup_cli(sys.argv[1:])

    script = os.path.join(pl.Path(__file__).resolve().parent, "snt_face_dataset_generation.py")
    data_path = os.path.join(pl.Path(__file__).resolve().parent.parent.parent,'mblab_asset_data')
    print(script)
    for i in range(args.number):
        index = i + args.start_index
        cmd = ["blender", "-b", "--python", script, "--", str(data_path), "-l", "debug", "-o",
               f"{args.output}/run_{index}", "--render", "--gpu", f"{args.gpu}", "--studio-lights",
               "--random-engine", args.random_engine]
        sp.call(cmd, env=os.environ, shell=True)
