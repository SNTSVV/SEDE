# MB-Lab interface

`mblab_interface` is a python package allowing to automatically create realistically looking human models based on 
the [Blender](https://www.blender.org/) addon [MB-Lab](https://mb-lab-community.github.io/MB-Lab.github.io/).

***

## Installation

A requirement for the library to properly work is that `MB-Lab` is installed as an addon in `Blender`.

`mblab_interface` can then be installed as 

```sh
> /blender-prefix/2.8x/python/bin/python3.7m -m pip install -e . [--user]
```

Notice that this performs a developer install (`-e`), i.e. python loads the package directly from the repository 
without copying the files to the default locations. Consequently, changes in the source code are directly available 
without having to reinstall. If you only wish to use the code as is (without changing it), then the flag `-e` can be 
omitted.

The `--user` might have to be used if not enough writing permissions are available for user account.

***

## Usage

A key component of the library is a collection of assets (clothes, hair & beard, glasses, etc.) that can be added 
to the character to create a diverse collection of people. The most up-to-date collection is stored on the data server
`LUBISISCSI01` under `\\lubisiscsi01\db_blender\mblab_asset_data`.

### Random full character generation

The script [scripts/generate_humanoid.py](scripts/generate_humanoid.py) creates a human character with random body measures, phenotype, gender 
and clothes.

It can be called according to (e.g. on g02)

```sh
> blender -b --python scripts/generate_humanoid.py -- /data/db_blender/mblab_asset_data/ -l debug -o output_dir --render --studio lights. 
```

This instructs `mblab_interface` to look for the assets in the directory `/data/db_blender/mblab_asset_data/`, use a `debug` log-level,
store the output to the directory *output_dir*, render the result and add studio lights for nice lightning.

A camera setting that is more suited for rendering the face and therefore does not include pants and shoes is given in the script 
[scripts/generate_head.py](scripts/generate_head.py). Its command-line inteface is similar to that described above. To see all options, you can invoke it as follows

```
> blender -b --python scripts/generate_head.py -- -h
```

Finally, the script `generate_heads.py` takes one integer command line argument that indicates how many times to call the script `generate_head.py`
in order to generate several models. Notice that dependending on which platform you call it, you might have to adjust the location of the 
data or output directories.

### API

A standard way to call the API is given by the example [examples/simple_random_example.py](examples/simple_random_example.py) which can be called by specifying both the paths to 
the data directory (containing the assets) and the output directory (notice that both must exist).

```sh
> blender -b --python scripts/simple_random_example.py -- /path/to/data/directory /path/to/output/directory
```