from setuptools import setup, find_packages

setup(
    name="mblab_interface",
    version='0.1',
    description='Generation of realistically looking humanoids using MB-Lab from Python.',
    packages=find_packages('src'),  # , exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    package_dir={'': 'src'},
    author='',
    author_email='',
    url='',
    #scripts=['scripts/generate_humanoid.py'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
    },
    # List our dependencies
    install_requires=['toml', 'wave', 'psutil'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
