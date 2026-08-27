set shell := ["bash", "-uc"]
set positional-arguments
set dotenv-load

python:='python3'

install:
    cd fla-latest && {{python}} -m pip install -e .
    {{python}} -m pip install pandas pytest iniconfig docstring-parser
    {{python}} -m pip install -U huggingface_hub datasets
    {{python}} -m pip install --upgrade transformers
    # just install_flame
install_flame:
    cd flame && {{python}} -m pip install .
    {{python}} -m pip uninstall torchtitan -y
    {{python}} -m pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
    {{python}} -m pip install tyro
    {{python}} -m pip install -U datasets
lab:
    {{python}} run_experiment.py -c lab
