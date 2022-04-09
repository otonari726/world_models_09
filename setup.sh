#!/bin/bash

pip install -r requirements.txt

cd torchsearchsorted
python setup.py
pip install .

cd ..