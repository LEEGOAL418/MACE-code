#!/bin/bash 

mamba create -n mace python=3.9 -y
mamba activate mace
pip install --upgrade pip
pip install mace-torch
