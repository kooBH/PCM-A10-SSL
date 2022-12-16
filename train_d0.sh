#!/bin/bash

python src/train.py -c config/${VERSION}.yaml -v ${VERSION} -d cuda:0 --chkpt <TODO:root>/chkpt/${VERSION}/bestmodel.pt -s 0