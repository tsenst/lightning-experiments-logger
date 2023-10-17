#!/bin/bash
cd ..
cp experiments_addon/logger.py experiments_addon/logger.py.backup
sed -i '/@_sagemaker_run/d' experiments_addon/logger.py
