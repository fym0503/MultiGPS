# MultiGPS: Multi-task Gene Panel Selection for Imaging-based Spatial Profiling
MultiGPS is a multi-task deep learning framework for informativetarget panel selection. It combines stochastic gates with multi-task optimization, enabling end-to-end differentiable gene panel selection with multiple objectives covering comprehensive aspects regarding ST datasets.

# Installation
The requirements of MultiGPS can be installed by:  
`pip install -r requirements.txt`  
then use `pip install -e .` or directly import the code.

# Tutorials
MultiGPS provides some tutorials about the selection and evaluation. It can be used on various kinds of data including scRNA and ST dataset.  
[preprocess_ALM.ipynb]() and [select_eval_ALM.ipynb]() provides how to preprocess, select and evaluate the scRNA dataset.  
[preprocess_MERFISH.ipynb]() and [select_eval_MERFISH.ipynb]() provides how to preprocess, select and evaluate the MERFISH dataset.  
[imputation_analysis.ipynb]() and [SVG_analysis.ipynb]() are two downstream analysis of ST data.