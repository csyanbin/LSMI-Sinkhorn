## LSMI-Sinkhorn: Semi-supervised Mutual Information Estimation with Optimal Transport
This is the implementation of our ECML/PKDD21 paper "LSMI-Sinkhorn: Semi-supervised Mutual Information Estimation with Optimal Transport" by Liu Y., Yamada M., Tsai YH., Le T., Salakhutdinov R., Yang Y.

### Requirements

    pytorch
    numpy
    PIL

### SMI Estimation on Synthetic Data

    python run_synthetic_exps.py

The results are in "synthetic_result" folder

### Image Summarization Experiments

First, download the images from http://users.sussex.ac.uk/~nq28/kernelized_sorting.html, unzip and put the "images" folder under this codebase.
Then, run:

    python main_layout_ECML-PKDD.py

The result layout images are in "layout" folder.

### Bibtex

If you use this code or results for your research, please consider citing:
````
@inproceedings{liu2019lsmi,
  title={LSMI-Sinkhorn: Semi-supervised Mutual Information Estimation with Optimal Transport},
  author={Liu, Yanbin and Yamada, Makoto and Tsai, Yao-Hung Hubert and Le, Tam and Salakhutdinov, Ruslan and Yang, Yi},
  booktitle={ECML/PKDD},
  year={2021}
}
````