# dp-merf


### Dependencies
Versions numbers are based on our system and may not need to be exact matches. 

    python 3.6
    torch 1.3.1              
    torchvision 0.4.2
    numpy 1.16.4
    scipy 1.3.1
    pandas 1.0.1
    scikit-learn 0.21.2
    matplotlib 3.1.0 (plotting)
    seaborn 0.10.0 (more plotting)
    sdgym 0.1.0 (handling tabular datasets)
    autodp 0.1 (privacy analysis)
    backpack-for-pytorch 1.0.1 (efficient DP-SGD for DP-MERF+AE)
    tensorboardX 1.7 (some logging)
    tensorflow-gpu 1.14.0 (DP-CGAN)


## Repository Structure

### Tabular data

`code_tab/single_generator_priv_all.py` contains the code for the tabular experiments


### Balanced Datasets
`code_balanced` contains code for MNIST and Gaussian data experiments. See `code_balanced/README.md` for instructions on how to run the experiments.



### comparison models
- `dpcgan` contains code from <https://github.com/reihaneh-torkzadehmahani/DP-CGAN> with changes to data-loading 
- `dpgan` contains our implementation of a naive DP-GAN as described in the paper
- `gs-wgan` contains code from <https://github.com/DingfanChen/GS-WGAN> with edits to data-loading





