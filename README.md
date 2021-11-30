# Group25-Sustainable-AI
"Sustainable AI for Wireless Spectrum Sensing" project 

1. In order to run these files, you will need the RML2016.10b dataset from www.deepsig.ai/datasets . Once this is installed you will need to extract it into a .dat file and import that file into the 'data' folder.

If you would like to use PCA for datapreprocessing:
  2. Next, you will need the L2 PCA data. Run the 'l2_pca_dim_reduction_v0.py' file. The data will now be in the 'data' folder under 'l2'. You are now ready to train the model.

3. 
  - If you would like to train a resnet model without dimensionality reduction, run the 'resnet.py' file. 
  - If you would like to train a resnet model with L2 PCA dimensionality reduction, run the 'resnetl2pca.py' file.

4. Great! You have succesfully trained the model. Your results/plots will now be in the 'results' folder and your saved tensorflow model will be in the 'models' folder.
