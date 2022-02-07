# Intel&reg; OpenFL ML Extension

This project aims to develop an extension of the Intel&reg; OpenFL framework to allow the federated use of traditional
Machine Learning models, such as Decision Trees.

In order to use this extension, the original OpenFL framework should be installed on your computer. Then simply replace
the import of the original OpenFL classes with the one provided in the unito folder and substitute the OpenFL
"protocols" folder with the one provided by this repo. A simple example script named Federated_RF is available.

In order to serialize traditional ML models, this extension exploits the Dill framework.

Up to now only SciKit Learn models have been tested within this experimental software.
