# Deep Learning with Watson Studio
The _Deep Learning Service_ with Watson Machine Learning simplifies solving the most challenging and computationally expensive machine learning problems with clarity and ease. 

**Choose your path**
* [Neural Network Modeler](./NeuralNetworkModeler) provides a flexible and extensible graphical interface for building nerual networks and training them with GPUs in the Cloud
* [Notebooks](./NotebooksPath) offers intuitive and advances tools for training multiple models and managing 
* [Command Line](./CommandLineInterface) tools make it easy to script your deep learning experiments without leaving your terminal. 
This folder contains the Fashion MNIST datasets for the deep learning tutorials. 


## The Data
We're going to use the Fashion MNIST Data from Zolando Research. 
More info about this data [here](https://github.com/zalandoresearch/fashion-mnist/tree/master/data). You can also [download with Keras](https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles).
<br> 

Watson Studio [Documentation for Deep Learning and Cloud Object Storage](https://dataplatform.ibm.com/docs/content/analyze-data/ml_dlaas_object_store.html?audience=wdp&context=analytics). 

```

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. 
Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. 
It shares the same image size and structure of training and testing splits.

``` 


In [Original Data](./original_data), there is the `ubyte.gz` data in the same format as MNIST. 

In the [NNM](./NeuralNetworkModeler) peice, there is a file [data.zip](./NeuralNetworkModeler/data.zip) which contains the same data in `python pickle` format. 

__________

**Data in this folder:**

**Samples**
<img src="https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png" alt="embedding"> 


**Embedding**
<img src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/embedding.gif?raw=true" alt="embedding"> 
___________


**Additional Reading:**
* [Deep Learning Advances from IBM Research](https://www.ibm.com/blogs/research/2018/03/deep-learning-advances/)
* [Deep Learning as a Service, IBM makes advanced AI more accessible for users everywhere](https://www.ibm.com/blogs/watson/2018/03/deep-learning-service-ibm-makes-advanced-ai-accessible-users-everywhere/)


**Additional Resources:**

* [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio) offers collaborative ecosystem of powerful tools for data-driven teams
* [Watson Machine Learning](https://developer.ibm.com/clouddataservices/docs/ibm-watson-machine-learning/) provides intuitive and flexible model management, deployment, and scoring services alongside an [open source python package](http://wml-api-pyclient.mybluemix.net/) and [command line tools]()
* [Cloud Object Storage](https://console.bluemix.net/docs/services/cloud-object-storage/basics/order-storage.html#order-storage) encrypts and disperses data across multiple geographic locations, providing access over HTTP using a REST API. COS makes use of the distributed storage technologies provided by the IBM Cloud Object Storage System

