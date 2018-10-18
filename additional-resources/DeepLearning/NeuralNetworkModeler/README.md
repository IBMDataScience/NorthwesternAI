<img src="images/IBM.png">

# Deep Learning in IBM Watson Studio

Building a deep learning model can be complicated. However, Watson Studio has a great feature that allows data scientists build deep learning models simply by using UI. In this demo, I'll show how to build a deep learning model in Watson Studio and run the model by leveraging the cloud computing power of IBM step by step.



### Dataset
We use fashin_mnist in our demo, which is a dataset of Zalando's article images. More details can be found in https://github.com/zalandoresearch/fashion-mnist

<img src="images/p1.png">

There three data files can be downloaded in the "data.zip". Download the zip file to local device and unzip it

The dataset includes a training set, a testing set and a validation set, which is required by the neural network modeler in Watson Studio. All sets have labeled images.

<img src="images/p2.png">



### Get Started
Watson Studio provides a suite of tools for data scientists, application developers and subject matter experts to collaboratively and easily work with data and use that data to build, train and deploy models at scale. (offical)

In my demo, as I mentioned at beginning, I can use Watson Studio to build a deep learning model without writing a single line of codes. So let's get started to see how could this happen!

First, get a free IBM cloud account at bluemix.net and then log in. The console looks like this:

<img src="images/p3.png">

The reason why we should start from an IBM cloud account is that the Watson Studio is a product based on the IBM cloud. Deep learning models have requirement in computing power, and that's the advantage of using cloud computing. One thing we must notice is that we can get basic cloud computing service for free (no kidding!) on the IBM cloud! And even the free plan has strong computing power using GPU!

Now let's click Menu at top left and select Watson

<img src="images/p4.png">

Scroll down a little bit and click "Try Watson Studio"

<img src="images/p5.png">

Click "Continue"

<img src="images/p6.png">

Click "Get Started"

<img src="images/p7.png">

The page jumps to Watson Studio Dashboard

<img src="images/p8.png">




### Create new project
On the dashboard, click "New Project", choose “Complete” and click "OK"

<img src="images/p9.png">

Type in a project name and click “Add” on left (Select storage service)

<img src="images/p10.png">

Scroll down, choose Lite (free) and click "Create"

<img src="images/p11.png">

Click "Confirm".

The storage service is very common nowadays. We need this cloud storage to store our data, our model and literally everything relates to our project in Watson Studio. (It's free! 25G!)

<img src="images/p12.png">

After confirming, click “Refresh” in the “Define Storage” panel and then click “Create”

<img src="images/p13.png">

The page jumps to project dashboard

<img src="images/p14.png">



### Add machine learning service to project
Click on "Setting" tab, scroll down to "Associated Service", click "Add service" and select "Machine Learning". (We need to associate instances of services with projects in order to get the most out of them in Watson Studio)

<img src="images/p15.png">

Choose the Lite(free) version and confirm

<img src="images/p16.png">



### Upload dataset
Before we upload the datasets, we want to create two buckets in our cloud object storage, one is for storing the training data and one is for storing the results of deep learning model. Obviously training data are the three data files we prepared for the model. As for the results, they are logs, model files, etc mention that we write the logs to these files as the models are training and so on. Go to the IBM cloud console (https://console.bluemix.net/dashboard/apps). Click the storage we just created

<img src="images/p17.png">

In Buckets, click “Create bucket”

<img src="images/p18.png">

Type in the name “image-demo” for storing training data. Repeat the process and create “results-demo” for storing results.

<img src="images/p19.png">

Click the image-demo bucket

<img src="images/p20.png">

Click “Add objects” and upload the prepared fashion_mnist dataset

<img src="images/p21.png">

Now we have datasets on cloud storage

<img src="images/p22.png">



### Build deep learning model
Finally we get to build our model! Go back to Watson Studio project page, click Assets tab. Scroll down to Modeler Flows and add new flow

<img src="images/p23.png">

Type the flow name and select flow type as “Neural Network Modeler” (NNM), click "Create".

NNM is a graphic interface that you can add deep learning layers, manage the data flows and tune the parameters. Like I said before, there is not a single line of codes I should write. I will get my deep learning model done by using NNM.

<img src="images/p24.png">

(You can definitely try the "From example" tab to explore more about the model builder. IBM integrates many examples to get you familiar with this feature)

<img src="images/p1a.png">

Now we can start building our model by clicking Palette at top left side and drag the layers from it

<img src="images/p25.png">

Assume we build a model like following, it is a nice way of visualizing how the data flows from one layer to another following the arrow

<img src="images/p26.png">

Double click "Image Data" and click "Create a connection"

<img src="images/p27.png">

Select corresponding dataset and expand Settings and change the parameters

<img src="images/p28.png"><img src="images/p29.png">

Double click Conv 2D layer and change parameters, repeat for all Conv 2D layers

Double click Pool 2D layer and change parameters, repeat for all Pool 2D layers

Double click Dense layer and change parameters

<img src="images/p30.png"><img src="images/p31.png"><img src="images/p32.png">

Keep others default and "Publish training definition"

<img src="images/p33.png">

Click "Publish", this will save the training definition in your Watson Machine Learning repository. You can check it in your IBM cloud

<img src="images/p34.png">



### Run the model
Now that we have model built, go back to project dashboard and click Assets tab, scroll down to Experiments and add new experiment, type in the experiment name and select Cloud Object Storage bucket

<img src="images/p35.png">

Select “Connection to project COS”. Select the buckets we created. (You can also create new buckets if you like. However, please make sure your data is in your new bucket)

<img src="images/p36.png">

Click "Add training definition" 

<img src="images/p37.png">

Choose "Existing training definition" and select our model. Click "Select"

<img src="images/p38.png">

Click "Create and run"

<img src="images/p39.png">

Wait for the task to be completed

<img src="images/p40.png">

Completed! click on the run to see more details

<img src="images/p41.png">

Look at the logs, it's a nice model!

<img src="images/p42.png">

Now we can save and distribute our model in Python code. We finally can see some codes :)

Go back to Assets and click the model we just built in "Modeler flows". Click the download button and choose the model you want. In my demo, I chose "Download as Keras model". 

<img src="images/p43.png">

Open the downloaded model. Watson Studio writes all those complicate codes and processes for us!

<img src="images/p44.png">


