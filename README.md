# Fader-Networks-Tensorflow

It is the tensorflow implementaion of Fader Networks described in the [paper](https://arxiv.org/pdf/1706.00409.pdf) by Lample et al. In this paper, they have tried to disentangle the information of a face and changing the image by varying different attributes of the image.

> This property could allow for applications where users can modify an image using sliding knobs, like faders on a mixing console, to change the facial expression of a portrait, or to update the color of some objects

For this they have looked at the CelebA dataset, which is easily avaiable at their site.

In the model, they have designed an Encoder such that given an input and image X it outputs the embedding of an image, which is independent of the features of the image X. So suppose if there are two images X and X' (with atrribute list Y and Y') which differ in only one feature lets say, in one image, say in X, eyes of the subject are open, while in other eyes are closed. Given this Enc(X,Y) = Enc(X',Y'). Definitely, this is the ideal case, but this is the main idea behind the fader networks.

Picture below, depicts the architecture in the best way.

<p align="center"> 
  <img src="./images/model.PNG">
</p>

<p align="center"> 
  <i> Image from original <a href="https://arxiv.org/pdf/1706.00409.pdf"> paper </a>   </i>
</p>


It looks like a GAN-like architecture where Encoder tries to output an embedding X_emb , such that discriminator cannot guess what actual attributes of the images are, and at the same time, we train the Discriminator such that it tries to guess the attributes of image X even from the embedding X_emb. So, it acts like a two player game where performance of each will complement the performance of other in training. In the end, we will get a good encoder that can be used to create embedding from image X which is not dependent on its features. 
After this, we will have the decoder, which we will simply use to get the image back, given embedding and the new attributes for the output image. For training, we will feed the same attribute Y and will try to get the original image back.

In whole model is an auto encoder with a mix of adverserial network.

Training:

Before training one needs to download the celeba dataset from the <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"> link </a>. One also have to download the list of attributes as well.

To start training the network, run the following command:

``` python main.py ```

You can also specify the dataset directory using ```--dataset``` argument to the above command. The output will be created in <b>output<b> folder.