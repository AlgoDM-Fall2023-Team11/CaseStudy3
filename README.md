# CaseStudy3
 The Visual Search application combines various deep learning and image processing techniques to help users find images based on different criteria: artistic style, product type, and visual similarity using VAEs. 

* Codelabs Link:

  https://codelabs-preview.appspot.com/?file_id=1ldFdnOQLou_sz7LPEpDj4i-I8gnRSN4umOKqVvHFy-0#1
  
* Video Description:

  https://drive.google.com/file/d/1pveAxUnd4p2YceAedtoe-WEAgHx8fOky/view?usp=sharing
  
* Code Spaces Functionality Demo(Part2):

  https://drive.google.com/file/d/1bdGWsjARChgFZEwIrJwOBy-xOWtWrHJB/view?usp=sharing
 
## Visual Search leveraging Tensorflow, OpenCV and Streamlit

### Members
<a href="https://www.linkedin.com/in/krishna-barfiwala/" target="_blank">Krishna Barfiwala  002771997</a> | 
<a href="https://www.linkedin.com/in/nakulshiledar/" target="_blank">Nakul Shiledar 002738981</a> | 
<a href="https://www.linkedin.com/in/usashisurajitroy/" target="_blank">Usashi Roy 002752872</a>



## TensorHouse Data - Search
The TensorHouse Data Science Project is dedicated to the exploration and implementation of models, techniques, and datasets that have their origins in collaborative efforts between industry practitioners and academic researchers, often in partnership with prominent companies across diverse sectors, including technology, retail, manufacturing, and more. The primary focus of TensorHouse is the utilization of industry-proven methods and models rather than purely theoretical research. Notable examples of these practical applications are the Visual Search by Artistic Style using VGG16, the Visual Search with Variational Autoencoders, and the Visual Search based on Product Type utilizing EfficientNetB0. These notebooks represent practical, data-driven approaches that leverage the expertise of both academia and industry to solve real-world problems, making TensorHouse a hub for cutting-edge, industry-relevant data science solutions.

# Visual Search by Artistic Style using VGG16


### Data Source

The dataset used in this project comprises a collection of 32 images of artworks created by various artists. These images are stored within the "images-by-style" directory, which is part of the "tensor-house-data" repository. To access the dataset and the images, you can follow this link: Tensor-House Data Repository - Images by Style.


### Description

The notebook presented here showcases an image search method based on artistic style. The application of this technique is particularly relevant in scenarios like eCommerce websites, where users seek to find artwork similar to a sample they provide. The system not only takes into account artistic style but can also incorporate other similarity metrics, such as image subject or category, which can be extracted using deep neural networks.
This implementation is based on a TensorFlow tutorial for Neural Style Transfer, which, in turn, draws inspiration from the seminal paper "A Neural Algorithm of Artistic Style" by Gatys et al.


### Procedure

* The notebook loads and resizes the images for analysis.
* Style embeddings, representing the artistic style of each image, are computed using TensorFlow. Gram matrices of style layers are employed for this purpose.
* To visualize style embeddings in 2D space, t-SNE is applied, providing a visual representation of image styles.
* The notebook demonstrates how to find images with styles similar to a reference image by calculating cosine distances between style embeddings and returning the most similar images.


### Conclusion 

The notebook showcases practical style-based image search, valuable for eCommerce and other applications. Style embeddings enable efficient retrieval of images with similar styles, and t-SNE visualization enhances the understanding of style embeddings in a lower-dimensional space.


### Streamlit Application
Link: https://algodm-fall2023-team11-casestudy3-style-transfer-nakul-xihfzx.streamlit.app/ 
The application allows you to upload an image of any artist using the upload field and then shows you all other images which belong to the same artist using VGG16.



### Use

Online retailers can utilize style-based image search to allow customers to find products with similar visual styles. For example, if a customer likes a particular fashion item, they can search for similar styles within the inventory. This feature simplifies the shopping process and can lead to increased sales.





# Visual Search on Product utilizing EfficientNetB0




### Data Source

The use of the Clothing Dataset in your project opens up several possibilities for data-driven applications and analysis. This dataset, consisting of over 5,000 images spanning 20 different classes.To access the dataset, you can refer to the following link: Clothing Dataset on GitHub.


### Description

In this notebook, the author has developed visual search models for retrieving similar objects based on visual similarity or similarity in the space of object attributes. The notebook uses the Clothing Dataset 



### Procedure

* Load the Clothing Dataset and preprocess the data.
* Fine-tune a pre-trained EfficientNetB0 model for image classification.
* Visualize image class probabilities for test images.
* Compute image embeddings and project them using t-SNE for visualization.
* Summarize the model architecture.


### Conclusion

This notebook demonstrates the development of a visual search system for retrieving visually similar objects, particularly focused on clothing items. By fine-tuning a deep learning model, it achieves good accuracy in classifying clothing items. The t-SNE visualization of the embedding space enables users to explore how images are distributed and clustered in this space, which is valuable for visual search applications.


### Streamlit Application
Link: https://casestudy3team11part2.streamlit.app/ 
The application enables image upload and utilizes EfficientNetB0 for classifying products into clothing types and their attribute associations.



### Use

The application's image classification and attribute association capability can be used in marketing for tasks such as automated product categorization, personalized product recommendations, and targeted advertising based on a user's clothing preferences and attributes. This can enhance the user experience and optimize marketing strategies in e-commerce and fashion-related industries.




# Visual Search with Variational Autoencoders


### Data Source

The Fashion-MNIST dataset, available at https://www.kaggle.com/datasets/zalando-research/fashionmnist, comprises 60,000 training examples and 10,000 test examples, each representing a grayscale image of size 28x28 pixels. These images are associated with labels categorizing them into one of 10 distinct fashion classes.

### Description

This notebook demonstrates the development of a prototype for unsupervised embedding learning using a Variational Autoencoder (VAE). The VAE is trained on the Fashion MNIST dataset and is capable of learning a smooth, regular manifold. The learned embeddings can be used for tasks like nearest neighbor search.



### Procedure

* Load the Fashion MNIST dataset, which consists of grayscale images of size 28x28 pixels categorized into 10 classes.
* Define and train a Variational Autoencoder (VAE) model on the dataset. The VAE model consists of an encoder and a decoder. The encoder maps the input images into a latent space, while the * decoder reconstructs images from the latent space.
* Train the VAE model using the defined encoder and decoder for a specified number of epochs.
* Create a grid over the learned semantic space and decode these points into images to visualize the reconstructed images.
* Query nearest neighbors in the latent space based on a chosen image and visualize the results.



### Conclusion

The VAE model successfully learns embeddings that allow for the reconstruction of images and the retrieval of nearest neighbors in the latent space. This capability can be used for tasks like visual search and similarity-based recommendations.


### Streamlit Application
The application enables image upload and utilizes Variational Autoencoders for classifying products into their categories and suggests all other items that fall into that category.



### Use

The learned embeddings from the VAE model can be used for visual search applications. Given an input image, you can encode it into the latent space and search for visually similar images in the dataset by comparing their embeddings. This is valuable for e-commerce platforms to provide visually similar product recommendations. The VAE can also be utilized for anomaly detection. Images that deviate significantly from the learned manifold in the latent space may be considered anomalies. This is helpful in quality control and fraud detection.
