# Underwater enhancement based on a self-learning strategy and attention mechanism for high-intensity regions

### Authors:
Claudio D. Mello Jr., Bryan U. Moreira, Paulo J. O. Evald, Paulo L. Drews Jr. and Silvia S. Botelho.

### Abstract:

Images acquired during underwater activities suffer from environmental properties of the water, such as turbidity and light attenuation. These phenomena cause color distortion, blurring, and contrast reduction. In addition, irregular ambient light distribution causes color channel unbalance and regions with high-intensity pixels. Recent works related to underwater image enhancement, and based on deep learning approaches, tackle the lack of paired datasets generating synthetic ground-truth. In this paper, we present a self-supervised learning methodology for underwater image enhancement based on deep learning that requires no paired datasets. The proposed method estimates the degradation present in underwater images. Besides, an autoencoder reconstructs this image, and its output image is degraded using the estimated degradation information. Therefore, the strategy replaces the output image with the degraded version in the loss function during the training phase. This procedure \textit{misleads} the neural network that learns to compensate the additional degradation. As a result, the reconstructed image is an enhanced version of the input image. Also, the algorithm presents an attention module to reduce high-intensity areas generated in enhanced images by color channel unbalances and outlier regions. Furthermore, the proposed methodology requires no ground-truth. Besides, only real underwater images were used to train the neural network, and the results indicate the effectiveness of the method in terms of color preservation, color cast reduction, and contrast improvement.

### Description of the software:

The model was implemented in Keras/Tensorflow. The code contains the autoencoder and the degradation function used
to generate the enhancement of the underwater images, as described in the paper. The method does not use paired dataset.
The dataset is loaded by the model as a single file, containing the training and test images. The format of the input image
are RGB 256x256x3 and scaled to [0., 1.0].

### Package requirements:
Versions of the softwares used in the implementation:

- Python = 3.7.6
- Keras = 2.3.1
- Tensorflow = 2.20
- Python IDE Spyder = 4.0.1 (Anaconda=1.10.0)
- O.S. Ubuntu 18.04.5 LTS

