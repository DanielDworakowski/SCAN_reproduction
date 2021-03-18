## The SCAN paper

Recreation of "SCAN: Learning to Classify Images without Labels" (https://arxiv.org/pdf/2005.12320.pdf) using PyTorch Lightning. The evaluation code and random augmentation code was borrowed from the original repository (https://github.com/wvangansbeke/Unsupervised-Classification).

In contrast to the original paper, cluster centers for nearest neighbor finding are performed using the average of multiple randomly augmented version of the same image. This is done to improve the estimate of the feature vector for the class instance.

Achieves approximately 83% accuracy on the CIFAR-10 dataset.
