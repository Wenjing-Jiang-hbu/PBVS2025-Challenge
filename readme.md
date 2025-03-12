# CDFF-Net: Cross-Domain Feature Fusion Network for SAR Image  Classification

## Installation

-Python 3.10

-PyTorch 1.12.1

-CUDA 11.3

-cuDNN 8.2.1

Building Gabor Conv

```bash
cd ultralytics/nn/modules
python setup.py build install 
```

## Running

Train the model:

```bash
python mytrain.py
```

Evaluate the model for output class_id:

```bash
python eval-class.py
```

Evaluate the model for output score:

```bash
python eval-score.py
```

## References

We would like to acknowledge the following works and their authors for inspiring and supporting our research:

@article{Khanam2024YOLOv11AO,
  title={YOLOv11: An Overview of the Key Architectural Enhancements},
  author={Rahima Khanam and Muhammad Hussain},
  journal={ArXiv},
  year={2024},
  volume={abs/2410.17725},
  url={https://api.semanticscholar.org/CorpusID:273532028}
}

@ARTICLE{8357578,
  author={Luan, Shangzhen and Chen, Chen and Zhang, Baochang and Han, Jungong and Liu, Jianzhuang},
  journal={IEEE Transactions on Image Processing}, 
  title={Gabor Convolutional Networks}, 
  year={2018},
  volume={27},
  number={9},
  pages={4357-4366},
}

@INPROCEEDINGS{9880272,
  author={Ristea, Nicolae-Cătălin and Madan, Neelu and Ionescu, Radu Tudor and Nasrollahi, Kamal and Khan, Fahad Shahbaz and Moeslund, Thomas B. and Shah, Mubarak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
  title={Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection}, 
  year={2022},
  volume={},
  number={},
  pages={13566-13576},
}

@INPROCEEDINGS{10208541,
  author={Low, Spencer and Nina, Oliver and Sappa, Angel D. and Blasch, Erik and Inkawhich, Nathan},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={Multi-modal Aerial View Object Classification Challenge Results - PBVS 2023}, 
  year={2023},
  pages={412-421},
}
