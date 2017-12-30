# Contest4 - StackGAN++
* ## StackGAN++ implemented by Tensorflow
    * CS565600 Deep Learning Contest4
    * For inclass contest
* ## Installation Guide
    * `pip install -r requirement.txt`
    * `CUDA 8.0`,  `cudnn 6.0` and `Tensorflow 1.4`
    * Download and unzip [dataset](https://drive.google.com/file/d/19qZUGEWotm_YSW5pa1d5w0A01yckaqM3/view?usp=sharing) besides main.py
    * Download and unzip [testing tool](https://drive.google.com/file/d/1Av9L29Dm11ajlS4gj4Ym7eoOyDGu2jnL/view?usp=sharing) besides main.py
    * Install and setup config [kaggle cli](https://github.com/floydwch/kaggle-cli)
    * `mkdir inference` besides main.py
* ## Training
    * `python main.py`
    * If you occur memory error when training, you could resize the batch_size in `main.py` line 19
* ## Evaluate and Submit to Kaggle
    * `cd testing && python inception_score.py ../inference ../score.csv && kg submit ../score.csv`
    * If you have 2 GPUs, you can uncomment `main.py` line 806 to evaluate in loop. At the same time you should specified the `testing/inception_score.py` to run on different GPU with enough memory space.
* ## Reference
```
@article{Han17stackgan2,
  author    = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
  title     = {StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks},
  journal   = {arXiv: 1710.10916},
  year      = {2017},
}
```

```
@inproceedings{han2017stackgan,
Author = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
Title = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks},
Year = {2017},
booktitle = {{ICCV}},
}
```
