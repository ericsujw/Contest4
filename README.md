# Contest4 - StackGAN++
* ## StackGAN++ implemented by Tensorflow
    * CS565600 Deep Learning Contest4
    * For inclass contest
* ## Framework
<img src="framework.jpg" width="900px" height="350px"></img>
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
    * StackGAN++ PyTorch [paper](https://arxiv.org/abs/1710.10916) [code](https://github.com/hanzhanggit/StackGAN-v2)
