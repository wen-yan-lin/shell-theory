Shell theory makes three predictions: 
1) There exist naturally occurring class boundaries that are defined in terms of the mean of variance of each class;
2) The boundaries can be discovered (and classification achieved) by simply estimating the class mean of shell-normalized instances;
3) Results can be enhanced with shell-normalization, which involves subtracting the mean of the test-data from each instance; then scaling the instance to be a unit-vector.

Shell-normalization differs from “classical normalization”  in its use of test-mean, rather than training-mean for normalization. This greatly affects the results in problems like one-class learning and anomaly detection. 

In this code, we demonstrate shell theory on both one-class svm and anomaly detection, with comparisons provided against one-class svm; comparisons with other algorithms can be found in the main-paper [1], with their code

Our experiments indicates that shell learning provides almost perfect results on such very difficult problems. 

Resnet50 features used are available at :
https://drive.google.com/file/d/1QZtdt5Nh_rXAtgP6CI0alOYwb6g8Z019/view?usp=sharing
and should be placed in the data folder.

Raw images are not strictly necessary for this code but you may want to play with them yourselves. They can be downloaded from:
https://drive.google.com/file/d/1bdBqmD9plsM2uLHAmon1n45mIZNCROrg/view?usp=sharing
and should be placed in the images folder.

Have fun!

By the way, if you found the code useful, could you please cite:

[1] Lin, Wen-Yan, et al. "Shell Theory: A Statistical Model of Reality." IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).

@ARTICLE{9444188,
  author={Lin, Wen-Yan and Liu, Siying and Ren, Changhao and Cheung, Ngai-Man and Li, Hongdong and Matsushita, Yasuyuki},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Shell Theory: A Statistical Model of Reality}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3084598}}

