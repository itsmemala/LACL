# Look-Ahead Continual Learning

This repository provides the code for implementing look-ahead (LA) continual learning and baseline continual learning methods for text classification.

Parts of the code are derived from the following repositories:  
1. https://github.com/ZixuanKe/PyContinual
2. https://github.com/hyscn/AdaBOP
3. https://github.com/SunWenJu123/rp2f
4. https://github.com/mohmdelsayed/upgd

## Abstract
Achieving continual learning (CL) with deep neural networks requires balancing stability and plasticity while enabling knowledge transfer. In this work, we focus on offline learning algorithms under the constraints: (I) no access to training data from prior tasks (II) no access to task-id at inference time. We introduce a novel measure, the relative parameter-importance, which measures the relative importance of each parameter with respect to both the current and past tasks. Parameters with high relative importance are interpreted as more important for maintaining past-task stability and thus heavily regularised, whereas parameters with low relative-importance are allowed to be more freely updated. Unlike existing methods, our approach allows the update of parameters with high past-task importance when they have low relative-importance, thus enabling backward knowledge transfer in addition to tackling the stability-plasticity trade-off. We demonstrate improvements against state-of-the-art CL methods on both class-incremental and domain-incremental learning text classification problems.


## Table of Contents
1. [Installation](#installation)
2. [Workflow](#workflow)
<!-- 3. [How to Cite](#how-to-cite) -->


## Installation
Run the following commands to set things up.
```
git clone https://github.com/itsmemala/LACL.git
cd LACL
conda create -n lacl python==3.10
pip install requirements.txt
```


## Workflow

To run LA experiments (for the Intent classification dataset, for example) using default hyper-parameters, run the following command. Change the hyper-parameter values as required through the command line or by updating the file.
```
bash scripts\\intent_sh_la_mas_chsf.sh random0 0 0 0.04854989 1641.28483697 1.0 1.0 0.8 True 0.1
bash scripts\\intent_sh_la_mas_chsf.sh random3 3 0 4.49536009 77.30662811 1.0 1.0 0.8 True 0.1
bash scripts\\intent_sh_la_mas_chsf.sh random6 6 0 28.24295365 246.34804902 1.0 1.0 0.8 True 0.1
```

<!-- ## How to Cite

```

``` -->
