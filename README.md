# Mini Places Challenge

Due Date: See the deadline for project in course announcements. 

You must work alone on this project.

In this project we will train a scene classifier for images.
The dataset we'll use is the MIT Mini Places dataset,
consisting of images from one of 100 scene categories.
It's a subset of the much larger MIT [Places2](http://places2.csail.mit.edu/) dataset.
The training set of the Mini Places dataset has 100,000 images;
the validation and test sets have 10,000 images each.


## Data

You can use the included `get_miniplaces.sh` script, or manually grab from
[here](http://dl.caffe.berkeleyvision.org/mit_mini_places/).

## Example

This repo provides an [example](https://github.com/pulkitag/CS280_SP18_Fallback_Project/blob/master/train_places_net.py)
of using [PyTorch](http://pytorch.org/) to train and evaluate a convnet --
[a variant of AlexNet](https://github.com/pulkitag/CS280_SP18_Fallback_Project/blob/master/models.py) -- for this task.
Assuming you have run `./get_miniplaces.sh` to download and setup the data, and have
Pytorch installed properly, you should be able to run it by simply doing

```
python train_places_net.py
```

Note: this code is only tested to work properly under python 3.6 with PyTorch 3.1. PyTorch 3.1 documentation is available [here](http://pytorch.org/docs/0.3.1/). Note that the documentation on PyTorchwebsite links to the docs for PyTorch github master branch by default. Running this code with PyTorch compiled from github master branch will raise several warnings.

A log of the output you should get from running this is available
(see [`log/miniplaces.train_log.txt`](https://github.com/pulkitag/CS280_SP18_Fallback_Project/blob/master/log/miniplaces.train_log.txt)).
The full command used to train and log to the provided file is as follows:

```
python -u train_places_net.py --gpus 0 --val_epoch 5 > log/miniplaces.train_log.txt 2>&1 &
```

The `-u` flag tells Python not to buffer the output so you can watch training progress,
e.g. by doing `tail -f log/miniplaces.train_log.txt`.
You may specify one or more GPU devices for training using the `--gpus` flag, e.g., `--gpus 0 1`.

At the end of training, the script will measure the model's performance on both the training and validation sets,
as well as producing a file containing your test set predictions for submission to the evaluation server
(if you're satisfied with your validation set performance).

The model takes about 38 minutes to train on a Titan Xp GPU with CUDA 9 and cuDNN 7,
and achieves 34.48% top 1 accuracy and 63.57% top 5 accuracy on the validation set.
See training and evaluation log at `./log/miniplaces.train_log.txt`.

## Your Task

You should consider the result from the provided example as a baseline
and try to come up with a model that does significantly better on this task.
(It's definitely possible! See
[MIT's leaderboard](http://miniplaces.csail.mit.edu/leaderboard-team.php)
for some motivation -- the winning team achieved 86.07% top 5 accuracy on the test set.)

You may **not** use outside data in your submission -- including using pretrained models on, e.g., ImageNet.
See **Competition rules** below for details.

Besides that rule, you're free to try whatever techniques you've heard of in and outside of class,
and of course free to think up new ones.
We're excited to see what you come up with!

You're encouraged to use PyTorch as the provided example is done in PyTorch (check the official tutorial [here](http://pytorch.org/),
but you're free to use any software you like, and furthermore you're
free to design your model and learning schemes however you like.
(Don't buy this whole "deep learning" craze? Hand-engineer away!)

If you are interested in more things, feel free to analyze the one or more of the following (you will get extra credits!):

- Download the CIFAR-10 dataset, train a model from scratch. Compare the results with a model pre-trained on the mini-places dataset.

- Instead of learning features on an image classification task, *self-supervised* learning is a mechanism for learning features by performing a pretext task that doesnot involves labels. For e.g., you can simply rotate images by [0, -90, 90, 180] and predict the rotation. You can now compare the performance of such a network against a model pre-trained on mini-places on the task of classifying CIFAR-10 images. Can you come up with other such pre-text tasks? 

- In the mini-places training set, make the training labels noisy (i.e. randomly change x% labels). How do the values of x =10, 20, 50 affect the performance on testing set? You might also want to evaluate transfer performance on CIFAR-10. Can you come up with ways to make the training robust to noise? 

- Getting annotations is expensive. Is it possible to achieve the same performance as using all labeled data, but by only using a subset of labelled examples? One idea is to first train a model with x% of the data. We will use this model to select for which images we require labels. One heuristic to chose such images is where the confidence of the model is closer to 0.5. There are many other heuristics one can chose. If you are interested in exploring this further, contact the course instructors.  


## Competition rules

*Any* use of data outside the provided Mini Places dataset is **not allowed** for competition submissions.
This includes the use of weights from models trained on outside datasets such as ImageNet or the full MIT Places(2) dataset.
(Transfer learning by way of training on larger datasets is a great idea in general, but not allowed here.)

On the other hand, using outside data *is* permissible for further analysis and results reported in your write-up
(as well as for any extensions you might do for your final project).
However, if you do this, please be very careful to avoid submitting any results to the evaluation server
from models leveraging outside data.

Also, don't annotate (or have others annotate) the test set, or manually adjust your model's predictions after looking at test set images.
Your submission should be the direct output of a fully automatic classification system.

Besides these rules, you're mostly free to do whatever you want!

## Evaluation server and competition leaderboard

You will create a text (CSV) file specifying your model's
top 5 predicted classes (in order of confidence) for each image in the test set,
and submit this file to an evaluation server for scoring as a deliverable of this project.

A sample is provided in this repo ([`sample_submission.csv`](https://github.com/pulkitag/CS280_SP18_Fallback_Project/blob/master/sample_submission.csv)).
If you're using the provided PyTorch example (`train_places_net.py`),
the test set predictions are formatted in this manner for you
and dumped to a file `top_5_predictions.test.csv` after training.

You will be limited to a small number of submissions to the evaluation server
for the duration of the competition, to avoid allowing overfitting to the test set.
You should primarily use the validation set to measure how your method is performing.
Scores will be ranked and posted to a leaderboard visible to the class -- healthy competition is encouraged!

The evaluation server is hosted in [Kaggle](https://www.kaggle.com/c/berkeley-cs280-backup-project). You should be able to make submissions once you link your account to a berkeley.edu email address. Note that due to limitations of Kaggle's built-in evaluation system, the leaderboard is ranked based on the top-1 (instead of top-5) accuracy.

## Deliverables

Your should submit to the evaluation server at least one set of test set predictions
from a model that you implemented and trained.

Your grade will depend on how you perform 

In your report include, 

  - Your team's name on the evaluation server.
  - Details on how you designed and trained your best model(s),
    and other things you tried that didn't work as well
  - A couple bits of experimental and/or analytical analysis, e.g.,
      - Ablation studies: how does varying this parameter,
        removing this layer, etc., affect performance?
      - What categories are easiest/hardest for the model? Why?
      - Visualizing and understanding what the model learns,
        e.g., through activation mining, feature visualization techniques,
        t-SNE, ...
      - Does using outside data in some way help?
        (Just be careful not to mix outside data into your competition submissions!)
      - Anything else you can come up with!

## AWS EC2 Cloud GPUs

If you don't have access to your own CUDA-capable GPU and would like to train models using a GPU,
you can use GPU instances via AWS's EC2 service.
[AWS Educate](https://aws.amazon.com/education/awseducate/)
provides $100 worth of free credits per year to Berkeley students,
which is enough for roughly a week's worth of GPU instance use (g2.2xlarge/p2.xlarge instance type).
You should be careful to turn off your GPU instance(s)
when not in use for training to get the most out of your credits and maximize GPU time.
Consider working on designing your model elsewhere,
e.g., on a GPU-less AWS instance or on your own machine.

See [this document](https://github.com/pulkitag/CS280_SP18_Fallback_Project/blob/master/NVIDIA_AWS%20Educate%20Student%20Onboard.pdf)
for AWS Educate signup instructions. Initially, your instance limit for g2.2xlarge and p2.xlarge instance will be 0 (login to AWS dashboard, go to EC2 tab, then you can find limits tab on the left side).
You need to click ``requst instance limit`` to increase your GPU instance to 1. 

Make sure you always choose to contact AWS by phone, which is the fast way.

When creating an instance, make sure you choose an environment that with CuDNN and pytorch pre-installed. Then you need to clone the github repo to get started and run the example immediately by doing
```
$ cd CS280MiniPlaces
$ python train_places_net.py --cudnn --iters 100 # just train for 100 iterations to make sure everything works; remove '--iters 100' to do the full training
```

(The first time you run training on a fresh AWS instance, the startup time may be very slow likely due to disks spinning up.
Sometimes it helps to run the above `python` command once, wait a bit for training to start, then Ctrl+C to interrupt and rerun the command.)

## Acknowledgements

This project is "heavily inspired" by the [MIT assignment](http://6.869.csail.mit.edu/fa15/project.html).
Thanks to all of the instructors for creating this challenge dataset,
with special thanks to Bolei Zhou and Prof. Aude Oliva for all their help getting us set up!
