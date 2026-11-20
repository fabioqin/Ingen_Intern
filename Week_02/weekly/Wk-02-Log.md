# Week 02 log

I started with the tensor exercises and the small PyTorch model. I checked the shape of the input, the weights and the output of each layer. The bias gradient made more sense once I saw why it had to be summed across the batch. I checked the gradients against PyTorch and ran the model on MNIST. After one epoch on 10,000 training images, it got 88.42% accuracy on the test set.

For the second experiment I used UNSW-NB15 network flows. I removed the attack category because it would give away the answer. I also removed duplicate flows before taking the samples. I trained a majority-class baseline, logistic regression and a small MLP. I kept the same train, validation and test rows for each run. This made it easier to see what changed when I tried a different seed or learning rate.

The part I found most useful was counting false alarms. At the default threshold, my seed-42 MLP found 3,418 attacks, but it also flagged 1,960 normal flows. If I only looked at recall, I would have missed that. When I read the ROC curve at a 1% false-positive rate, the model caught about 56% of attacks. Logistic regression caught about 42% at the same point.

The loss plot helped me pick a checkpoint. For seed 42, validation loss was lowest at epoch 29. Running five more epochs did not improve it. I saved the loss values so I can compare future runs with this one.

Next time I would pick a threshold on validation data first and try it on another traffic capture. I also want to group the missed attacks by category. That would tell me which examples to inspect first and whether the overall recall hides a particular weakness.

[MNIST result](../evidence/mnist.json) · [Intrusion report](../W02_Intrusion_Classifier_Report.md)
