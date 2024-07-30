model_(date)_(accuracy*1000)_(epoch time in seconds * 10)

# model_07.29.20.39.39_a892: 
16 Filters, MaxPool(2), 16 Filters, MaxPool(2), 4096->512->4
Dropout 0.3 each conv layer and first full layer
~2M Params

# model_07.29.21.09.48_a916:
Testing Bigger Conv Layer, want to see the GPU light up
128 Filters x 128 Filters x 32,768 fully connected layer input neurons
16,930,948 Parameters
Only made CPU go up
Miniscule increase in accuracy
Cuda wasn't available - same error from before "oserror: [winerror 126] the specified module could not be found"
Needed to add Desktop Development with C++ in Visual Studio Installer

# model_07.30.00.44.08_a858:
Set test set size to 200 (from 100, 800 total)
Need to randomly select test set
12.5s per epoch

# model_07.30.00.52.27_a847:
266,196 parameters:
Maxpool(4), 1024->256->4
11.5s per epoch
Maybe a bigger fully connected layer is needed?
What's the slow part? 1 second gain?!

# model_07.30.10.51.35_a877:
Removed Dropout Layers, no noticable difference
Add an epoch timer

# model_07.30.11.56.44_a842_t83
Randomly select images for eval & test set
Eval_loader was using the test set, now it's actually an evaluation, not much difference though?

# model_07.30.12.36.48_a861_t1898
Remove MaxPool(2) layer, fc: 16384->4096->4
67,132,116 Parameters
3 minute epoch!