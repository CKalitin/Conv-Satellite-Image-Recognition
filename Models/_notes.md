model_(date)_(accuracy*1000)

# model_07.29.20.39.39_a892: 
16 Filters, MaxPool(2), 16 Filters, MaxPool(2), 4096->512->4
Dropout 0.3 each conv layer and first full layer
~2M Params

# model_07.29.21.09.48_a916
Testing Bigger Conv Layer, want to see the GPU light up
128 Filters x 128 Filters x 32,768 fully connected layer input neurons
16,930,948 Parameters
Only made CPU go up
Miniscule increase in accuracy
Cuda wasn't available - same error "oserror: [winerror 126] the specified module could not be found" from before
Needed to add Desktop Development with C++ in Visual Studio Installer