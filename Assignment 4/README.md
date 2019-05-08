# Assignment 4

### Part 3 Images
<img src="Graph.png" align="center" width=700>
<img src="Scalars.png" align="center" width=700>


### Part 5: Brief Write Up

  - Logistic regression with 100 epochs:
  ```sh
  Epoch: 0100 cost = 0.019103495
Validation Error: 0.018999993801116943
Optimization Finished!
Test Accuracy: 0.9785
```
  - Original (two hidden layers) multilayer perceptron with 1000 epochs:
  ```sh
  ```
- Original (two hidden layers) multilayer perceptron with 100 epochs:
```sh
Epoch: 0100 cost = 0.282158803
Validation Error: 0.0745999813079834
Optimization Finished!
Test Accuracy: 0.9223
```
- Modified (one hidden layer) multilayer perceptron with 100 epochs
 ```sh
 Epoch: 0100 cost = 0.524606975
Validation Error: 0.20920002460479736
Optimization Finished!
Test Accuracy: 0.7862
  ```
  
- Did running many more (1,000 vs 100) epochs yield better or worse results for the original multilayer perceptron?

- Did the multilayer pereceptron do better or worse than logistic regression when you ran them both for 100 epochs?
- 
- Did decreasing the number of hidden layers reduce the success of the multilayer perceptron?
Yes! when I used 2 layers the test accuracy was around 92%, 1 layer reduced the accuracy to 78% - **14% less** for the same amount of epochs.
- What general lesson might you deduce from your answers to these three questions?
