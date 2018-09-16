# LinearRegression

Linear regression is a frequent used method in the field of machine learning, specifically for tasks that require a continous output. Hence, for a given datamatrix X (rows represent samples / columns represent features) we try to find a line that fits the given data reasonable well. Consider the following point distribution:

![alt text](https://github.com/NumericalMax/LinearRegression/blob/master/images/rawData.png)

We aim to find a hpyerplane (in a 1d context: line) that minimizes the distance from the points to the plane. We can find such an optimal plane using the in statistics well known ordinary least squares (OLS) method. We even can find an analytically closed solution. For more information on this you can have a look at the Wikipedia article about [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression). The visual solution to above point distribution is:

![alt text](https://github.com/NumericalMax/LinearRegression/blob/master/images/regressionLine.png)

In this repository we implement the analytical correct solution of OLS in C++ for a given datamatrix. Note that for large scaled data matrices a numerical solution (e.g. gradient descent or newton method) might be more feasable. Also in the case that your matrix is sparse there are more advanced and more suitable solution to do the computation.

# Structure of the Repository
- images: Contains png-files drawing the outcome of the algorithm
- generateData.py: Code for generating the datamatrix X and target y (here you can customize the distribution of data points)
- X.txt: Datamatrix X (here with one feature, i.e. 1d case)
- y.txt: Targetvector y
- main.cpp: C++ code for computing the parameters for the best fitting line
- output.txt Automatically generated txt file that contains the parameters for the best fitting line
- createImages.py: Takes the output.txt file as well as X.txt and y.txt and saves the point distribution as well as the best fitting line into one image (right now this works only for the 1d case and if the intercept is fitted)

# Exemplary execution in the terminal

```
max:LinearRegression Max$ g++ main.cpp -o main
max:LinearRegression Max$ python generateData.py 
max:LinearRegression Max$ ./main 
Number of rows: 100
Number of columns: 1
Fit intercept? 1
max:LinearRegression Max$ python createImages.py
```

# Machine
The Code was solely implemented and tested on a MacBook Pro i5 / 8GB RAM.

# Dependencies of the Code
- C++: iostream, iomanip, fstream, sstream
- Python: Numpy, Matplotlib

# Compiler
The g++ compiler with the c++11 standard is used.

# ToDo
The main code works also for the case of multiple features (multiple linear regression). We just have to adapt the loadData function such that it is capable of loading a comma seperated file into the datamatrix.

# Author
Max Kapsecker, 2018: Remark, that there is always room for optimization. The Code can indeed be optimized regarding time and space efficiency. Feel free to report any mistake to max.kapsecker@gmail.com
