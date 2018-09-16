/* MIT License
 
 Copyright (c) 2018 Max Kapsecker
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace matrixOperation{
    
    void matrixProductTranspose(float **A, float **B, float **C, int l, int m){
        
        /*
         Description:   - Computed the product of A and the transpose of B (A*B^T)
         Runtime:       - O(l*m^2)
         Input:         - (float **A)   Reference on input matrix A
                        - (float **B)   Refrence on input matrix B
                        - (float **C)   Reference on output matrix C
                        - (int size_l)  row length of B
                        - (int size_m)  row and col length of A / col length of B
         Output:        - void / call by reference
         */
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < l; j++){
                C[i][j] = 0.0;
                for(int k = 0; k < m; k++){
                    C[i][j] += A[i][k] * B[j][k];
                }
            }
        }
    }
    
    void matrixVectorMultiplication(float **A, float *y, float *result, int l, int m){
        
        /*
         Description:   - Classical matrix-vector multiplication Ay
         Runtime:       - O(l*m)
         Input:         - (float **A)   Reference on input matrix A
                        - (float *y)   Refrence on input vector y
                        - (float *result)   Reference on output vector theta
                        - (int size_l)  row length of A
                        - (int size_m)  col length of A and length of vector y
         Output:        - void / call by reference
         */
        
        for(int i = 0; i < l; i++){
            result[i] = 0.0;
            for(int j = 0; j < m; j++){
                result[i] += A[i][j] * y[j];
            }
        }
    }
    
    void matrixProductWithTranspose(float **A, float **B, int l, int m){
        
        /*
         Description:   - Multiplies the transpose of matrix A with matrix A (A^T*A)
         Runtime:       - O(l*m*m)
         Input:         - (float **A)   Reference on input matrix A
                        - (float **B)   Refrence on output matrix B (square matrix)
         Output:        - void / call by reference
         */
        
        for(int i = 0; i < m; i++){
            for(int j = 0; j < m; j++){
                B[i][j] = 0.0;
                for(int k = 0; k < l; k++){
                    B[i][j] += A[k][j] * A[k][i];
                }
            }
        }
    }
    
    
    void invertMatrix(float **matrix, float **inv, int l){
        
        /*
         Description:   - Inverts matrix with gaussian elimination method (Remark: Actually we have to check whether the invert exists at least)
         Input:         - (float **A)   Reference on input matrix matrix
                        - (float **B)   Refrence on output matrix inv
         Output:        - void / call by reference
         */
        
        int i,j,k;
        float t;
        float **a = new float*[l];
        for(int i = 0; i < l; ++i){
            a[i] = new float[l];
        }
        for(i=0;i<l;i++){
            for(j=0;j<l;j++){
                a[i][j] = matrix[i][j];
            }
        }
        // fill the unit matrix next to the original matrix
        for(i=0;i<l;i++){
            for(j=l;j<2*l;j++){
                if(i==j-l){
                    a[i][j]=1;
                }
                else{
                    a[i][j]=0;
                }
            }
        }
        for(i=0; i<l; i++){
            t=a[i][i];
            // normalize row according to leading entry
            for(j=i; j<2*l; j++){
                a[i][j]=a[i][j]/t;
            }
            // elimination step
            for(j=0; j<l; j++){
                if(i!=j){
                    t = a[j][i];
                    for(k=0; k<2*l; k++){
                        a[j][k]= a[j][k] - t * a[i][k];
                    }
                }
            }
        }
        for(i=0;i<l;i++){
            for(j=l;j<2*l;j++){
                inv[i][j - l] = a[i][j];
            }
        }
    }
}

namespace ioOperation {
    
    void loadData(std::string pathX, float **X, std::string pathy, float *y, int l, int m, bool intercept){
        
        /*
         Description:   - Loads the data matrix X and target vector y
         Input:         - (string pathX)    Path to the data matrix X
                        - (float **X)       Reference on the data matrix
                        - (string pathy)    Path to the target vector y
                        - (float *y)        Reference on the target vector y
                        - (int l)           samples (rows) of data matrix
                        - (int m)           features (cols) of data matrix
                        - (bool intercept)  Boolean whether to fit the intercept as well
         Output:        - void / call by reference
         */
        
        std::ifstream data(pathX);
        std::ifstream target(pathy);
        
        if (!data) {
            std::cout << "Cannot open data matrix.";
            return;
        }
        if (!target) {
            std::cout << "Cannot open data matrix.";
            return;
        }
        
        for(int row = 0; row < l; ++row){
            
            std::string line_data;
            std::string line_target;
            std::getline(data, line_data);
            std::getline(target, line_target);
            
            y[row] = std::stof(line_target);
            X[row][0] = std::stof(line_data);
            if(intercept){
                X[row][1] = 1.0;
            }
        }
        data.close();
        target.close();
    }
    
    void writeData(std::string path, float *theta, int m){
        
        /*
         Description:   - Writes the parameters for the best fitting line to the filesystem
         Input:         - (string path)     Storagepath for the parameters
                        - (float *theta)    Reference on the parameters
                        - int m             Number of parameters
         Output:        - void
         */
        
        std::ofstream f;
        f.open(path);
        for(int i = 0; i < m; ++i){
            f << theta[i];
            f << "\n";
        }
        f.close();
    }
}

int main(){
    
    int l, m;
    bool intercept;
    std::cout << "Number of rows: ";
    std::cin >> l;
    std::cout << "Number of columns: ";
    std::cin >> m;
    std::cout << "Fit intercept? ";
    std::cin >> intercept;
    
    if(intercept){
        m += 1;
    }
    
    float **X = new float*[l];
    float **partX = new float*[l];
    float **partXX = new float*[l];
    float **partXXX = new float*[m];
    float *y = new float[l];
    float *result = new float[m];
    
    for(int i = 0; i < l; ++i){
        X[i] = new float[m];
        partX[i] = new float[l];
        partXX[i] = new float[l];
    }
    for(int i = 0; i < m; ++i){
        partXXX[i] = new float[l];
    }
    
    ioOperation::loadData("X.txt", X, "y.txt", y, l, m, intercept);
    matrixOperation::matrixProductWithTranspose(X, partX, l, m);
    matrixOperation::invertMatrix(partX, partXX, m);
    matrixOperation::matrixProductTranspose(partXX, X, partXXX, l, m);
    matrixOperation::matrixVectorMultiplication(partXXX, y, result, m, l);
    ioOperation::writeData("output.txt", result, m);
    
    for( int i = 0 ; i < l ; i++ ){
        delete[] X[i];
        delete[] partX[i];
        delete[] partXX[i];
    }
    for( int j = 0 ; j < m ; j++ ){
        delete[] partXXX[j];
    }

    delete[] X;
    delete[] partX;
    delete[] partXX;
    delete[] partXXX;
    delete[] y;
    delete[] result;
    
    return 0;
}
