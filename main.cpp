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
#include <cmath>

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
    
    
    bool invertMatrix(float **matrix, float **inv, int l){
        
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
            a[i] = new float[2 * l];
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
            int pivot = i;
            for(j = i + 1; j < l; j++){
                if(std::fabs(a[j][i]) > std::fabs(a[pivot][i])){
                    pivot = j;
                }
            }
            if(std::fabs(a[pivot][i]) < 1e-7f){
                for(int row = 0; row < l; ++row){
                    delete[] a[row];
                }
                delete[] a;
                return false;
            }
            if(pivot != i){
                for(k = 0; k < 2 * l; k++){
                    float tmp = a[i][k];
                    a[i][k] = a[pivot][k];
                    a[pivot][k] = tmp;
                }
            }
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
        for(i = 0; i < l; ++i){
            delete[] a[i];
        }
        delete[] a;
        return true;
    }
}

namespace ioOperation {
    
    bool loadData(std::string pathX, float **X, std::string pathy, float *y, int l, int m, bool intercept){
        
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
        int featureCount = intercept ? m - 1 : m;
        
        if (!data) {
            std::cout << "Cannot open data matrix.";
            return false;
        }
        if (!target) {
            std::cout << "Cannot open target vector.";
            return false;
        }
        
        for(int row = 0; row < l; ++row){
            
            std::string line_data;
            std::string line_target;
            std::getline(data, line_data);
            std::getline(target, line_target);
            if(line_data.empty() || line_target.empty()){
                std::cout << "Input files contain fewer rows than expected.";
                return false;
            }
            for(char &c : line_data){
                if(c == ','){
                    c = ' ';
                }
            }
            std::stringstream featureStream(line_data);
            for(int col = 0; col < featureCount; ++col){
                if(!(featureStream >> X[row][col])){
                    std::cout << "Data matrix row has fewer columns than expected.";
                    return false;
                }
            }
            
            y[row] = std::stof(line_target);
            if(intercept){
                X[row][featureCount] = 1.0;
            }
        }
        data.close();
        target.close();
        return true;
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
    float **partX = new float*[m];
    float **partXX = new float*[m];
    float **partXXX = new float*[m];
    float *y = new float[l];
    float *result = new float[m];
    
    for(int i = 0; i < l; ++i){
        X[i] = new float[m];
    }
    for(int i = 0; i < m; ++i){
        partX[i] = new float[m];
        partXX[i] = new float[m];
        partXXX[i] = new float[l];
    }
    
    if(!ioOperation::loadData("X.txt", X, "y.txt", y, l, m, intercept)){
        for(int i = 0; i < l; ++i){
            delete[] X[i];
        }
        for(int i = 0; i < m; ++i){
            delete[] partX[i];
            delete[] partXX[i];
            delete[] partXXX[i];
        }
        delete[] X;
        delete[] partX;
        delete[] partXX;
        delete[] partXXX;
        delete[] y;
        delete[] result;
        return 1;
    }
    matrixOperation::matrixProductWithTranspose(X, partX, l, m);
    if(!matrixOperation::invertMatrix(partX, partXX, m)){
        std::cout << "Normal equation matrix is singular and cannot be inverted.";
        for(int i = 0; i < l; ++i){
            delete[] X[i];
        }
        for(int i = 0; i < m; ++i){
            delete[] partX[i];
            delete[] partXX[i];
            delete[] partXXX[i];
        }
        delete[] X;
        delete[] partX;
        delete[] partXX;
        delete[] partXXX;
        delete[] y;
        delete[] result;
        return 1;
    }
    matrixOperation::matrixProductTranspose(partXX, X, partXXX, l, m);
    matrixOperation::matrixVectorMultiplication(partXXX, y, result, m, l);
    ioOperation::writeData("output.txt", result, m);
    
    for( int i = 0 ; i < l ; i++ ){
        delete[] X[i];
    }
    for( int j = 0 ; j < m ; j++ ){
        delete[] partX[j];
        delete[] partXX[j];
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
