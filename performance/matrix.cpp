#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <thread>

class Matrix {
private:
    std::vector<std::vector<int>> data;
    int size;

public:
    
    Matrix(int size) : size(size) {        
    
        data.resize(size, std::vector<int>(size));
        for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                data[i][j] = std::rand() % 100;
            }
        }
    }

    static void multiplyThread(const Matrix& a, const Matrix& b, Matrix& result, int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < a.size; ++j) {
                double value = 0;
                for (int k = 0; k < a.size; ++k) {
                    value += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, value);
            }
        }
    }
    
    double get(int row, int col) const {
        return data[row][col];
    }

    void set(int row, int col, double value) {
        data[row][col] = value;
    }

    Matrix operator+(const Matrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

        Matrix result(size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result.set(i, j, get(i, j) + other.get(i, j));
            }
        }
        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

        Matrix result(size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result.set(i, j, get(i, j) - other.get(i, j));
            }
        }
        return result;
    }

    // Matrix multiplication
    Matrix  operator*(const Matrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Matrix sizes do not match");
        }

        Matrix result(size);

        int num_threads = 4;
        int chunk_size = size / num_threads;

        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            int start_row = i * chunk_size;
            int end_row = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
            threads.push_back(std::thread(multiplyThread, std::ref(*this), std::ref(other), std::ref(result), start_row, end_row));
        }
        
        return result;
    }

    
    Matrix transpose() const {
        Matrix result(size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                result.set(i, j, get(j, i));
            }
        }
        return result;
    }

    
    void show() {
        for(const auto& row : data) {
            for(const auto& elem : row) {
                std::cout << elem << ' ';
            }
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << " [Matrix Size]" << std::endl;
        return 1;
    }

    int size = std::atoi(argv[1]);

    if (size <= 0) {
        std::cout << "Please enter a positive integer for matrix size." << std::endl;
        return 1;
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    
    Matrix matrix1(size);
    Matrix matrix2(size);

    Matrix matrix_result = matrix1 * matrix2;
    matrix_result.show();
    (matrix1 + matrix2).show();
    (matrix1 - matrix2).show();

    return 0;
}