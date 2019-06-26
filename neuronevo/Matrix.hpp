#pragma once
#include <iostream>
#include<time.h>
using  namespace std;
template <typename T>
class Matrix
{
	T **data;
	int row_size, col_size;
public:
	Matrix();

	Matrix(int N, int M, T **);
	Matrix(int N, T *);
	Matrix(int N);
	Matrix(int N,int M);
	Matrix(Matrix  const &m);
	void random(int a,int b);//make a random element between a and b
	Matrix map(T (*function)(T));
	Matrix operator*(Matrix const &m);
	Matrix operator*(T m);
	Matrix operator~();
	Matrix transpose();
	int getRow_size();
	int getCol_size();
	Matrix operator+(Matrix const &m);
	Matrix operator-(Matrix const &m);
	Matrix operator=(Matrix const &m);
	T &operator()(int i,int j);
	T min();
	int argmin();
	T max();
	int argmax();
	void show();
	~Matrix();
};
template <typename T>
Matrix<T> operator*(T scaler, Matrix<T> const &m);
template<typename T, int N, int M>
T ** arrayAloc(T m[][M]);
