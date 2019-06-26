#include "stdafx.h"
#include "Matrix.hpp"

template<typename T>
Matrix<T>::Matrix()
{
	data = NULL;
	this->row_size = this->col_size = 0;
}

template<typename T>
Matrix<T>::Matrix(int N,int M, T  ** m)
{
	col_size = M;
	row_size = N;
	data = new T*[N];
	for (size_t i = 0; i < N; i++)
	{
		data[i] = new T[M];
		for (size_t j = 0; j < M; j++)
		{
			data[i][j] = m[i][j];
		}
	}
}

template<typename T>
Matrix<T>::Matrix(int N, T *vector)
{
	col_size = N;
	row_size = 1;
	data = new T*[1];
	
		data[0] = new T[N];
		for (size_t j = 0; j < N; j++)
		{
			data[0][j] = vector[j];
		}
}

template<typename T>
Matrix<T>::Matrix(int N)
{
	col_size = row_size = N;
	data = new T*[N];
		for (size_t i = 0; i < N; i++)
		{
			data[i] = new T[N];
			for (size_t j = 0; j < N; j++)
			{
				data[i][j] = 0;
			}
		}
}

template<typename T>
Matrix<T>::Matrix(int N, int M)
{
	col_size = M;
	row_size = N;
	data = new T*[N];
		for (size_t i = 0; i < N; i++)
		{
			data[i] = new T[M];
			for (size_t j = 0; j < M; j++)
			{
				data[i][j] = 0;
			}
		}
}

template<typename T>
Matrix<T>::Matrix(Matrix<T> const & m)
{
	col_size = m.col_size;
	row_size = m.row_size;
	data = new T*[row_size];
	for (size_t i = 0; i < row_size; i++)
	{
		data[i] = new T[col_size];
		for (size_t j = 0; j < col_size; j++)
		{
			data[i][j] = m.data[i][j];
		}
	}
}

template<typename T>
void Matrix<T>::random(int a, int b)
{
	srand(time(NULL));
	for (size_t i = 0; i < this->row_size; i++)
	{
		
		for (size_t j = 0; j < this->col_size; j++)
		{

			data[i][j] = (rand() % (b-a+1))+a;
		}
	}
}

template<typename T>
Matrix<T> Matrix<T>::map(T (* function)(T))
{
	Matrix<T> product(*this);
	//cout << product.col_size;
	for (size_t i = 0; i < row_size; i++)
	{
		for (size_t j = 0; j < col_size; j++)
		{
			product.data[i][j] = (T)function(product.data[i][j]);
		}
		
	}
	return product;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> const &m)
{
	Matrix<T> product(this->row_size,m.col_size);
	for (size_t i = 0; i < product.row_size; i++)
	{
		for (size_t k = 0; k < col_size; k++)
		{

			for (size_t j = 0; j < product.col_size; j++)
			{
				product.data[i][j] += this->data[i][k] * m.data[k][j];
			}
		}
	}
	return product;
	
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T m)
{
	Matrix<T> product(*this);
	for (size_t i = 0; i < product.row_size; i++)
	{
		for (size_t j = 0; j < product.col_size; j++)
		{
			product.data[i][j] = m* product.data[i][j];
		}
	}
	return product;
}

template<typename T>
Matrix<T> Matrix<T>::operator~()
{
	Matrix<T> product(this->col_size, this->row_size);
	for (size_t i = 0; i < product.row_size; i++)
	{
			for (size_t j = 0; j < product.col_size; j++)
			{
				product.data[i][j] = this->data[j][i];
			}
	}
	return product;
}

template<typename T>
Matrix<T> Matrix<T>::transpose()
{
	Matrix<T> product;
	product = ~(*this);
	return product;
}

template<typename T>
int Matrix<T>::getRow_size()
{
	return this->row_size;
}

template<typename T>
int Matrix<T>::getCol_size()
{
	return this->col_size;
}
template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> const &m)
{
	Matrix<T> product(this->row_size, this->col_size);
	for (size_t i = 0; i < product.row_size; i++)
	{
			for (size_t j = 0; j < product.col_size; j++)
			{
				product.data[i][j] = this->data[i][j] + m.data[i][j];
			}
	}
	return product;

}




template<typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T> const & m)
{
	Matrix<T> product(this->row_size, this->col_size);
	for (size_t i = 0; i < product.row_size; i++)
	{
		for (size_t j = 0; j < product.col_size; j++)
		{
			product.data[i][j] = this->data[i][j] - m.data[i][j];
		}
	}
	return product;
}
template<typename T>
Matrix<T> Matrix<T>::operator=(Matrix<T> const & m)
{
	~Matrix();
	
	col_size = m.col_size;
	row_size = m.row_size;
	
	data = new T*[row_size];
	for (size_t i = 0; i < row_size; i++)
	{
		data[i] = new T[col_size];
		for (size_t j = 0; j < col_size; j++)
		{
			 data[i][j] = m.data[i][j];
			 
		}
	}
	return *this;
}
template<typename T>
T & Matrix<T>::operator()(int i, int j)
{
	return this->data[i][j];
}
template<typename T>
T Matrix<T>::min()
{
	T min = data[0][0];
	for (int i = 0; i<row_size; i++)
		for (int j = 0; j < col_size; j++)
		{
			if (data[i][j] < min)min = data[i][j];

		}
	return min;
}
template<typename T>
int Matrix<T>::argmin()
{
	int maxi = 0, maxj = 0;
	for (int i = 0; i<row_size; i++)
		for (int j = 0; j < col_size; j++)
		{
			if (data[i][j] < data[maxi][maxj]) { maxi = i; maxj = j; }

		}
	return maxi * col_size + maxj;
}
template<typename T>
T Matrix<T>::max()
{
	T max = data[0][0];
	for(int i=0;i<row_size;i++)
		for (int j = 0; j < col_size; j++)
		{
			if (data[i][j] > max)max = data[i][j];

		}
	return max;
}
template<typename T>
int Matrix<T>::argmax()
{
	int maxi=0,maxj =0;
	for (int i = 0; i<row_size; i++)
		for (int j = 0; j < col_size; j++)
		{
			if (data[i][j] > data[maxi][maxj]) { maxi = i; maxj =j; }

		}
	return maxi*col_size+maxj;
}
template<typename T>
void Matrix<T>::show()
{
	for (size_t i = 0; i < row_size; i++)
	{
		for (size_t j = 0; j < col_size; j++)
		{
			cout << data[i][j]<<"\t";
		}
		cout << "\n";
	}
}

template<typename T>
Matrix<T>::~Matrix()
{
	for (int i = 0; i < row_size; i++)
		delete[] data[i];
	delete[] data;
}

template<typename T>
Matrix<T> operator*(T scaler, Matrix<T> const &m)
{
	Matrix<T> temp(m);
	//temp = temp * scaler;
	return temp * scaler;
}

template<typename T, int N, int M>
T ** arrayAloc(T m[][M])
{
	T ** ar = new T*[N];
	for (int i = 0; i < N; i++)
	{
		ar[i] = new T[M];
		for (int j = 0; j < M; j++)
			ar[i][j] = m[i][j];
	}
	return ar;
}
