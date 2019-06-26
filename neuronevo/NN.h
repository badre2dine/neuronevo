#pragma once
#include<vector>
#include "Matrix.hpp"
#include "Matrix.cpp"
#define SIGMOID sigmoid
#define SOFTAMAX softmax
#define RANDOM random
class NN
{
	std::vector<Matrix<double>*> weights;
	std::vector<Matrix<double>*> bais;
	std::vector<Matrix<double>(*)(Matrix<double> &)> activationFunctions;
	
	int numberLayers;
public:
	NN();
	int getNumberLayers();
	Matrix<double> decision(double *data);
	void initial(double(*randtype)(double));
	void addLayer(int size, Matrix<double>(*activationFuntion)(Matrix<double> &));
	~NN();
};
Matrix<double> sigmoid(Matrix<double> &m);
Matrix<double> softmax(Matrix<double> &m);
double random(double x);
