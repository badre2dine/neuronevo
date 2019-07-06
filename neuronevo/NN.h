#pragma once
#include<vector>
#include "Matrix.hpp"
#include "Matrix.cpp"
#define SIGMOID		sigmoid
#define SOFTAMAX	softmax
#define IDENTITY	identity
#define RANDOM		random
#define TANH		tanH
#define RELU		ReLu
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


double random(double x);

//https://en.wikipedia.org/wiki/Activation_function

Matrix<double> identity(Matrix<double>& m);
Matrix<double> sigmoid(Matrix<double>& m);
Matrix<double> softmax(Matrix<double>& m);
Matrix<double> tanH(Matrix<double>& m);
Matrix<double> ReLu(Matrix<double>& m);

//derivative

Matrix<double> didentity(Matrix<double>& m);
Matrix<double> dsigmoid(Matrix<double>& m);
Matrix<double> dsoftmax(Matrix<double>& m);
Matrix<double> dtanH(Matrix<double>& m);
Matrix<double> dReLu(Matrix<double>& m);