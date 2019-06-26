#include "stdafx.h"
#include "NN.h"
NN::NN()
{
	numberLayers = 0;
}

int NN::getNumberLayers()
{
	return this->numberLayers;
}

Matrix<double> NN::decision(double * data)
{
	//cout << weights[0]->getRow_size();
	Matrix<double> decesion(2,data);
	for (int i = 0; i < weights.size();i++) {
		//cout << weights[i]->getRow_size()<<"\n";
		decesion = decesion * (*weights[i]);
		decesion = decesion + *bais[i];
		decesion = activationFunctions[i](decesion);
	}
	return decesion;
}

void NN::initial(double(*randtype)(double))
{

	srand(time(NULL));
	for (int i = 0; i < weights.size(); i++) {
	
		*weights[i] = weights[i]->map(randtype);
		
	}
	
	
}

void NN::addLayer(int size, Matrix<double>(*activationFuntion)(Matrix<double> &) )
{
	
	if (this->numberLayers == 0)
	{
		weights.push_back(new Matrix<double>(1, size));
	}
	else if (this->numberLayers ==1)
	{
		
		int precedentLayer = weights[0]->getCol_size();
		
		delete weights[0];
		weights.clear();
		weights.push_back(new Matrix<double>(precedentLayer, size));
		bais.push_back(new Matrix<double>( 1,size));
		this->activationFunctions.push_back(activationFuntion);
	}
	else
	{
		weights.push_back(new Matrix<double>(weights[weights.size()-1]->getCol_size(), size));
		bais.push_back(new Matrix<double>(1, size));
		this->activationFunctions.push_back(activationFuntion);
	}
	this->numberLayers++;
}

NN::~NN()
{
	for (auto& weight : weights)
	{
		delete weight;

	}
	weights.clear();
	activationFunctions.clear();
	
}

Matrix<double> sigmoid(Matrix<double> &m)
{
	Matrix<double> result;
	result = m.map([](double x) {
		return 1 / (1 + exp(-x));

	});
		return result;
}

Matrix<double> softmax(Matrix<double>  &m)
{
	Matrix<double> result(m.getRow_size(),m.getCol_size());
	
	double sum = 0;
	for (int i = 0; i < m.getCol_size(); i++)
	{
		sum += exp(m(0, i));
	}
	for (int i = 0; i < m.getCol_size(); i++)
	{
		result(0, i) = exp(m(0, i)) / sum;
	}
	
	return result;
}

double random(double x)
{
	
	
	return  (rand() / (double)RAND_MAX);
}
