/*
Autor: Álvaro Martín Cortinas
Fecha: 07/09/2016
Lugar: Madrid

Este programa escrito en C/C++ fue utilizado
dentro del proyecto para entrenar las redes
neuronales. El número de reiteraciones con
el que se entrenaba cada red se cambió
de forma experimental para obtener el mejor resultado.

Para compilarlo y ejecutarlo es necesario tener instalado
las librerías de inteligencia
artificial y redes neuronales Fast Artificial Neural Network
(FANN).
*/

#include <iostream>
#include <doublefann.h>

using namespace std;

int main(){
	// Creates a standar fully connected backpropagation neural network
	// with 7 input neurons, 2 hidden and 1 output
	struct fann *ann = fann_create_standard(4, 2500, 120, 120, 9);
	fann_set_activation_function_output(ann, FANN_ELLIOT);
	fann_set_activation_function_hidden(ann, FANN_ELLIOT_SYMMETRIC);
	// fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
	//fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	fann_print_parameters(ann);
	fann_train_on_file(ann, "./trainData/trainTriangle_v0.data", 10000, 10, 0.00001);
	fann_save(ann, "networks/triangle_v0.1.net");
	//fann_randomize_weights(ann, min_weight, max_weight);
	fann_destroy(ann);
}
