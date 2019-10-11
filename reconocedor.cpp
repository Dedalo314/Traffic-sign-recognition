/*
Autor: Álvaro Martín Cortinas
Fecha: 07/09/2016
Lugar: Madrid

Este programa escrito en C/C++ es el programa principal
del proyecto y por lo tanto del reconocedor. Ejecuta
todas las acciones necesarias a través de la clase
reconocedor.

Tiene como entradas opcionales "save2train", que se utilizó
para guardar las imágenes como imágenes de entrenamiento
de las redes; y "savePixelsOnFile", que extrae el valor de los
píxeles de la imagen introducida a 50x50 y los guarda
en el fichero dado.

Tiene como entrada obligatoria la ruta de la imagen
dentro del ordenador.

Para compilarlo y ejecutarlo es necesario tener instalado
las librerías de visión computarizada OpenCV, y de inteligencia
artificial y redes neuronales Fast Artificial Neural Network
(FANN).
*/


#include "ClassReconocedor.h"
//#include "MyLibrary.h"

/*****************************************************************/
////
// FUNCION PRINCIPAL
////

int main(int argc, char *argv[]){

	cout << "Usage: ./reconocedor <image.ext> [save2train] [savePixelsOnFile <filename>]\n";
	//cout << argc << endl;

	// Comprobamos que han metido la direccion de la imagen
	if(argv[1] == NULL){ cout << "No input image!\n"; exit(-1); }
	
	// Variables
	Reconocedor reconocedor(argv[1]); // Lo inicializamos con la imagen introducida
	
	if (argc == 2){

		// Detectamos las señales
		reconocedor.signalDetection();
		
		// Obtenemos el simbolo de dentro de la señal
		reconocedor.getInnerSymbol();

		// Las reconocemos / clasificamos
		// NOTA: PEDIR REDES COMO PARAMETROS
		reconocedor.recognize("networks/circle_v0.1.net", "networks/circle_filled_v0.net", "networks/triangle_v0.1.net");
		
		// Presenta el resultado
		cout << "Showing!\n";
		reconocedor.show();
	} else {

		// Usado para el entrenamiento
		if (strcmp(argv[2], "save2train") == 0)
			reconocedor.save2train("train.jpg", "train2.jpg");

		if (strcmp(argv[2], "savePixelsOnFile") == 0 && argc == 4){ reconocedor.createGray(argv[1]); reconocedor.savePixelsOnFile(argv[3]); }
	}
}
