/*
Autor: Álvaro Martín Cortinas
Fecha: 07/09/2016
Lugar: Madrid

Esta librería contiene las funciones desarrolladas
de manera manual por el autor del proyecto.
Son necesarias para el funcionamiento del reconocedor.

Cada función está brevemente explicada en su
declaración.

Para compilarlo y ejecutarlo es necesario tener instalado
las librerías de visión computarizada OpenCV, y de inteligencia
artificial y redes neuronales Fast Artificial Neural Network
(FANN).
*/


// Bibliotecas incluidas
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <doublefann.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Evitamos mencionar todo el rato las clases de las funciones cv:: y std::
using namespace cv;
using namespace std;

// Constantes para los tipos de señales
unsigned const int CIRCLE = 0;
unsigned const int CIRCLE_FILLED = 1;
unsigned const int FORBIDDEN = 2;
unsigned const int STOP = 3;
unsigned const int TRIANGLE = 4;
unsigned const int TRIANGLE_REVERSE = 5;

// Constantes para las distintas señales que reconoce
unsigned const int CIRCULACION_PROHIBIDA = 0;
unsigned const int ADELANTAR = 1;
unsigned const int VEL_MAX_100 = 2;
unsigned const int VEL_MAX_40 = 3;
unsigned const int OB_DCHA = 4;
unsigned const int OB_IZDA = 5;
unsigned const int OB_FRENTE = 6;
unsigned const int ROTONDA = 7;
unsigned const int LUCES_CORTO_ALCANCE = 8;
unsigned const int SIGNAL_STOP = 9;
unsigned const int SIGNAL_CEDA = 10;
unsigned const int SIGNAL_PROHIBIDO = 11;
unsigned const int PRIORIDAD = 12;
unsigned const int RESALTO = 13;
unsigned const int INTERSECCION_GIRATORIA = 14;
unsigned const int CURVA_DCHA = 15;
unsigned const int CURVA_IZDA = 16;
unsigned const int CURVAS_DCHA = 17;
unsigned const int CURVAS_IZDA = 18;
unsigned const int NINOS = 19;
unsigned const int CICLISTAS = 20;
unsigned const int NADA = 21;

/*************************************************************/
////
// FUNCIONES GENERALES
////

string type2str(int type);

// Devuelve el porcentaje en el que coinciden ambas imagenes (normalizadas: = tamaño)
unsigned int matchImages(Mat img_1, Mat img_2, unsigned int width = 50, unsigned int height = 50){

	// Las binarizamos para simplifcar la tarea
	threshold(img_1, img_1, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	threshold(img_2, img_2, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// Son binarias. Comparamos pixel a pixel la diferencia y vemos cuantos coinciden
	Scalar intensity1;
	Scalar intensity2;
	unsigned int pixelsIguales = 0;
	for (unsigned int y = 0; y < height; y++){
		for (unsigned int x = 0; x < width; x++){
			intensity1 = img_1.at<uchar>(Point(x, y)); // Devuelve la intensidad del pixel en img_1 en el punto P(x, y)
			intensity2 = img_2.at<uchar>(Point(x, y)); // Igual con img_2
			if (intensity1 == intensity2)
				pixelsIguales++;
		}
	}
	
	// Calculamos los que coinciden en comparacion con el total
	unsigned int semejanza = (100 * pixelsIguales) / (width * height);

	return semejanza;	
}

// Devuelve el tipo en string de un objeto Mat
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


// Devuelve el numero mayor
double max(double a, double b){
	if (a > b){
		return a;
	} else {
		return b;
	}
}

// Prepara la imagen para el reconocedor. img es la imagen a preparar y dst la que queremos que nos devuelva preparada
Mat prepareImageForRecognition(Mat img){

	imshow("Signal detected", img);
	
	cout << "Preparing image for recognition!\n";
	
	Mat dst = img.clone(); // La clonamos para no devolver la original tocada

	// Pasamos la imagen a grises
	cvtColor(img, dst, CV_BGR2GRAY);

	// La ponemos a 50x50 
	resize(dst, dst, Size(50, 50));

	// Contrast stretching
	normalize(dst, dst, 0, 255, CV_MINMAX);

	// Binarizamos (deprecated: Dificil saber cuando el fondo va a ser blanco o negro)
	//threshold(dst, dst, 40, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU); // Binarizamos

	return(dst);
}

// Obtiene el valor de los píxeles de una imagen
double * getPixelsValue(Mat img, unsigned int image_pixels){
	double * pixels = (double *)malloc(image_pixels * sizeof(double));
	unsigned int indice = 0;
	for (int y = 0; y < img.size().height; y++){
		for(int x = 0; x < img.size().width; x++){
			Scalar intensity = img.at<uchar>(Point(x,y));
			pixels[indice] = intensity[0];
			indice++;
		}
	}
	return pixels;
}

// Quita el ruido del fondo por fuera del contorno introducido en una imagen en escala de grises con fondo negro
Mat removeBackground(Mat img, vector<Point> contour){
	
	// Clonamos para no modificar la original
	Mat dst = img.clone();

	// Suponemos que la imagen ya esta en gris
	vector<vector<Point> > contours;
	contours.push_back(contour);
	drawContours(dst, contours, 0, Scalar(150,0,0), CV_FILLED, 8);
	//imshow("Contour", dst);
	
	// Obtenemos los valores de los pixeles uno a uno y pintamos todo lo que no sea de 150 o blanco de blanco (0)
	for (int y = 0; y < dst.size().height; y++){
		for (int x = 0; x < dst.size().width; x++){
			Scalar intensity = dst.at<uchar>(Point(x,y));
			if (intensity[0] != 150 && intensity[0] != 0)
				dst.at<uchar>(Point(x,y)) = 0;
		}
	}
	//imshow("Just signal", dst);

	// Obtenemos la parte del contorno en img
	drawContours(dst, contours, 0, Scalar(255,0,0), CV_FILLED, 8);
	Mat nonBackground;
	img.copyTo(nonBackground, dst);
	//imshow("Non Background", nonBackground);
	//waitKey(0);
	
	return(nonBackground);
}

// Pinta el fondo dada la señal en blanco y el fondo en negro
// NOTA: A LAS SEÑALES AZULES HAY QUE INVERTIRLAS ANTES
Mat paintBackground(Mat img){

	//cout << "Painting background\n";

	Mat dst = img.clone();
	resize(dst, dst, Size(50,50));
	//imshow("Image before background", dst);
        //imwrite("image_before_background.jpg", dst);
	//waitKey(0);
	Scalar intensity;

	// Pintamos de izquierda a derecha
	//cout << "\n->\n";
	for (int y = 0; y < dst.size().height; y++){
		for (int x = 0; x < dst.size().width; x++){
			// Si es negro, lo hacemos blanco. Si es blanco, bajamos de columna y empezamos de 0
			//printf("Pixel (x: %d,y: %d)\t", x, y);
			intensity = dst.at<uchar>(Point(x,y));
			if (intensity[0] < 190){
				//cout << "Coloreando\t";
				dst.at<uchar>(Point(x,y)) = 255;
			} else {
				if (y < (dst.size().height - 1)){
					y++;
				} else {
					y = dst.size().height;
					break;
				}
				x = -1;
			}
		}
	}

	// Pintamos de derecha a izquierda
	//cout << "\n<-\n";
        for (int y = 0; y < dst.size().height; y++){
                for (int x = dst.size().width - 1; x > -1; x--){
                        // Si es negro, lo hacemos blanco. Si es blanco, bajamos de columna y empezamos de 0
                        intensity = dst.at<uchar>(y,x);
			if (intensity[0] < 190){
				//cout << "Coloreando\t";
                                dst.at<uchar>(Point(x,y)) = 255;
                        } else {
				//cout << "Blanco encontrado\n";
                                if (y < (dst.size().height - 1)){
                                        y++;
                                } else {
                                        y = dst.size().height;
                                        break;
                                }
                                x = dst.size().width; // Empezamos de nuevo por la izquierda (va a restarle uno)
                        }
                }
        }

	// Pintamos de abajo a arriba
	//cout << "\n<-\n";
        for (int x = 0; x < dst.size().width; x++){
                for (int y = dst.size().height - 1; y > -1; y--){
                        // Si es negro, lo hacemos blanco. Si es blanco, bajamos de columna y empezamos de 0
                        intensity = dst.at<uchar>(y,x);
                        if (intensity[0] < 190){
                                //cout << "Coloreando\t";
                                dst.at<uchar>(Point(x,y)) = 255;
                        } else {
                                //cout << "Blanco encontrado\n";
                                if (x < (dst.size().width - 1)){
                                        x++;
                                } else {
                                        x = dst.size().width;
                                        break;
                                }
                                y = -1; // Empezamos de nuevo por abajo (va a restarle uno)
                        }
                }
        }

	// Pintamos de arriba a abajo
        //cout << "\n<-\n";
        for (int x = 0; x < dst.size().width; x++){
                for (int y = 0; y < dst.size().height; y++){
                        // Si es negro, lo hacemos blanco. Si es blanco, bajamos de columna y empezamos de 0
                        intensity = dst.at<uchar>(y,x);
                        if (intensity[0] < 190){
                                //cout << "Coloreando\t";
                                dst.at<uchar>(Point(x,y)) = 255;
                        } else {
                                //cout << "Blanco encontrado\n";
                                if (x < (dst.size().width - 1)){
                                        x++;
                                } else {
                                        x = dst.size().width;
                                        break;
                                }
                                y = dst.size().width; // Empezamos de nuevo por abajo (va a restarle uno)
                        }
                }
        }

	//imshow("Imagen coloreada", dst);
	//waitKey(0);
	//cout << "Imagenes coloreadas\n";
	
	return(dst);
}

// Muestra las imagenes introducidas a tamaño 50x50 (se presupone asi) mostrando como principal img1
void displayImages(const char * title, Mat img1, vector<Mat> realSignals, vector<Mat> idealSignals, unsigned int nSignals){
	
	if (nSignals == 0){ cout << "No signals detected!\n"; exit(1); }


	// Establecemos el tamaño de la imagen que vamos a desplegar en la ventana
	unsigned int dstWidth;
	unsigned int dstHeight;
	unsigned int x = 0;
	unsigned int y = 0;
	unsigned int columnas_filas; // Numero de columnas/filas necesarios
	Mat dst;
	
	// Le cambiamos un poco el tamaño para que quepa
	while (img1.cols > 1000 || img1.rows > 1000){ resize(img1, img1, Size(img1.cols / 2, img1.rows / 2)); }
	//cout << "Image width: " << img1.cols << img1.size().width << endl;
	//cout << "Image height: " << img1.rows << img1.size().height << endl; 
	//imshow("Resized image", img1);
	//waitKey(0);
		
    	if (img1.size().width > img1.size().height){
		columnas_filas = (int)((50 * nSignals) / img1.size().width + 1);
		dstWidth = img1.size().width;
		dstHeight = img1.size().height + columnas_filas * 50 * 2 + (columnas_filas - 1) * 5; // Reservamos el espacio para las dos señales y 5 de margen
		
		dst = Mat(Size(dstWidth, dstHeight), img1.type()); // make dstWidth x dstHeight matrix filled with 1 * 255
                dst = Scalar(255, 240,216);
		//imshow("dst init", dst);
                //waitKey(0);
  
  
                // Copiamos la imagen inicial en dst
                //cout << "Cols & width: " << img1.cols << " & " << img1.size().width << "\tRows & height: " << img1.rows << " & " << img1.size().height << endl;
                Rect roi(x, y, img1.cols, img1.rows);
                img1.copyTo(dst(roi));
                y = img1.size().height;
                //cout << "First image copied!\n";
                //imshow("dst", dst);
                //waitKey(0);
 
 
                for (unsigned int i = 0; i < nSignals; i++){
                        //cout << i << ":\n";
                        // Establecemos el mismo tipo para poder copiarlas
                        realSignals[i].convertTo(realSignals[i], dst.type());
                        idealSignals[i].convertTo(idealSignals[i], dst.type());
                        //printf("Channels dst, real image, ideal image: %d, %d, %d\n", dst.channels(), realSignals[i].channels(), idealSignals[i].channels());
                        // Copiamos la señal real en x e y de dst
                        roi = Rect(x, y, realSignals[i].cols, realSignals[i].rows);
                        realSignals[i].copyTo(dst(roi));
                        y = y + realSignals[i].size().height;
                        //cout << "Imagen real copiada dude\t";
                        // Copiamos la señal ideal en x e y + 50 de dst (debajo)
                        roi = Rect(x, y, idealSignals[i].cols, idealSignals[i].rows);
                        idealSignals[i].copyTo(dst(roi));
                        //cout << "Imagen ideal copiada dude\n";
                        if ((unsigned int)(x + realSignals[i].size().width) < (unsigned int)img1.size().width){
                                x = x + realSignals[i].size().width;
                                y = img1.size().height;
                        } else {
                                x = 0;
                                y = y + realSignals[i].size().height + 5;
                        }
                }

	} else {
		columnas_filas = (int)((50 * nSignals) / img1.size().height + 1);
		//cout << "Columnas/filas: " << columnas_filas << endl;
		dstWidth = img1.size().width + columnas_filas * 50 * 2 + (columnas_filas - 1) * 5;
		dstHeight = img1.size().height;
		//cout << "dstWidth & dstHeight: " << dstWidth << " & " << dstHeight << endl;
		//waitKey(0);
		
		dst = Mat(Size(dstWidth, dstHeight), img1.type()); // make dstWidth x dstHeight matrix filled with 1 * 255
		dst = Scalar(255,240,216);
		//imshow("dst init", dst);
		//waitKey(0);
		
		
		// Copiamos la imagen inicial en dst
		//cout << "Cols & width: " << img1.cols << " & " << img1.size().width << "\tRows & height: " << img1.rows << " & " << img1.size().height << endl;
		Rect roi(x, y, img1.cols, img1.rows);
		img1.copyTo(dst(roi));	
		x = img1.size().width;
		//cout << "First image copied!\n";
		//imshow("dst", dst);
		//waitKey(0);


		for (unsigned int i = 0; i < nSignals; i++){
			//cout << i << ":\n";
			// Establecemos el mismo tipo para poder copiarlas
			realSignals[i].convertTo(realSignals[i], dst.type());
			idealSignals[i].convertTo(idealSignals[i], dst.type());
			//printf("Channels dst, real image, ideal image: %d, %d, %d\n", dst.channels(), realSignals[i].channels(), idealSignals[i].channels());
			// Copiamos la señal real en x e y de dst
			roi = Rect(x, y, realSignals[i].cols, realSignals[i].rows);
			realSignals[i].copyTo(dst(roi));
			x = x + realSignals[i].size().width;
			//cout << "Imagen real copiada dude\t";
			// Copiamos la señal ideal en x e y + 50 de dst (debajo)
			roi = Rect(x, y, idealSignals[i].cols, idealSignals[i].rows);
			idealSignals[i].copyTo(dst(roi));
			//cout << "Imagen ideal copiada dude\n";
			if ((unsigned int)(y + realSignals[i].size().height) < (unsigned int)img1.size().height){
				y = y + realSignals[i].size().height;
				x = img1.size().width;
			} else {
				y = 0;
				x = x + realSignals[i].size().width + 5;
			}
		}
	}
		
	// Mostramos la imagen finalmente
	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, dst);
	waitKey(0);
}
