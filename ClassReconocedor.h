/*
Autor: Álvaro Martín Cortinas
Fecha: 07/09/2016
Lugar: Madrid

Esta clase de C++ sirve para reconocer señales de tráfico en
imágenes de carretera ambiente. Utiliza las funciones
desarrolladas de manera manual por el mismo autor y 
recogidas en MyLibrary.h.

Para compilarlo y ejecutarlo es necesario tener instalado
las librerías de visión computarizada OpenCV, y de inteligencia
artificial y redes neuronales Fast Artificial Neural Network
(FANN).
*/

#include "MyLibrary.h"

/***********************************************************************/
////
// CLASES
////

// Clase para el reconocedor 
class Reconocedor{
	Mat src;
	vector<vector<Point> > goodContours; // Va a contener los contornos
	vector<Rect> goodBoundRect; // Va a contener los rectangulos de las señales adecuadas
	unsigned int * signalsType;
	unsigned int * signalsDef; // Contiene finalmente las señales presentes
	public:
		unsigned int numSignalsBlue;
		unsigned int numSignalsRed;
		vector<Mat> signals;		
		vector<Mat> innerSymbols;

		// Esta funcion obtiene las señales presentes y las guarda en signalsRed y signalsBlue. Además guarda en signalsType el tipo de señal
		// que es cada una, por orden, primero rojas y luego azules
		void signalDetection();
		
		// Constructores
		void create(const char * imgPath) 
			{ src = imread(imgPath, IMREAD_COLOR); cout << "Image: " << imgPath << endl; this->srcEmpty(); imshow("Source image", src); waitKey(0); }
		void createGray(const char * imgPath)
			{ src = imread(imgPath, IMREAD_GRAYSCALE); cout << "Image gray: " << imgPath << endl; this->srcEmpty(); imshow("Source image", src); waitKey(0); }
		Reconocedor(const char * imgPath) 
			{ src = imread(imgPath, IMREAD_COLOR); cout << "Image: " << imgPath << endl; this->srcEmpty(); imshow("Source image", src); waitKey(0); }
	
		// Comprueba si src ha sido cargada. Si no, se sale
		void srcEmpty(){ if(src.empty()){ cout << "There is no image in the Reconocedor! Please initialize the object. Is the name right?\n"; exit(-1);  }  }

		// Tranformador para entrenamiento
		void save2train(const char * filename1, const char * filename2);

		// Guardar en el archivo de entrenamiento
		void savePixelsOnFile(const char * filename);

		// Reconoce las señales y lo guarda en signals
		void recognize(const char * netCircle, const char * netCircleFilled, const char * netTriangle);

		// Obtiene el simbolo de dentro de la señal
		void getInnerSymbol();

		// Finalmente muestra el resultado, las señales detectadas y donde se han detectado
		void show();
	
};


/************************************************************************/
//////
// FUNCIONES DE LAS CLASES
/////

// Muestra las señales como se ve en el output
void Reconocedor::show(){

	// Modificamos la source image para señalar las señales detectadas
	Scalar color = Scalar(0, 255, 0);
	for (unsigned int i = 0; i < goodBoundRect.size(); i++){
		rectangle(src, goodBoundRect[i].tl(), goodBoundRect[i].br(), color, 2, 8, 0);
	}
		
	// Leemos una a una las señales
	vector<Mat> idealSignals(signals.size());
	for (unsigned int i = 0; i < (numSignalsRed + numSignalsBlue); i++){
		switch(signalsDef[i]){
			case CIRCULACION_PROHIBIDA:
				idealSignals[i] = imread("./Imagenes_señales/circulacion_prohibida.jpg", IMREAD_COLOR);
				resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case ADELANTAR:
				idealSignals[i] = imread("./Imagenes_señales/adelantar7.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case VEL_MAX_40:
				idealSignals[i] = imread("./Imagenes_señales/40.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case VEL_MAX_100:
				idealSignals[i] = imread("./Imagenes_señales/velMax100.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case OB_DCHA:
				idealSignals[i] = imread("./Imagenes_señales/obDcha.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case OB_IZDA:
				idealSignals[i] = imread("./Imagenes_señales/obIzda.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case OB_FRENTE:
				idealSignals[i] = imread("./Imagenes_señales/obFrente.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case ROTONDA:
				idealSignals[i] = imread("./Imagenes_señales/rotonda5.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case LUCES_CORTO_ALCANCE:
				idealSignals[i] = imread("./Imagenes_señales/alumbrado.jpeg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case PRIORIDAD:
				idealSignals[i] = imread("./Imagenes_señales/interseccion_prioridad.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case RESALTO:
				idealSignals[i] = imread("./Imagenes_señales/resalto.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case CURVA_DCHA:
				idealSignals[i] = imread("./Imagenes_señales/curva_peligrosa_dcha.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case CURVA_IZDA:
				idealSignals[i] = imread("./Imagenes_señales/curva_peligrosa_izda.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case CURVAS_DCHA:
				idealSignals[i] = imread("./Imagenes_señales/curvas_peligrosas_dcha.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case CURVAS_IZDA:
				idealSignals[i] = imread("./Imagenes_señales/curvas_peligrosas_izda.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case NINOS:
				idealSignals[i] = imread("./Imagenes_señales/niños.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case CICLISTAS:
				idealSignals[i] = imread("./Imagenes_señales/ciclistas.jpeg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case SIGNAL_STOP:
				idealSignals[i] = imread("./Imagenes_señales/stop.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case SIGNAL_CEDA:
				idealSignals[i] = imread("./Imagenes_señales/ceda1.png", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case SIGNAL_PROHIBIDO:
				idealSignals[i] = imread("./Imagenes_señales/prohibido3.jpg", IMREAD_COLOR);
                                resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;
			case NADA:
				idealSignals[i] = Mat::zeros(50, 50, signals[i].type());
				resize(idealSignals[i], idealSignals[i], Size(50,50));
				break;

		}

		// Nos aseguramos que las señales estan a 50x50
	        resize(signals[i], signals[i], Size(50,50));
		//imshow("Ideal Signal", idealSignals[i]);
		//waitKey(0);
	}
	
	// Finalmente mostramos el resultado
        displayImages("Output", src, signals, idealSignals, (this->numSignalsRed + this->numSignalsBlue));
}

// Obtiene el símbolo de dentro de la señal
void Reconocedor::getInnerSymbol(){
	
	//cout << "Getting the inner symbol for each signal!\n";

	this->srcEmpty();

	// Comprobamos que signals no esta vacio y por lo tanto las señales se han detectado
        if (signals[0].empty()){ cout << "No signals detected!\n"; exit(-1);  }

	// Variables
	Mat maskWhite;
	innerSymbols.resize(signals.size());

	/*
        NOTA: Mejor coger mas posibles señales que no captarlas
        */
	for (unsigned int i = 0; i < (numSignalsBlue + numSignalsRed); i++){
		

		// Obtenemos la parte blanca de la imagen
        	inRange(signals[i], Scalar(140, 140, 140), Scalar(255, 255, 255), maskWhite);
        	//imshow("MaskWhite(normal)", maskWhite);

		// Binarizamos la imagen		
		if (signalsType[i] == CIRCLE_FILLED){
			//cout << "Inverting blue signal!\n";
			threshold(maskWhite, maskWhite, 40, 255, CV_THRESH_BINARY_INV);
		} else {
			threshold(maskWhite, maskWhite, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		}

	        // Obtenemos los contornos de la imagen mask filtrada
	        Mat maskWhite_copy = maskWhite.clone();
		Mat drawing = maskWhite.clone();
		vector<vector<Point> > contoursWhite;
		vector<Vec4i> hierarchyWhite;
	        findContours(maskWhite_copy, contoursWhite, hierarchyWhite, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
		for (unsigned int i = 0; i < contoursWhite.size(); i++){ drawContours(drawing, contoursWhite, i, 150, CV_FILLED, 8); }
		//imshow("Contours White", drawing);		
		//waitKey(0);
	
	        // Aproximamos los contornos a poligonos y obtenemos un cuadrado a su alrededor
	        vector<Point> contours_poly_white;
	        Rect boundRectWhite;
		Rect maxBoundRect;
		vector<Point> maxContour;
		int areaMax = 0;
        	for(unsigned int t = 0; t < contoursWhite.size(); t++){
        	        approxPolyDP(Mat(contoursWhite[t]), contours_poly_white, 3, true);
        	        boundRectWhite = boundingRect(Mat(contours_poly_white));

			// Calculamos la maxima area
			areaMax = max(areaMax, (int)contourArea(contours_poly_white, false));

			// Si la maxima area coincide con ese rectangulo, guardamos el rectangulo y su contorno
			if (areaMax == (int)contourArea(contours_poly_white, false)){ maxBoundRect = boundRectWhite; maxContour = contours_poly_white; }
        	} 
		
		// Obtiene cada uno de los simbolos blancos sin fondo
		innerSymbols[i] = removeBackground(maskWhite, maxContour)(maxBoundRect).clone();
		
		// Pinta el fondo negro en blanco y pone la imagen a 50x50
		innerSymbols[i] = paintBackground(innerSymbols[i]);

		// Aplica un poco de morfologia matematica para simplificarla
		Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1));
		dilate(innerSymbols[i], innerSymbols[i], element);
		erode(innerSymbols[i], innerSymbols[i], element);
		//imshow("Inner Symbol", innerSymbols[i]);
		//waitKey(0);
	}
}

// Funcion que clasifica las señales y devuelve en signalsDef el int correspondiente a cada una que haya 
void Reconocedor::recognize(const char * netCircle, const char * netCircleFilled, const char * netTriangle){
	
	//cout << "\nRecognizing!\n";

	// Comprobamos que el objeto ha sido inicializado
	this->srcEmpty();

	// Comprobamos que signals no esta vacio y por lo tanto las señales se han detectado
	if (signals[0].empty()){ cout << "No image preprocessing done!\n"; exit(-1);  }

	// Variables
	struct fann * ann;
	double * pixelsValue;
	const double * output;	
	double maxSignal;
	signalsDef = (unsigned int *)malloc((numSignalsRed + numSignalsBlue) * sizeof(int));

	// Introducimos en la red una a una las señales obtenidas por el detector y guardadas en vector<Mat> signals
	for (unsigned int i = 0; i < (numSignalsRed + numSignalsBlue); i++){

		switch(signalsType[i]){
			case CIRCLE:

				//cout << "Clasificando circulo\n";

				// Obtiene la red para circle y prepara la imagen
				ann = fann_create_from_file(netCircle);
				pixelsValue = (double *)malloc(2500 * sizeof(double));
				output = (double *)malloc(4 * sizeof(double));
				//imshow("Processed Image", innerSymbols[i]);
				//waitKey(0);

				// Obtenemos el valor de los pixeles (INPUT)
				pixelsValue = getPixelsValue(innerSymbols[i], 2500);
			
				// Ejecutamos la red
				output = fann_run(ann, pixelsValue);
				
				cout << "\nOutput CIRCLE: " << endl; 
				printf("Circulacion prohibida: %f\nAdelantar: %f\nVel. Max. 40: %f\nVel. Max. 100: %f\n\n", output[0], output[1], output[2], output[3]);

				// Obtiene la máxima salida
				maxSignal = max(max(max(output[0], output[1]), output[2]), output[3]);

				if (maxSignal < 0.8){
					//cout << "No image detected.\n";
					signalsDef[i] = NADA;
				} else {
					if(maxSignal == output[0])
						signalsDef[i] = CIRCULACION_PROHIBIDA;
						
					if(maxSignal == output[1])
						signalsDef[i] = ADELANTAR;
						
					if(maxSignal == output[2])
						signalsDef[i] = VEL_MAX_40;
						
					if(maxSignal == output[3])
						signalsDef[i] = VEL_MAX_100;
				}

				break;

			case CIRCLE_FILLED:

				//cout << "Clasificando circle_filled\n";

				// Obtiene la red para circle y prepara la imagen
                                ann = fann_create_from_file(netCircleFilled);
                                pixelsValue = (double *)malloc(2500 * sizeof(double));
                                output = (double *)malloc(5 * sizeof(double));
                                //imshow("Processed Image", innerSymbols[i]);
                                //waitKey(0);
 
                                // Obtenemos el valor de los pixeles (INPUT)
                                pixelsValue = getPixelsValue(innerSymbols[i], 2500);
 
                                // Ejecutamos la red
                                output = fann_run(ann, pixelsValue);
 
                                cout << "\nOutput CIRCLE_FILLED: " << endl;
				printf("Sent. Ob. Dcha.: %f\nSent. Ob. Izda.: %f\nSent. Ob. Frente: %f\nRotonda: %f\nLuces corto alcance: %f\n\n", 
					output[0], output[1], output[2], output[3], output[4]);
		
				// Obtenemos el output maximo
				maxSignal = max(max(max(max(output[0], output[1]), output[2]), output[3]), output[4]);
				if (maxSignal < 0.8){
					signalsDef[i] = NADA;
					break;
				}

				if(maxSignal == output[0])
					signalsDef[i] = OB_DCHA;
					
				if(maxSignal == output[1])
					signalsDef[i] = OB_IZDA;
					
				if(maxSignal == output[2])
					signalsDef[i] = OB_FRENTE;
					
				if(maxSignal == output[3])
					signalsDef[i] = ROTONDA;
						
				if(maxSignal == output[4])
					signalsDef[i] = LUCES_CORTO_ALCANCE;
				
                                break;
				
			case FORBIDDEN:

				//cout << "Es un prohibido\n";

				// Solo puede ser el prohibido, la clasificamos directamente
				signalsDef[i] = SIGNAL_PROHIBIDO;
				break;
			
			case STOP:
	
				//cout << "Es un stop\n";

				// Solo puede ser el stop, la clasificamos directamente
				signalsDef[i] = SIGNAL_STOP;
				break;

			case TRIANGLE:

				//cout << "Clasificando triangulo\n";
 
                                // Obtiene la red para circle y prepara la imagen
                                ann = fann_create_from_file(netTriangle);
                                pixelsValue = (double *)malloc(2500 * sizeof(double));
                                output = (double *)malloc(9 * sizeof(double));
                                //imshow("Processed Image", innerSymbols[i]);
                                //waitKey(0);
 
                                // Obtenemos el valor de los pixeles (INPUT)
                                pixelsValue = getPixelsValue(innerSymbols[i], 2500);
 
                                // Ejecutamos la red
                                output = fann_run(ann, pixelsValue);
 
                                cout << "\nOutput TRIANGLE: " << endl;
                                printf("Prioridad: %f\nResalto: %f\nCirc. giratoria: %f\nCurva dcha: %f\nCurva izda: %f\nCurvas peligrosas izda: %f\nCurvas peligrosas dcha: %f\nNiños: %f\nCiclistas: %f\n\n", 
					output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]);
				
				// Obtenemos el output mayor
				maxSignal = max(max(max(max(max(max(max(max(output[0], output[1]), output[2]), output[3]), output[4]), output[5]), output[6]), output[7]), output[8]);				
				if (maxSignal < 0.8){
					signalsDef[i] = NADA;
					break;
				}

				if(maxSignal == output[0])
					signalsDef[i] = PRIORIDAD;
					
				if(maxSignal == output[1])
					signalsDef[i] = RESALTO;
					
				if(maxSignal == output[2])
					signalsDef[i] = INTERSECCION_GIRATORIA;
					
				if(maxSignal == output[3])
					signalsDef[i] = CURVA_DCHA;
					
				if(maxSignal == output[4])
					signalsDef[i] = CURVA_IZDA;
					
				if(maxSignal == output[5])
					signalsDef[i] = CURVAS_IZDA;
					
				if(maxSignal == output[6])
					signalsDef[i] = CURVAS_DCHA;
					
				if(maxSignal == output[7])
					signalsDef[i] = NINOS;
					
				if(maxSignal == output[8])
					signalsDef[i] = CICLISTAS;
						
				break;
		
			case TRIANGLE_REVERSE:
				
				//cout << "Es un ceda\n";
			
				// Solo puede ser el ceda
				signalsDef[i] = SIGNAL_CEDA;
				break;
		}	
	}
}

// Guarda la imagen como entrenamiento
void Reconocedor::savePixelsOnFile(const char * filename){
	
	cout << "Saving the train on file!\n";

	// Comprobamos que el objeto esta inicializado
	this->srcEmpty();

	// Variables
	FILE * trainFile = fopen(filename, "a");
	Scalar intensity;
	double pixVal; // Aunque la red sea de doubles, al escribir da igual usar double o int porque son numeros exactos
	int pixels = 0;
	double * output = (double *)malloc(9 * sizeof(double));
	
	// Leemos la imagen y guardamos el valor de los pixeles
	for (int y = 0; y < src.size().height; y++){
		for (int x = 0; x < src.size().width; x++){
			intensity = src.at<uchar>(Point(x, y));
			pixVal = intensity[0];
			//printf("(%d, %d)", x, y);
			cout << pixVal << " ";
			fprintf(trainFile, "%f ", pixVal);
			pixels++;
		}
	}
	cout << endl << "Output: " << endl;
	for (int i = 0; i < 9; i++){
		cin >> output[i];
	}
	//fprintf(trainFile, "\n\n%f %f %f %f %f %f %f %f %f\n\n", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8]); // TRIANGLE
	//fprintf(trainFile, "\n\n%f %f %f %f\n\n", output[0], output[1], output[2], output[3]); // CIRCLE
	//fprintf(trainFile, "\n\n%f %f %f %f %f\n\n", output[0], output[1], output[2], output[3], output[4]); // CIRCLE FILLED
	fclose(trainFile);
	cout << "Pixels: " << pixels << endl;
}

// Guarda las señales para el entrenamiento
void Reconocedor::save2train(const char * filename1, const char * filename2){
		
	cout << "Save2train activated!\n";

	Mat imageBox;
	
	// Comprobamos si no esta vacio el reconocedor
	this->srcEmpty();

	// Detectamos la señal que queramos guardar para entrenar
	this->signalDetection();

	// Obtenemos la parte de dentro de la señal		
	this->getInnerSymbol();

	// Ahora tenemos que obtener la imagen de la señal y guardarla en blanco y negro
	for(unsigned int i = 0; i < (numSignalsRed + numSignalsBlue); i = i + 2){
		
		// Le bajamos un poco la calidad
		GaussianBlur(innerSymbols[i], imageBox, Size(3,3), 3, 3);
		imwrite(filename1, innerSymbols[i]);
		imwrite(filename2, imageBox);
	}
}


// Detecta las señales y las guarda en signals
void Reconocedor::signalDetection(){

	// Comprobamos que esta inicializado
	if (src.empty()){ cout << "Object isn't initialized!\n"; exit(-1); }

	// Variables
	Mat dst, maskRed, maskBlue;
	vector<Vec4i> hierarchyRed;
	vector<Vec4i> hierarchyBlue;
	vector<vector<Point> > contoursRed;
	vector<vector<Point> > contoursBlue;
	Size sz = src.size();


	// Resizing
        if (sz.height > 2000 && sz.width > 2000){
                cout << "Resizing...";
                resize(src, src, Size(sz.width/2, sz.height/2)); // Resize the image
                cout << "DONE\n";
        }


	// Mejoramos el contraste de la imagen para su reconocimiento
	/*cvtColor(src, dst, CV_BGR2YCrCb);
	split(dst, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, dst);
	cvtColor(dst, dst, CV_YCrCb2BGR);
	imshow("Source", src);
	imshow("Equalized", dst);
	cout << "Equalized...DONE\n";*/
	
	// Pasamos a HSV para mejorar deteccion
	//cvtColor(src, src, COLOR_BGR2HSV);

	// Obtenemos los pixeles de la imagen entre esos rangos (ROJOS)
	/*
	NOTA: Mejor coger mas posibles señales que no captarlas
	*/
	/*
	inRange(src, Scalar(0,53,185,0), Scalar(15,255,255,0), maskRed0);
	inRange(src, Scalar(165,53,185,0), Scalar(180,255,255,0), maskRed1);
	bitwise_or(maskRed0, maskRed1, maskRed);
	cvtColor(src, src, COLOR_HSV2BGR); // Devuelve la imagen a su estado normal en BRG
	*/
	inRange(src, Scalar(0,0,120), Scalar(130,130,255), maskRed);
	inRange(src, Scalar(120,0,0), Scalar(255,120,120), maskBlue);
	//inRange(src, Scalar(90, 0, 0), Scalar(255, 120, 120), maskRed); // Para detectar señales en grises
	//inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), maskBlue);
	//imshow("MaskRed(normal)", maskRed);
	//imshow("MaskBlue(normal)", maskBlue);
	

	// Limpiamos el ruido con erode y dilate (procesamiento morfológico)
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1,-1));
	erode(maskRed, maskRed, element);
	erode(maskBlue, maskBlue, element);
	dilate(maskBlue, maskBlue, element);
	dilate(maskRed, maskRed, element);
	//imshow("MaskRed(filtered)", maskRed);
	//imshow("MaskBlue(filtered)", maskBlue);

	// Obtenemos los contornos de la imagen mask filtrada
	Mat maskRed_copy = maskRed.clone();
	Mat maskBlue_copy = maskBlue.clone();
	findContours(maskRed_copy, contoursRed, hierarchyRed, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
	findContours(maskBlue_copy, contoursBlue, hierarchyBlue, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	// Aproximamos los contornos a poligonos y dibujamos un cuadrado a su alrededor
	vector<vector<Point> > contours_poly_red(contoursRed.size());
	vector<vector<Point> > contours_poly_blue(contoursBlue.size());
	vector<Rect> boundRectRed(contoursRed.size());
	vector<Rect> boundRectBlue(contoursBlue.size());	
	// Rojo
	for(unsigned int i = 0; i < contoursRed.size(); i++){
		approxPolyDP(Mat(contoursRed[i]), contours_poly_red[i], 3, true);
		boundRectRed[i] = boundingRect(Mat(contours_poly_red[i]));
	}
	// Azul
	for(unsigned int i = 0; i < contoursBlue.size(); i++){
		approxPolyDP(Mat(contoursBlue[i]), contours_poly_blue[i], 3, true);
		boundRectBlue[i] = boundingRect(Mat(contours_poly_blue[i]));
	}


	// Dibujamos los contornos con sus rectangulos
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	Mat src_copy = src.clone();
	// Rojo
	for(unsigned int i = 0; i < contoursRed.size(); i++){
		// Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); // Uncomment para colorines
		Scalar colorRed = Scalar(0, 0, 255);
		drawContours(drawing, contours_poly_red, i, colorRed, 1, 8, hierarchyRed, 0, Point());
		rectangle(src_copy, boundRectRed[i].tl(), boundRectRed[i].br(), colorRed, 2, 8, 0);
		rectangle(drawing, boundRectRed[i].tl(), boundRectRed[i].br(), colorRed, 2, 8, 0);
	}
	// Azul
	for(unsigned int i = 0; i< contoursBlue.size(); i++){
		Scalar colorBlue = Scalar(255, 0, 0);
		drawContours(drawing, contours_poly_blue, i, colorBlue, 1, 8, hierarchyBlue, 0, Point());
		rectangle(src_copy, boundRectBlue[i].tl(), boundRectBlue[i].br(), colorBlue, 2, 8, 0);
		rectangle(drawing, boundRectBlue[i].tl(), boundRectBlue[i].br(), colorBlue, 2, 8, 0);
	}
	//imshow("Bounding box", drawing);
	//imshow("Detected possible signals", src_copy);
	//waitKey(0);

	// Comprobamos las posibles señales
	// Variables
	unsigned int semejanza;
	unsigned int indice = 0;
	Mat filterCircle = imread("./filters/filterCircle.png", IMREAD_GRAYSCALE); // Señales prohibicion (rojas)
	Mat filterCircleFilled = imread("./filters/filterCircleFilled.png", IMREAD_GRAYSCALE); // Señales obligacion (azules)
	Mat filterForbidden = imread("./filters/filterForbidden.png", IMREAD_GRAYSCALE); // Señales prohibido (rojas)
	Mat filterStop = imread("./filters/filterStop.png", IMREAD_GRAYSCALE); // Señales stop (rojas)
	Mat filterStop2 = imread("./filters/filterStop2.png", IMREAD_GRAYSCALE);
	Mat filterTriangle = imread("./filters/filterTriangle.png", IMREAD_GRAYSCALE); // Señales peligro (rojas)
	Mat filterTriangleReverse = imread("./filters/filterTriangleReverse.png", IMREAD_GRAYSCALE); // Señales ceda (rojas)
	signals.resize(contoursRed.size() + contoursBlue.size()); // Contiene todas las señales, rojas y azules
	signalsType = (unsigned int *)malloc(contoursRed.size() * sizeof(int) + contoursBlue.size() * sizeof(int));
	goodContours.resize(contoursRed.size() + contoursBlue.size()); 
	goodBoundRect.resize(boundRectRed.size() + boundRectBlue.size());
	// Comprobamos las posibles señales rojas
	Mat croppedRed;
	//cout << "Comprobando rojas\n";
	for(unsigned int i = 0; i < contoursRed.size(); i++){
	
		// Obtenemos la parte roja detectada y la normalizamos a 50x50
		croppedRed = maskRed(boundRectRed[i]).clone(); // Parte en binario roja de la posible señal
		resize(croppedRed, croppedRed, Size(50,50));
		//imshow("Cropped Red", croppedRed);
	
		// Vemos cuanto se parece al patron
		unsigned int signal;
		for (unsigned int z = 0; z < 5; z++){
			unsigned int semejanza2;
			switch(z){
				case 0:
					semejanza = matchImages(croppedRed, filterCircle);
					signal = CIRCLE;	
					break;
				case 1:
					semejanza2 = matchImages(croppedRed, filterForbidden);
					semejanza = max(semejanza2, semejanza);
					if (semejanza == semejanza2)
						signal = FORBIDDEN;
					break;
				case 2: 
					semejanza2 = matchImages(croppedRed, filterStop);
					semejanza = max(semejanza2, semejanza);
					if (semejanza == semejanza2)
						signal = STOP;
					semejanza2 = matchImages(croppedRed, filterStop2);
					semejanza = max(semejanza2, semejanza);
					if (semejanza == semejanza2)
						signal = STOP;
					break;
				case 3: 
					semejanza2 = matchImages(croppedRed, filterTriangle);
					semejanza = max(semejanza2, semejanza);
					if (semejanza == semejanza2)
						signal = TRIANGLE;
					break;
				case 4: 
					semejanza2 = matchImages(croppedRed, filterTriangleReverse);
					semejanza = max(semejanza2, semejanza);
					if (semejanza2 == semejanza)
						signal = TRIANGLE_REVERSE;
					break;
			}
		}
		//cout << "Semejanza: " << semejanza << endl;
		//cout << "Signal: " << signal << endl;
			
		// Guardamos en el array aquellas señales que son validas
		if (semejanza > 85){
			//cout << "Semejanza: " << semejanza << endl;
			//cout << "Signal: " << signal << endl;
			//imshow("Red signal", croppedRed);
			signalsType[indice] = signal;
			goodBoundRect[indice] = boundRectRed[i];
			goodContours[indice] = contours_poly_red[i];
			signals[indice] = src(boundRectRed[i]).clone(); indice++;
			//signals[indice] = maskRed(boundRectRed[i]).clone(); indice++;
			//waitKey(0);
		}
	}
	numSignalsRed = indice;
	//cout << "Señales reales rojas detectadas: " << numSignalsRed << endl;
	// Comprobamos las posibles señales azules
	//cout << "Comprobando azules!\n";
	Mat croppedBlue;
	for(unsigned int i = 0; i < contoursBlue.size(); i++){

		// Rellenamos la imagen para comparar con el circulo filled (la unica azul en nuestra clasificacion)
		drawContours(maskBlue, contours_poly_blue, i, 255, CV_FILLED, 8, hierarchyBlue, 0, Point());

		// Obtenemos la parte detectada
		croppedBlue = maskBlue(boundRectBlue[i]).clone(); // Parte en binario azul de la posible señal

		// La normalizamos a 50x50
		resize(croppedBlue, croppedBlue, Size(50,50));

		// Vemos cuanto se parece con el patron
		semejanza = matchImages(croppedBlue, filterCircleFilled, 50, 50);
		
		// Guardamos en el array aquellas que son validas
		if (semejanza > 90){
			//cout << "Semejanza: " << semejanza << endl;
			//imshow("Cropped Blue", croppedBlue);
			signalsType[indice] = CIRCLE_FILLED;
			goodBoundRect[indice] = boundRectBlue[i];
			goodContours[indice] = contours_poly_blue[i];
			signals[indice] = src(boundRectBlue[i]).clone(); indice++;
			//signals[indice] = maskBlue(boundRectBlue[i]).clone(); indice++;
			//waitKey(0);
		}
	}
	numSignalsBlue = indice - numSignalsRed;
	//cout << "Señales reales azules detectadas: " << numSignalsBlue << endl;
}

/**********************************************************************************************************/

