CC = g++
DEBUG = -g -Wall
OBJS_IMAGE_PROCESSOR = reconocedor.cpp
LFLAGS_OPENCV = `pkg-config --cflags --libs opencv`
LFLAGS_FANN = -ldoublefann
EXEC=RECONOCEDOR_IMAGENES

all: $(EXEC)

RECONOCEDOR_IMAGENES: reconocedor.o
		$(CC) $(DEBUG) $(LFLAGS_OPENCV) $(LFLAGS_FANN)  $^ -o $@

%.o: %.c
	$(CC) $(DEBUG) $(LFLAGS_OPENCV) $(LFLAGS_FANN) -c $< -o $@

reconocedor.o: MyLibrary.h ClassReconocedor.h

clean: 
	rm -rf *.o

mrproper: clean
	rm -rf $(EXEC)
