face: FaceDetect.o SkinDetect.o MoveDetect.o tool.o config.o Classifier.o main.cpp/
	g++ main.cpp FaceDetect.o SkinDetect.o MoveDetect.o tool.o config.o Classifier.o -o face `pkg-config --libs opencv`

FaceDetect.o: FaceDetect.cpp FaceDetect.h /
	g++ FaceDetect.cpp -c

SkinDetect.o: SkinDetect.cpp SkinDetect.h /
	g++ SkinDetect.cpp -c

MoveDetect.o: MoveDetect.cpp MoveDetect.h /
	g++ MoveDetect.cpp -c

tool.o: tool.cpp tool.h /
	g++ tool.cpp -c

config.o: config.h config.cpp /
	g++ config.cpp -c

Classifier.o: Classifier.h Classifier.cpp /
	g++ Classifier.cpp -c

clean: /
	rm *.o face
