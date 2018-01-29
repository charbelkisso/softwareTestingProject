
all: test clean


test:
	python Main.py

clean:
	rm -f *.pyc
	rm -f *.pyo
	rm -f *.txt

