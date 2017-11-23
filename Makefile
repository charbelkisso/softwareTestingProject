
all: test clean


test:
	python scipy_test.py

clean:
	rm -f *.pyc
	rm -f *.pyo
	rm -f *.txt

