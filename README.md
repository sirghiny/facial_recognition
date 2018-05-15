## Facial Recognition
***

### Set-Up

Clone the repository at: 

	 https://github.com/sirghiny/facial_recognition

Create a virtual environment and activate it:
	
	$ virtualenv venv
	$ source venv/bin/activate


Install all requirements:

	 $ pip install -r requirements.txt


To generate necessary data, run: 

	$ python prep_data.py

To create and train the network, run:

	$ python network.py
