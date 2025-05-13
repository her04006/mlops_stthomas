# mlops_homework5

### To Install
Create a virtual env
```
python3 -m venv venv
```

Activate the virtual env
```
source ./venv/bin/activate
```

Install the requirements
```
pip install -r requirements.txt
```

### Run the server
```
uvicorn --host 0.0.0.0 src.main:app --reload
```

To view the swagger documentation navigate to the ipaddress of the server 
```
<IP ADDRESS>:8000/docs
```

If you don't add the /docs part then you will not see the swagger documentation.



Run tests:

```bash
pytest tests/api/ -v
```

```bash
pytest tests/retriever/ -v
```