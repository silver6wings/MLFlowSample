# docker image build -t my_mlflow .
# docker run -p 5005:5000 -d my_mlflow

# start by pulling the python image
FROM python:3.9

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install mlflow==2.4.1

# configure the container to run in an executed manner
ENTRYPOINT [ "mlflow" ]

CMD ["server" ]
