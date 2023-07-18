
# Docker

```
docker build -t my_mlflow .
```

```
docker run -d \
--name my_mlflow_instance \
-p 5005:5000 \
-v /Users/junchao.yu/Desktop/MLFlowSample:/mlflow \
my_mlflow
```

```
docker exec -it my_mlflow_instance bash

docker container list

docker stop my_mlflow_instance 
docker container rm my_mlflow_instance
docker image rm my_mlflow
```
