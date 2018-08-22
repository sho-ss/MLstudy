# やる事
## 注意
tensorflowのバージョンを1.7より大きくするとedwardでエラーが発生する。


## imageのbuild

```
docker build -t [image name]:[tag] -f ./Dockerfile .
```

## コンテナ作成

```
docker run -p 8888:8888 -v [このディレクトリへの絶対パス]:/work/ --name [name] -it [image name]:[tag] /bin/bash
```

## 作業はjupyterがおすすめ
コンテナ内で

```
jupyter notebook --ip=0.0.0.0 --allow-root
```
