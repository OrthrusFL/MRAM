# MARM

## Environment

- OS: Ubuntu 18.04.4 LST
- Language: python 3.6.9
- pytorch: 1.0.0
- javalang: 0.11.0

## Dataset
 You need download the method level bug localization dataset of open-source projects 
 (AspectJ, Birt, JDT, SWT, and Tomcat) from [dataset](https://jbox.sjtu.edu.cn/l/J5z6bj).
```
mkdir report
cd report
unrar x dataset.rar
```

You need download the relations data of methods and commits from [here](https://jbox.sjtu.edu.cn/l/45eBpZ).
and put it into  ```./output/project_name/sim/``` or specify the dir yourself in [config.py](./config.py)


## Usage
optional arguments:
  -h, --help            show this help message and exit
  --project {swt,tomcat,aspectj,jdt,birt}
                        specific the target project
  --parse               parse specific code revision
  --prepare             prepare training data and testing data
  --vocab               build vocabulary
  --within              within project prediction
  --cross {swt,tomcat,aspectj,jdt,birt}
                        cross project prediction (the para here indicates the
                        source project)
  --bias                whether to use localized bug reports
  --log LOG             specific file to record log
  --gpu GPU             specify gpu device


### build vocabulary
```
python run.py --vocab
``` 

### parse specific code revision
```
python run.py --parse
```

### prepare data

```
python run.py --prepare
```

### within-project fault localization
```
python run.py --project project_name --within
```

### cross-project fault localization
```
python run.py --project source_project --cross -target_name
```
