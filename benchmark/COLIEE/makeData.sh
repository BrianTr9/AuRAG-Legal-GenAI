#!/bin/bash
wget https://laws.e-gov.go.jp/data/Act/129AC0000000089/618417_1/129AC0000000089_20240524_506AC0000000033_xml.zip
unzip 129AC0000000089_20240524_506AC0000000033_xml.zip
# Please use python 3.10 or higher for avoiding the TypeError
python3 makeArticleText.py 129AC0000000089_20240524_506AC0000000033.xml civil.xml
python3 makeEntailment.py civil.xml simple/simple_H18_jp.xml train/riteval_H18_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H19_jp.xml train/riteval_H19_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H20_jp.xml train/riteval_H20_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H21_jp.xml train/riteval_H21_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H22_jp.xml train/riteval_H22_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H23_jp.xml train/riteval_H23_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H24_jp.xml train/riteval_H24_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H25_jp.xml train/riteval_H25_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H26_jp.xml train/riteval_H26_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H27_jp.xml train/riteval_H27_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H28_jp.xml train/riteval_H28_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H29_jp.xml train/riteval_H29_jp.xml
python3 makeEntailment.py civil.xml simple/simple_H30_jp.xml train/riteval_H30_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R01_jp.xml train/riteval_R01_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R02_jp.xml train/riteval_R02_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R03_jp.xml train/riteval_R03_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R04_jp.xml train/riteval_R04_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R05_jp.xml train/riteval_R05_jp.xml
python3 makeEntailment.py civil.xml simple/simple_R06_jp.xml train/riteval_R06_jp.xml
