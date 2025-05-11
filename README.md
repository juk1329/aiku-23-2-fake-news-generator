# 팀 어그로 (가짜 뉴스 생성하기)

📢 2023년 2학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

🥕🥕🥕🥕🥕🥕🥕🥕🥕🥕

## 소개
교묘하게 특정 표현만 바꾸어 사실을 왜곡하는 이른바 '어그로 뉴스'에 대한 문제의식으로 시작된 프로젝트입니다. 어그로성 뉴스와 원본 뉴스 사이의 style을 딥러닝 모델링을 통해 찾아내고, 이를 학습하여 원본 뉴스 데이터를 input하면 가짜 뉴스 스타일의 제목으로 Output 해주는 모델을 구축한 프로젝트입니다. 

## 방법론
AI-Hub의 낚시성 뉴스 데이터셋과 직접 구축한 유명인사/기업명 데이터셋을 사용해 학습했습니다. 모델링은 크게 두 단계로 진행했습니다.
1) KCElectra 학습으로 NER 2) 1)의 결과에 PER, LOC, DAT에 해당하는 부분 스페셜 토큰 처리

어그로성 기사와 원본 기사의 의미적 차이가 많이 나타나는 분포는 주로 체언인 PER, LOC, DAT에서 기인한다는 것을 확인하여 위와 같은 방법론을 채택했습니다. 

## 환경 설정
Google Colab Pro
Python 3.10.12
Tesla V100

## 사용 방법

python3 train.py

## 예시 결과

<img width="1185" alt="image" src="https://github.com/AIKU-Official/aiku-23-2-fake-news-generator/assets/137471403/bcd54bb0-627e-43ff-9555-bed46f64e88f">
<img width="992" alt="image" src="https://github.com/AIKU-Official/aiku-23-2-fake-news-generator/assets/137471403/b99e64fa-9b59-42cd-b620-03f16d3138aa">

<img width="1066" alt="image" src="https://github.com/AIKU-Official/aiku-23-2-fake-news-generator/assets/137471403/d6daa1b5-7d47-4996-aefe-4f1154b8d276">
<img width="840" alt="image" src="https://github.com/AIKU-Official/aiku-23-2-fake-news-generator/assets/137471403/04ede43e-5d8b-4f39-9fb4-79f7d1426a22">

## 팀원


- [김예랑](https://github.com/01tilinfinity): (문제 정의, 학습 코드 작성, 논문 리서치)
- [박정규](LLM-Innovators-Challenge): (코드 작성, 데이터 정제, 실험 진행)
- [임주원](홍길동의 github link): (데이터 정제, 실험 진행, 논문 리서치)
- [이정은](홍길동의 github link): (모델 서칭, 실험 진행, 코드 작성)
