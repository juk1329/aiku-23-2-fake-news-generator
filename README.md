#🎂 aiku-23-2-fake-news-generator
- - - 
 2023년 겨울학기 AIKU 활동으로 진행한 프로젝트입니다
#**소개**
- - - 
동일한 사실을 바탕으로, 더 자극적인 단어를 선택 및 추가하여 만드는 이른바 '낚시성(어그로성) 뉴스 기사'를 생성하는 프로젝트입니다. 비낚시성(일반 기사문) 텍스트를 넣으면, 낚시성 뉴스의 text style을 반영하여 낚시성 기사 제목을 생성해주는 것을 목표로 하는 프로젝트 입니다. 

#**문제 정의**
- - -
가짜 뉴스와 진짜 뉴스를 구별하는 의미적 기준이 있을 것이라고 판단하였고, 기존의 진짜 뉴스 - 가짜 뉴스 데이터 분석을 통해 의미적 기준을 찾으면 이를 통한 학습으로 text-style transfer를 시도해볼 수 있다고 판단하였습니다. 따라서 "unpaired text data를 바탕으로 한 가짜 뉴스 도메인으로의 text-style transfer"를 본 프로젝트의 목표로 정의하였습니다.

#**데이터셋**
- - -
NER 모델 학습 :   
한국해양대학교 NER 데이터셋   
유명인 이름 데이터셋 (from.Namuwiki)   
AI-Hub 낚시성 기사 탐지 데이터셋   
(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71338)    

#**Models**
- - - 
gogamza/kobart-summarization (https://huggingface.co/gogamza/kobart-summarization)


