# 졸업프로젝트: MRC를 활용한 뉴스 OpenQnA 서비스

## 구성원: <br/>
### 원성혁(팀장), 고원재, 황인권 <br/>

## 역할: <br/>
* 원성혁: 팀장 및 머신러닝 모델링 <br/>
* 고원재: 크롤링 및 데이터구축 <br/>
* 황인권: 데이터전처리 및 Front-End 개발

## 1. 개요
MRC는 검색엔진이나 문서를 활용한 자연어처리에서 굉장히 각광받는 기술이다. 우리 팀은
딥러닝을 활용해 MRC를 직접 학습시키고 실생활의 문제를 해결하는데 적용하고 싶었다.
뉴스기사에서 빠르게 정보를 얻고 싶은 사람들을 위해서 질문을 통해서 문단의 내용을
파악하는 서비스를 제공하고자 했다. 우리가 학습할 MRC는 그 역할을 정확히 해줄 것이라
믿고 프로젝트를 진행했다.
개발의 과정은 아래와 같다.
MRC를 사용한 뉴스기사 QnA 서비스를 구축
크롤링을 통해 실시간 데이터셋을 구축
kobigbrid를 활용해 긴 문장 기계독해 모델을 개발
flask를 사용해서 프런트엔드 구축
웹서비스를 통해 뉴스기사에 대한 QnA 서비스를 제공

## 2. 목표
1) BigBird를 QnA 데이터로 학습시켜 문단 안에서 가장 답에 가깝거나 같은 예측 답안을
제공한다.
2) Transformer 모델을 학습시켜 번역서비스를 제공한다.
3) 크롤링한 뉴스 데이터로 MRC와 NMT 서비스를 제공한다.

## 3. 요약
본 작품은 2개의 모델을 사용해 2개의 task를 사용한다. 첫번째 모델은 BigBird이며 MRC
task를 수행한다. 두번째 모델은 Transformer이며 NMT task를 수행한다. 미리 크롤링된
데이터를 선택하여 두가지의 task를 선택적으로 수행한다.

### 1) Crawling
N포털의 6가지 카테고리 전부 크롤링하며 한번 크롤링할시 약 200개의 데이터가
축척됩니다.
### 2) Preprocessing
Positioning embedding 방식으로 모든 토큰의 앞 뒤 position 을 데이터는 갖고
있도록 처리한다.
입력 문장을 tokenizer를 활용해 분할을 하며 masked model 학습방법을 위해
masking한다.
### 3) Training
전처리한 QnA 데이터로 BigBird를 학습시킨다. 또한 NMT 데이터로 Transformer
모델을 학습시킨다.
### 4) Testing
MRC 모델은 KLUE validation 데이터를 가지고 exact_match를 활용하거나
Levenshtein Distance를 사용해 성능 분석을 진행한다.
NMT 모델은 BLUE Score를 측정한다.
### 5) Service
Database와 Front-End를 사용해 뉴스 MRC,NMT 서비스를 제공한다.

## 4. 시스템 디자인
### 1) 전체 시스템 아키텍처
![1](https://user-images.githubusercontent.com/64239673/211190192-fd478b15-05e6-4f20-9971-9468363ccd02.png)

### 2) 모델 디자인

![2](https://user-images.githubusercontent.com/64239673/211190276-b828d27d-010c-4430-98cf-a73d8625d554.png) <br/>
(MRC - Bigbird) <br/>
Bigbird는 기본적으로 BERT와 유사한 masked language model로써 Encoder가 12개 쌓여져
있는 구조이다. 각 문장의 15%정도가 masking 되어 있어 masking된 단어를 예측하며
학습한다. Bigbird와 BERT의 차이는 Bigbird는 sparse-attention 을 사용해 학습한다는
점이다. 이는 BERT의 Sequence의 제곱으로 연산회수가 커지는 것을 피하며 Long
sequence 연산에 최적이다. BERT는 최대 512 token까지 가능한 반면 Bigbird는
4096token까지 가능하다. 프로젝트에 사용하는 모델은 한국어로 pretrained 된 KoBigbird
모델이기 때문에 모델의 용도에 맞게 학습시키고 Finetuning하는 것이 프로젝트의
핵심이다. <br/>

![3](https://user-images.githubusercontent.com/64239673/211190285-88940e51-3019-4c94-ab5f-a950fb35d518.png) <br/>
(NMT - Transformer) <br/>
NMT같은 경우는 pretrained model을 사용하지 않고 transformer을 사용했다. transformer는
seq2seq 구조에서 Encoder와 Decoder만 채용한 모델로 Self-Attention mechanism 이 큰
특징이다. 한-영 번역기를 목적을 했기 때문에 src-tokenizer는 KoBERT tokenizer을,
tgt-tokenizer는 GPT2 를 사용했다. <br/>

## 5. 다이어그램
### 1) 시퀀스 다이어그램
![4](https://user-images.githubusercontent.com/64239673/211190560-7b0f5a17-fea6-4936-930c-456b746ad841.png)

### 2) 클래스 다이어그램
![5](https://user-images.githubusercontent.com/64239673/211190568-aa3838ec-e223-4df4-a04b-f3e5cc8505e9.png)

### 6. 시연
![image](https://user-images.githubusercontent.com/64239673/229745987-aa41e1d7-5017-47e4-bd15-2ffd5d42543e.png)
![image](https://user-images.githubusercontent.com/64239673/229746034-af657a2c-8cfd-476e-b2df-e04bafaf4b3b.png)
![image](https://user-images.githubusercontent.com/64239673/229746074-bdcc6791-323c-4628-b27a-34d8c51ec428.png)
![image](https://user-images.githubusercontent.com/64239673/229746132-aa097b45-b24e-4574-8e08-16c0c4780e50.png)




