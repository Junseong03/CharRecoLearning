#pragma once
#define LEARNING_RATE 0.1
class CMLP {
public:
	CMLP();
	~CMLP();

	//신경망 구조선언을 위한 변수
	int m_iNumInNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;  // 히든레이어의수 (hidden only)
	int m_iNumTotalLayer;     // 전체레이어의 수 (inputlayer+hiddenlayer+outputlayer)
	int* m_NumNodes; // [0]-input node,[1..]-hidden layer,[m_iNumHiddenLayer+1],output layer,정답
	
	double*** m_Weight;               // [시작layer][시작노드][연결노드]
	double** m_NodeOut;      
	double** m_ErrorGradient; // [layer][node]

	double* pCorrectOutValue;   // 정답레이어
	double* pInValue, * pOutValue;                      // 입력레이어,출력레이어


	
	bool Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer);
private:
	void initW();
	double ActivationFunc(double weightsum);
public:
	void Forward();
	void BackPropagationLearning();
	bool SaveWeight(const char* fname);
	bool LoadWeight(const char* fname);
};