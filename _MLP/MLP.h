#pragma once
#define LEARNING_RATE 0.1
class CMLP {
public:
	CMLP();
	~CMLP();

	//�Ű�� ���������� ���� ����
	int m_iNumInNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;  // ���緹�̾��Ǽ� (hidden only)
	int m_iNumTotalLayer;     // ��ü���̾��� �� (inputlayer+hiddenlayer+outputlayer)
	int* m_NumNodes; // [0]-input node,[1..]-hidden layer,[m_iNumHiddenLayer+1],output layer,����
	
	double*** m_Weight;               // [����layer][���۳��][������]
	double** m_NodeOut;      
	double** m_ErrorGradient; // [layer][node]

	double* pCorrectOutValue;   // ���䷹�̾�
	double* pInValue, * pOutValue;                      // �Է·��̾�,��·��̾�


	
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