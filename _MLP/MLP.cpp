#include "MLP.h"
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

CMLP::CMLP() {
	int layer;
	m_iNumInNodes = 0;
	m_iNumOutNodes = 0;
	m_NumNodes = NULL;

	m_NodeOut = NULL;
	m_Weight = NULL;
	m_ErrorGradient = NULL;

	pInValue = NULL;
	pOutValue = NULL;
	pCorrectOutValue = NULL;
}
CMLP::~CMLP()
{
	int layer, snode, enode;
	if (m_NodeOut != NULL)
	{
		for (int i = 0; i < (m_iNumTotalLayer + 1); i++)
			free(m_NodeOut[i]);
		free(m_NodeOut);
	}
	if (m_Weight != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++)
		{
			if (m_Weight[layer] != NULL) {
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
					free(m_Weight[layer][snode]);
				free(m_Weight[layer]);
			}
		}
		free(m_Weight);
	}
	if (m_ErrorGradient != NULL)
	{
		for (layer = 0; layer < (m_iNumTotalLayer); layer++)
			free(m_ErrorGradient[layer]);
		free(m_ErrorGradient);
	}
	if (m_NumNodes != NULL)
		free(m_NumNodes);
}



bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int numHiddenLayer)
{
	int layer, snode, enode;
	m_iNumInNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = numHiddenLayer;                // 입력,출력층 제외 
	m_iNumTotalLayer = numHiddenLayer + 2; // 히든+입력+출력

	//m_iNumNodes를 위한 메모리할당
	m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));    // 정답(+1) 

	m_NumNodes[0] = m_iNumInNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[1 + layer] = pHiddenNode[layer];

	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;    // 출력층 노드수 
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;         // 정답 노드수

	// 각노드별 출력메모리할당=[layerno][nodeno]
	// 입력:m_NodeOut[0][],출력m_NodeOut[m_iNumTotalLayer-1][] 
	// 정답:m_NodeOut[m_iNumTotalLayer][]

	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));              // 정답(+1)
	
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));    // 바이어스를 위해 +1
	
	// 정답(출력 노드와 같은 개수,바이어스가 필요없지만 첨자는 1에서 n까지)
	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer - 1] + 1) * sizeof(double));
	
	// 가중치 메모리할당 m_Weight[시작layer][시작노드][연결노드]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		m_Weight[layer] = (double**)malloc((m_NumNodes[layer] + 1) * sizeof(double*));       // 바이어스(+1) 
		for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
			m_Weight[layer][snode] = (double*)malloc((m_NumNodes[layer + 1] + 1) * sizeof(double));  // 다음레이어 의 노드수
	}
	pInValue = m_NodeOut[0];
	pOutValue = m_NodeOut[m_iNumTotalLayer - 1];
	pCorrectOutValue = m_NodeOut[m_iNumTotalLayer];

	initW();
	// 바이어스를 위한 출력값=1
	for (layer = 0; layer < m_iNumTotalLayer + 1; layer++)
	{
		m_NodeOut[layer][0] = 1;
	}
	return false;
}


void CMLP::initW()
{
	int layer, snode, enode;
	srand(time(NULL));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (snode = 0; snode <= m_NumNodes[layer]; snode++)           // for 바이어스를 위해 0부터 
		{
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) // 다음 레이어의 노드수 
			{
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;     // -0.5~0.5 
			}
		}
	}
}


double CMLP::ActivationFunc(double weightsum)
{
	//if (weightsum > 0) return 1.0;
	//else return 0.0;
	//시그모이드 
	return 1.0 / (1.0 + exp(-weightsum));
}


void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;// 가중합
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			wsum = 0.0; // 노드별 가중합
			wsum += m_Weight[layer][0][enode] * 1;
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				//바이어스
				wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
			}
			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}
}


void CMLP::BackPropagationLearning()
{
	int layer;
	// 에러경사를 위한 메모리 할당 
	if (m_ErrorGradient == NULL)
	{
		// 각노드별 출력메모리할당=m_ErrorGrident[layerno][nodeno]
		// 입력:m_ErrorGradient[0][],출력m_ErrorGradient[m_iNumTotalLayer-1][]
		m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));       // 
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));            // 바이어스(0)를 위해 +1
	}
	int snode, enode, node;
	// 출력층error경사계산
	for (node = 1; node <= m_iNumOutNodes; node++)
	{
		m_ErrorGradient[m_iNumTotalLayer - 1][node] =
			(pCorrectOutValue[node] - m_NodeOut[m_iNumTotalLayer - 1][node])
			* m_NodeOut[m_iNumTotalLayer - 1][node] * (1 - m_NodeOut[m_iNumTotalLayer - 1][node]);
	}
	// error경사계산
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (snode = 1; snode <= m_NumNodes[layer]; snode++)
		{
			m_ErrorGradient[layer][snode] = 0.0;
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
			{
				m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
			}
			m_ErrorGradient[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
		}
	}
	// 가중치갱신
	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			m_Weight[layer][0][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * 1);// 바이어스 
			for (snode = 1; snode <= m_NumNodes[layer]; snode++)
			{
				m_Weight[layer][snode][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * m_NodeOut[layer][snode]);
			}
		}
	}
}


bool CMLP::SaveWeight(const char* fname)
{
	int layer, snode, enode;
	FILE* fp;

	if ((fp = fopen(fname, "wt")) == NULL)
		return false;

	// 입력노드수 히든레이어수 출력노드수
	fprintf(fp, "%d %d %d\n", m_iNumInNodes, m_iNumHiddenLayer, m_iNumOutNodes);

	// node_layer0 node_layer1 node_layer2...
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fprintf(fp, "%d ", m_NumNodes[layer]);
	}
	fprintf(fp, "\n");

	// save weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode <= m_NumNodes[layer]; snode++) // 바이어스를 위해 0부터
			{
				fprintf(fp, "%.9lf ", m_Weight[layer][snode][enode]);
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	return true;
}

bool CMLP::LoadWeight(const char* fname)
{
	int layer, snode, enode;
	FILE* fp;

	if ((fp = fopen(fname, "rt")) == NULL)
		return false;

	// 입력노드수 히든레이어수 출력노드수
	fscanf(fp, "%d %d %d", &m_iNumInNodes, &m_iNumHiddenLayer, &m_iNumOutNodes);

	// node_layer0 node_layer1 node_layer2...
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
	{
		fscanf(fp, "%d ", &m_NumNodes[layer]);
	}

	// load weight
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)
	{
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++)
		{
			for (snode = 0; snode <= m_NumNodes[layer]; snode++) // 바이어스를 위해 0부터
			{
				fscanf(fp, "%lf ", &m_Weight[layer][snode][enode]);
			}
		}
	}
	fclose(fp);

	return true;
}

