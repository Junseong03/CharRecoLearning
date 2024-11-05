#include <stdio.h>

#include "../_MLP/MLP.h"
CMLP MultiLayer;

#define NUM_TRAINING_SET	3
#define NUM_INPUT			9
#define NUM_OUTPUT			3
#define MAX_EPOCH			1000000

/*
int main()
{
	// 신경망 네트워크 구성
	int HiddenNodes[1] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);

	// 학습데이터 준비
	double x[NUM_TRAINING_SET][NUM_INPUT] = { {1, 1, 1,	//ㄱ
											0, 0, 1,
											0, 0, 1},

											{1, 0, 0,	//ㄴ
											1, 0, 0,
											1, 1, 1},

											{1, 1, 1,	//ㄷ
											1, 0, 0,
											1, 1, 1} };

	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {1,0,0}, {0,1,0}, {0,0,1} };

	if (MultiLayer.LoadWeight("weight.txt"))
		printf("기존의 가중치로부터 학습을 시작합니다.\n");
	else
		printf("랜덤 가중치로부터 처음으로 시작됩니다.\n");
	
	printf("20220796 김준성\n");

	// 학습전 결과출력
	for (int n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (int i = 1; i <= NUM_INPUT; i++)
		{
			MultiLayer.pInValue[i] = x[n][i - 1];
		}

		MultiLayer.Forward();

		for (int i = 1; i <= NUM_INPUT; i++)
			printf("%lf ", MultiLayer.pInValue[i]);
		printf("= ");
		for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++)
			printf("%lf ", MultiLayer.pOutValue[out_no]);
		printf("\n");
	}
	getchar();


	// 학습
	printf("\n=>학습을 시작합니다. ******************\n");
	int i, j, p;
	int epoch;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// 입력과 정답을 전달
		for (i = 0; i < NUM_TRAINING_SET; i++)
		{
			// 입력 전달
			for (j = 0; j < NUM_INPUT; j++)
				MultiLayer.pInValue[j + 1] = x[i][j];
			// 정답 전달
			//MultiLayer.pCorrectOutValue[1] = d[i][0]; // ppt 구문 : 오류가 있는 듯함.
			for(p = 1; p <= NUM_OUTPUT; p++)
				MultiLayer.pCorrectOutValue[p] = d[i][p - 1];

			// 출력값 계산
			MultiLayer.Forward();
			// 역전파 학습
			MultiLayer.BackPropagationLearning();

			// 갱신 후에 출력과 에러값을 계산
			MultiLayer.Forward();
			for (p = 1; p <= NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]) * (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

		}

		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	MultiLayer.SaveWeight("weight.txt");
	printf("\n=>학습이 완료되었습니다. ******************\n");
	MultiLayer.SaveWeight("..\\_WEIGHT\\weight.txt");

	// 학습후 결과 출력
	printf("학번: 20220796\n");
	printf("이름: 김준성\n");
	printf("학습후 결과\n");
	for (int n = 0; n < NUM_TRAINING_SET; n++)
	{
		for (int i = 1; i <= NUM_INPUT; i++)
			MultiLayer.pInValue[i] = x[n][i - 1];
		MultiLayer.Forward();

		for (int i = 1; i <= NUM_INPUT; i++)
			printf("%lf ", MultiLayer.pInValue[i]);

		printf("= ");

		for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++)
			printf("%lf ", MultiLayer.pOutValue[out_no]);
		printf("\n");
	}

	return 0;
}
*/


int main()
{
	// 신경망 네트워크 구성
	int HiddenNodes[1] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);

	// 가중치 불러오기
	if (MultiLayer.LoadWeight("..\\_WEIGHT\\weight.txt") == true)
		printf("가중치를 읽었습니다.\n");
	else
	{
		printf("가중치를 읽을 수 없습니다.\n");
		return 0;
	}

	// 테스트 입력
	int test_input[9] = { 1,0,0,
						1,1,0,
						1,1,1 };

	// 입력 전달
	for (int i = 1; i <= NUM_INPUT; i++)
		MultiLayer.pInValue[i] = test_input[i - 1];

	// 계산 결과
	MultiLayer.Forward();

	printf("학번: 20220796\n");
	printf("이름: 김준성\n");
	printf("테스트 결과\n");

	for (int i = 1; i <= NUM_INPUT; i++)
		printf("%lf ", MultiLayer.pInValue[i]);
	printf("= ");
	for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++)
		printf("%lf ", MultiLayer.pOutValue[out_no]);
	printf("\n");

	return 0;
}
