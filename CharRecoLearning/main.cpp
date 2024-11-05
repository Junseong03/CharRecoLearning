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
	// �Ű�� ��Ʈ��ũ ����
	int HiddenNodes[1] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);

	// �н������� �غ�
	double x[NUM_TRAINING_SET][NUM_INPUT] = { {1, 1, 1,	//��
											0, 0, 1,
											0, 0, 1},

											{1, 0, 0,	//��
											1, 0, 0,
											1, 1, 1},

											{1, 1, 1,	//��
											1, 0, 0,
											1, 1, 1} };

	double d[NUM_TRAINING_SET][NUM_OUTPUT] = { {1,0,0}, {0,1,0}, {0,0,1} };

	if (MultiLayer.LoadWeight("weight.txt"))
		printf("������ ����ġ�κ��� �н��� �����մϴ�.\n");
	else
		printf("���� ����ġ�κ��� ó������ ���۵˴ϴ�.\n");
	
	printf("20220796 ���ؼ�\n");

	// �н��� ������
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


	// �н�
	printf("\n=>�н��� �����մϴ�. ******************\n");
	int i, j, p;
	int epoch;
	double MSE;
	for (epoch = 0; epoch < MAX_EPOCH; epoch++)
	{
		MSE = 0.0;
		// �Է°� ������ ����
		for (i = 0; i < NUM_TRAINING_SET; i++)
		{
			// �Է� ����
			for (j = 0; j < NUM_INPUT; j++)
				MultiLayer.pInValue[j + 1] = x[i][j];
			// ���� ����
			//MultiLayer.pCorrectOutValue[1] = d[i][0]; // ppt ���� : ������ �ִ� ����.
			for(p = 1; p <= NUM_OUTPUT; p++)
				MultiLayer.pCorrectOutValue[p] = d[i][p - 1];

			// ��°� ���
			MultiLayer.Forward();
			// ������ �н�
			MultiLayer.BackPropagationLearning();

			// ���� �Ŀ� ��°� �������� ���
			MultiLayer.Forward();
			for (p = 1; p <= NUM_OUTPUT; p++)
				MSE += (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]) * (MultiLayer.pCorrectOutValue[p] - MultiLayer.pOutValue[p]);

		}

		MSE /= NUM_TRAINING_SET;
		printf("Epoch%d(MSE)=%.15f\n", epoch, MSE);
	}
	MultiLayer.SaveWeight("weight.txt");
	printf("\n=>�н��� �Ϸ�Ǿ����ϴ�. ******************\n");
	MultiLayer.SaveWeight("..\\_WEIGHT\\weight.txt");

	// �н��� ��� ���
	printf("�й�: 20220796\n");
	printf("�̸�: ���ؼ�\n");
	printf("�н��� ���\n");
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
	// �Ű�� ��Ʈ��ũ ����
	int HiddenNodes[1] = { 2 };
	int numofHiddenLayer = 1;
	MultiLayer.Create(NUM_INPUT, HiddenNodes, NUM_OUTPUT, numofHiddenLayer);

	// ����ġ �ҷ�����
	if (MultiLayer.LoadWeight("..\\_WEIGHT\\weight.txt") == true)
		printf("����ġ�� �о����ϴ�.\n");
	else
	{
		printf("����ġ�� ���� �� �����ϴ�.\n");
		return 0;
	}

	// �׽�Ʈ �Է�
	int test_input[9] = { 1,0,0,
						1,1,0,
						1,1,1 };

	// �Է� ����
	for (int i = 1; i <= NUM_INPUT; i++)
		MultiLayer.pInValue[i] = test_input[i - 1];

	// ��� ���
	MultiLayer.Forward();

	printf("�й�: 20220796\n");
	printf("�̸�: ���ؼ�\n");
	printf("�׽�Ʈ ���\n");

	for (int i = 1; i <= NUM_INPUT; i++)
		printf("%lf ", MultiLayer.pInValue[i]);
	printf("= ");
	for (int out_no = 1; out_no <= NUM_OUTPUT; out_no++)
		printf("%lf ", MultiLayer.pOutValue[out_no]);
	printf("\n");

	return 0;
}
